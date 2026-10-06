/**
 *
 * @file pzlaswp.c
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zlaswp parallel algorithm for row permutation.
 *
 * @version 1.4.0
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2025-12-19
 * @precisions normal z -> s d c
 *
 */
#include "control/common.h"

#define A(m,n)  A,         m, n
#define Ws(m,n) &(ws->ws), m, n
#define Wu(m,n) ws->Wu,    m, n

/**
 *  Permutation of the panel n at step k
 */
static inline void
chameleon_pzlaswp_panel_permute( struct chameleon_pzlaswp_s *ws,
                                 cham_dir_t                  dir,
                                 CHAM_desc_t                *A,
                                 CHAM_ipiv_t                *ipiv,
                                 int                         k,
                                 int                         n,
                                 RUNTIME_option_t           *options )
{
    CHAM_reduce_t *reduce = &(ws->reduce);
    int m, i, p;
    int tempkm, tempnn, tempmm;
    int withlacpy;

    tempkm = A->get_blkdim( A, k, DIM_m, A->m );
    tempnn = A->get_blkdim( A, n, DIM_n, A->n );

    /* Extract selected rows into U */
    withlacpy = options->withlacpy;
    options->withlacpy = 1;
    CHAMELEON_FOREACH_RANK( reduce, A->myrank, i, p ) {
        INSERT_TASK_zlacpy( options, ChamUpperLower, tempkm, tempnn,
                            A(k, n), Wu(p, n) );
    }
    options->withlacpy = withlacpy;

    /* Each process gathers the rows of its own tiles */
    p = reduce->replicated ? A->get_rankof( A, k, n ) : A->myrank;
    INSERT_TASK_zlaswp_get( options, ChamLeft, dir, k*A->mb, tempkm, tempnn, tempkm,
                            ipiv, k, A(k, n), Wu(p, n) );

    for ( m = k + 1; m < A->mt; m++ ) {
        tempmm = A->get_blkdim( A, m, DIM_m, A->m );
        p = reduce->replicated ? A->get_rankof( A, m, n ) : A->myrank;
        /* Extract selected rows into A(k, n) */
        INSERT_TASK_zlaswp_get( options, ChamLeft, dir, m*A->mb, tempmm, tempnn, tempkm,
                                ipiv, k, A(m, n), Wu(p, n) );
        /* Copy rows from A(k,n) into their final position */
        INSERT_TASK_zlaswp_set( options, ChamLeft, dir, m*A->mb, tempmm, tempnn, tempkm,
                                ipiv, k, A(k, n), A(m, n) );
    }

#if defined(CHAMELEON_USE_MPI)
    if ( ws->allreduce ) {
        INSERT_TASK_zperm_allreduce( options, dir, A, Wu(A->myrank, n), ipiv, k, k, n, ws );
    }
    else {
        INSERT_TASK_zperm_reduce( options, dir, A(k, n), ipiv, k, Wu(A->myrank, n), ws, A->myrank, n );
    }
#endif
}

/**
 *  Permutation of the panel n at step k
 *  This version batched sets of tasks together (one batch for the set, one for
 *  the set)
 */
static inline void
chameleon_pzlaswp_panel_permute_batched( struct chameleon_pzlaswp_s *ws,
                                         cham_dir_t                  dir,
                                         CHAM_desc_t                *A,
                                         CHAM_ipiv_t                *ipiv,
                                         int                         k,
                                         int                         n,
                                         RUNTIME_option_t           *options )
{
    CHAM_reduce_t *reduce = &(ws->reduce);
    int m, i, p;
    int tempkm, tempmm, tempnn;
    int withlacpy;

    void **clargs = malloc( sizeof(char *) );
    *clargs = NULL;

    tempkm = A->get_blkdim( A, k, DIM_m, A->m );
    tempnn = A->get_blkdim( A, n, DIM_n, A->n );

    /* Extract selected rows into U */
    withlacpy = options->withlacpy;
    options->withlacpy = 1;
    CHAMELEON_FOREACH_RANK( reduce, A->myrank, i, p ) {
        INSERT_TASK_zlacpy( options, ChamUpperLower, tempkm, tempnn,
                            A(k, n), Wu(p, n) );
    }
    options->withlacpy = withlacpy;

    /* Each process gathers the rows of its own tiles */
    for ( m = k; m < A->mt; m++ ) {
        p = reduce->replicated ? A->get_rankof( A, m, n ) : A->myrank;
        tempmm = A->get_blkdim( A, m, DIM_m, A->m );
        INSERT_TASK_zlaswp_get_batched( options, ChamLeft, dir, m*A->mb, tempmm, tempnn, tempkm, (void *)ws, ipiv, k,
                                        A(m, n), Wu(p, n), clargs );
    }
    CHAMELEON_FOREACH_RANK( reduce, A->myrank, i, p ) {
        INSERT_TASK_zlaswp_get_batched_flush( options, dir, ipiv, k, Wu(p, n), clargs );
    }

    for ( m = k + 1; m < A->mt; m++ ) {
        tempmm = A->get_blkdim( A, m, DIM_m, A->m );
        INSERT_TASK_zlaswp_set_batched( options, ChamLeft, dir, m*A->mb, tempmm, tempnn, tempkm, (void *)ws, ipiv, k,
                                        A(k, n), A(m, n), clargs );
    }
    INSERT_TASK_zlaswp_set_batched_flush( options, dir, ipiv, k, A(k, n), clargs );

#if defined(CHAMELEON_USE_MPI)
    if ( ws->allreduce ) {
        INSERT_TASK_zperm_allreduce( options, dir, A, Wu(A->myrank, n), ipiv, k, k, n, ws );
    }
    else {
        INSERT_TASK_zperm_reduce( options, dir, A(k, n), ipiv, k, Wu(A->myrank, n), ws, A->myrank, n );
    }
#endif

    free( clargs );
}

void
chameleon_pzlaswp_panel( struct chameleon_pzlaswp_s *ws,
                         cham_bool_t                 inplace,
                         cham_dir_t                  dir,
                         CHAM_desc_t                *A,
                         CHAM_ipiv_t                *ipiv,
                         int                         k,
                         int                         n,
                         RUNTIME_option_t           *options )
{
    const RUNTIME_request_t *request = options->request;
    CHAM_reduce_t           *reduce  = &(ws->reduce);
    int                      tempkm, tempnn, i, p;

#if defined(CHAMELEON_USE_MPI)
    /* Initizalize the list of nodes invovlved in the panel n */
    chameleon_get_proc_involved_in_panelk_2dbc( A, k, n, reduce );

    /* If on the rank who owns the ipiv array */
    if ( A->myrank == ipiv->get_rankof( ipiv, k, k ) ) {
        /* Exchange between all nodes involved the perm array */
        INSERT_TASK_zperm_allreduce_send_perm( options, dir, ipiv, k, A->myrank,
                                               reduce->np_involved, reduce->proc_involved );

        /* Exchange between all nodes involved the invp array */
        INSERT_TASK_zperm_allreduce_send_invp_row( options, dir, ipiv, k, A, k, n );
    }

    /* Bcast the top tile of the panel to all involved nodes */
    if ( A->myrank == chameleon_getrankof_2d( A, k, n ) ) {
        INSERT_TASK_zperm_allreduce_send_A( options, A, k, n, A->myrank,
                                            reduce->np_involved, reduce->proc_involved );
    }

    /* If I'm not involved in the reduction, no need to go further */
    if ( !reduce->involved && !reduce->replicated ) {
        return;
    }
#endif

    /*
     * Perform the permutation on the panel, the final top tile is stored in
     * Wu(...,n) or Ws(...,n) when done depending on the configuration.
     */
    if ( ws->batch_size_swap == 0 ){
        chameleon_pzlaswp_panel_permute( ws, dir, A, ipiv, k, n, options );
    }
    else {
        chameleon_pzlaswp_panel_permute_batched( ws, dir, A, ipiv, k, n, options );
    }

    /*
     * Now, we need to set the final top tile to the right position. Either
     * directly in A (when peforming a standalone swap operation), or in the
     * temporary replicated buffer to perform a replicated trsm (LU
     * factorization for example).
     */
    if ( inplace ) {
        if ( ws->reduce.np_involved == 1 ) {
            /*
             * If we just perform a laswp, A must be equal to Atop, then we copy the
             * final version of the tile to its final location.
             */
            tempkm = A->get_blkdim( A, k, DIM_m, A->m );
            tempnn = A->get_blkdim( A, n, DIM_n, A->n );
            p = reduce->replicated ? chameleon_getrankof_2d( A, k, n ) : A->myrank;
            INSERT_TASK_zlacpy( options, ChamUpperLower, tempkm, tempnn,
                                Wu(p, n), A( k, n ) );
        }
#if defined(CHAMELEON_USE_MPI)
        else {
            if ( reduce->alg_allreduce == ChamStarPUTasks ) {
                /*
                 * Let's copy the final version of A to its final position
                 * (done by the owner of A(k,n), the root of the reduction)
                 */
                p = reduce->replicated ? chameleon_getrankof_2d( A, k, n ) : A->myrank;
                INSERT_TASK_zlaswp_ret( options, Ws(p, n), A(k, n) );
                RUNTIME_perm_flush( options->sequence, p, Ws(p, n) );
            }
        }
#endif
        chameleon_data_flush( options->sequence, A(k, n), request->flush );
    }
#if defined(CHAMELEON_USE_MPI)
    else { /* Outofplace with replication */
        if ( reduce->np_involved != 1 ) {
            CHAMELEON_FOREACH_RANK( reduce, A->myrank, i, p ) {
                if ( reduce->alg_allreduce == ChamStarPUTasks ) {
                    INSERT_TASK_zlaswp_ret( options, Ws(p, n), Wu(p, n) );
                }
                RUNTIME_perm_flush( options->sequence, p, Ws(p, n) );
            }
        }
    }
#endif
    (void)reduce;
    (void)i;
}

void
chameleon_pzlaswp( struct chameleon_pzlaswp_s *ws,
                   cham_dir_t                  dir,
                   CHAM_desc_t                *A,
                   CHAM_ipiv_t                *IPIV,
                   RUNTIME_sequence_t         *sequence,
                   RUNTIME_request_t          *request )
{
    CHAM_context_t   *chamctxt;
    RUNTIME_option_t  options;

    int n, k;

    chamctxt = chameleon_context_self();
    if ( sequence->status != CHAMELEON_SUCCESS ) {
        return;
    }
    RUNTIME_options_init( &options, chamctxt, sequence, request );
    RUNTIME_options_set_taskcolor( &options, CHAMELEON_DAG_COLOR_ALGORITHM( laswp ) );

    assert( A->get_rankof_init == chameleon_getrankof_2d );

    if ( dir == ChamDirForward ) {
        for ( k = 0; k < IPIV->mt; k++ ) {
            for ( n = 0; n < A->nt; n++ ) {
                options.priority = A->nt-n;

                chameleon_pzlaswp_panel( ws, CHAMELEON_TRUE, dir, A, IPIV, k, n, &options );
            }
            RUNTIME_ipiv_flushone( sequence, CHAMIPIV_PERM | CHAMIPIV_INVP, IPIV, k );
        }
    }
    else {
        for ( k = IPIV->mt - 1; k > -1; k-- ) {
            for ( n = 0; n < A->nt; n++ ) {
                options.priority = A->nt-n;
                chameleon_pzlaswp_panel( ws, CHAMELEON_TRUE, dir, A, IPIV, k, n, &options );
            }
            RUNTIME_ipiv_flushone( sequence, CHAMIPIV_PERM | CHAMIPIV_INVP, IPIV, k );
        }
    }
    /* Mark written data for synchronization */
    A->sync = 1;

    RUNTIME_options_finalize( &options, chamctxt );
}
