/**
 *
 * @file parsec/codelet_zperm_reduce.c
 *
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon parsec codelets to do the reduction
 *
 * @version 1.4.0
 * @author Matteo Marcos
 * @date 2026-10-05
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "control/compute_z.h"
#include <coreblas/cblas_wrapper.h>

/*
 * Reductions of the permutation workspaces.
 *
 * Without reduction in DTD, the replicated submission inserts for each
 * involved rank p an initialization task, executed on p, that copies the rows
 * of Wu(p) it owns in ws(p). The workspaces are then merged, in order, into the
 * workspace of the root. The allreduce copies the result back to the other
 * involved ranks.
 */

/* Coordinates of the workspace of rank in the row (left) or column (right) */
static inline void
chameleon_parsec_perm_coord( const CHAM_perm_t *ws, int rank, int Wm, int Wn, int *m, int *n )
{
    if ( ws->side == ChamLeft ) {
        *m = rank;
        *n = Wn;
    }
    else {
        *m = Wm;
        *n = rank;
    }
}

static inline int
CORE_zperm_reduce_init_parsec( parsec_execution_stream_t *context,
                               parsec_task_t             *this_task )
{
    cham_side_t side;
    int k, P, Q, mb, Am, An, me, mrows, ncols, lda;
    int *perm;
    CHAMELEON_Complex64_t *A, *rows;
    void *wsbuf;
    chameleon_parsec_perm_hdr_t *hdr;
    CHAM_laswpws_t lws;
    int i, idx, owner, nindex, A_inc;

    parsec_dtd_unpack_args( this_task, &side, &k, &P, &Q, &mb, &Am, &An, &me, &mrows, &ncols,
                            &perm, &A, &lda, &wsbuf );

    hdr = (chameleon_parsec_perm_hdr_t *)wsbuf;
    hdr->m    = mrows;
    hdr->n    = ncols;
    hdr->side = side;
    chameleon_parsec_perm_load( wsbuf, &lws );
    rows = (CHAMELEON_Complex64_t *)(lws.rows);

    if ( side == ChamLeft ) {
        A_inc = 1;
    }
    else {
        A_inc = lda;
        lda   = 1;
    }

    memset( lws.index, 0xff, sizeof(int) * mrows );
    nindex = 0;
    for ( i = 0; i < k; i++ ) {
        idx   = perm[i] / mb;
        owner = ( side == ChamLeft ) ? (idx % P) * Q + (An  % Q) :
                                       (Am  % P) * Q + (idx % Q);
        if ( owner != me ) {
            continue;
        }
        lws.index[i] = nindex;
        cblas_zcopy( ncols, A    + i      * A_inc, lda,
                            rows + nindex * ncols, 1 );
        nindex++;
    }
    hdr->nindex = nindex;

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

static inline int
CORE_zperm_reduce_merge_parsec( parsec_execution_stream_t *context,
                                parsec_task_t             *this_task )
{
    void *dstbuf, *srcbuf;
    chameleon_parsec_perm_hdr_t *hdst, *hsrc;
    CHAM_laswpws_t dst, src;
    size_t rowsize;
    int i, new_nindex;

    parsec_dtd_unpack_args( this_task, &dstbuf, &srcbuf );

    hdst = (chameleon_parsec_perm_hdr_t *)dstbuf;
    hsrc = (chameleon_parsec_perm_hdr_t *)srcbuf;
    assert( hdst->m == hsrc->m );
    assert( hdst->n == hsrc->n );
    chameleon_parsec_perm_load( dstbuf, &dst );
    chameleon_parsec_perm_load( srcbuf, &src );

    rowsize    = sizeof(CHAMELEON_Complex64_t) * hsrc->n;
    new_nindex = dst.nindex;
    for ( i = 0; i < hsrc->m; i++ ) {
        /* Skip the rows existing in dst, and the ones not owned by src */
        if ( ( dst.index[i] != -1 ) || ( src.index[i] == -1 ) ) {
            continue;
        }
        memcpy( (char *)dst.rows + rowsize * new_nindex,
                (char *)src.rows + rowsize * src.index[i], rowsize );
        dst.index[i] = new_nindex;
        new_nindex++;
    }
    assert( new_nindex <= hdst->m );
    hdst->nindex = new_nindex;

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

static inline int
CORE_zperm_copy_parsec( parsec_execution_stream_t *context,
                        parsec_task_t             *this_task )
{
    size_t size;
    void  *dstbuf, *srcbuf;

    parsec_dtd_unpack_args( this_task, &size, &dstbuf, &srcbuf );
    memcpy( dstbuf, srcbuf, size );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Submit the reduction of the workspaces of the involved ranks into the one of
 * the owner of A(Am, An), and if allreduce is set, copy the result back to the
 * other involved ranks.
 */
void
chameleon_parsec_zperm_reduce_submit( const RUNTIME_option_t *options,
                                      cham_dir_t              dir,
                                      const CHAM_desc_t      *A,
                                      int                     Am,
                                      int                     An,
                                      CHAM_ipiv_t            *ipiv,
                                      int                     ipivk,
                                      const CHAM_desc_t      *Wu,
                                      int                     Wum,
                                      int                     Wun,
                                      CHAM_perm_t            *ws,
                                      int                     Wm,
                                      int                     Wn,
                                      const CHAM_reduce_t    *reduce,
                                      int                     allreduce )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    chameleon_parsec_ipiv_family_t family = ( dir == ChamDirForward ) ? ChamParsecPerm : ChamParsecInvp;
    int  np    = reduce->np_involved;
    int  P     = chameleon_desc_datadist_get_iparam( A, 0 );
    int  Q     = chameleon_desc_datadist_get_iparam( A, 1 );
    int  mb    = ipiv->mb;
    int  root  = A->get_rankof( A, Am, An );
    int  side  = ws->side;
    int  mrows = ( side == ChamLeft ) ? ws->mb : ws->nb;
    int  i, p, m, n, ncols, tempkk;
    int  procs[np];

    if ( np == 1 ) {
        return;
    }

    /* The list of involved processes is not modified */
    memcpy( procs, reduce->proc_involved, np * sizeof(int) );

    tempkk = ( side == ChamLeft ) ? A->get_blkdim( A, ipivk, DIM_m, A->m )
                                  : A->get_blkdim( A, ipivk, DIM_n, A->n );

    for ( i = 0; i < np; i++ ) {
        int um, un;
        p = procs[i];
        chameleon_parsec_perm_coord( ws, p, Wm,  Wn,  &m,  &n  );
        chameleon_parsec_perm_coord( ws, p, Wum, Wun, &um, &un );
        ncols = ( side == ChamLeft ) ? ( ( n == ws->nt - 1 ) ? ws->n - n * ws->nb : ws->nb )
                                     : ( ( m == ws->mt - 1 ) ? ws->m - m * ws->mb : ws->mb );

        parsec_dtd_insert_task(
            PARSEC_dtd_taskpool, CORE_zperm_reduce_init_parsec, options->priority, PARSEC_DEV_CPU, "perm_reduce_init",
            sizeof(cham_side_t), &side,   PARSEC_VALUE,
            sizeof(int),         &tempkk, PARSEC_VALUE,
            sizeof(int),         &P,      PARSEC_VALUE,
            sizeof(int),         &Q,      PARSEC_VALUE,
            sizeof(int),         &mb,     PARSEC_VALUE,
            sizeof(int),         &Am,     PARSEC_VALUE,
            sizeof(int),         &An,     PARSEC_VALUE,
            sizeof(int),         &p,      PARSEC_VALUE,
            sizeof(int),         &mrows,  PARSEC_VALUE,
            sizeof(int),         &ncols,  PARSEC_VALUE,
            PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipiv, family, ipivk ),
                           chameleon_parsec_ipiv_arena( ipiv, family, ipivk ) | PARSEC_INPUT,
            PASSED_BY_REF, RTBLKADDR( Wu, ChamComplexDouble, um, un ),
                           chameleon_parsec_get_arena_index( Wu, um, un ) | PARSEC_INPUT,
            sizeof(int),         &(Wu->get_blktile( Wu, um, un )->ld), PARSEC_VALUE,
            PASSED_BY_REF, chameleon_parsec_perm_tile( options, ws, m, n ),
                           chameleon_parsec_perm_arena( ws ) | PARSEC_OUTPUT | PARSEC_AFFINITY,
            PARSEC_DTD_ARG_END );
    }

    /* Merge in the workspace of the root */
    for ( i = 0; i < np; i++ ) {
        int rm, rn;
        p = procs[i];
        if ( p == root ) {
            continue;
        }
        chameleon_parsec_perm_coord( ws, root, Wm, Wn, &rm, &rn );
        chameleon_parsec_perm_coord( ws, p,    Wm, Wn, &m,  &n  );
        parsec_dtd_insert_task(
            PARSEC_dtd_taskpool, CORE_zperm_reduce_merge_parsec, options->priority, PARSEC_DEV_CPU, "perm_reduce_merge",
            PASSED_BY_REF, chameleon_parsec_perm_tile( options, ws, rm, rn ),
                           chameleon_parsec_perm_arena( ws ) | PARSEC_INOUT | PARSEC_AFFINITY,
            PASSED_BY_REF, chameleon_parsec_perm_tile( options, ws, m, n ),
                           chameleon_parsec_perm_arena( ws ) | PARSEC_INPUT,
            PARSEC_DTD_ARG_END );
    }

    if ( !allreduce ) {
        return;
    }

    /* Copy back the result */
    for ( i = 0; i < np; i++ ) {
        int    rm, rn;
        size_t size = ((chameleon_parsec_vdc_t *)(ws->ws))->size;
        p = procs[i];
        if ( p == root ) {
            continue;
        }
        chameleon_parsec_perm_coord( ws, root, Wm, Wn, &rm, &rn );
        chameleon_parsec_perm_coord( ws, p,    Wm, Wn, &m,  &n  );
        parsec_dtd_insert_task(
            PARSEC_dtd_taskpool, CORE_zperm_copy_parsec, options->priority, PARSEC_DEV_CPU, "perm_copy",
            sizeof(size_t), &size, PARSEC_VALUE,
            PASSED_BY_REF, chameleon_parsec_perm_tile( options, ws, m, n ),
                           chameleon_parsec_perm_arena( ws ) | PARSEC_OUTPUT | PARSEC_AFFINITY,
            PASSED_BY_REF, chameleon_parsec_perm_tile( options, ws, rm, rn ),
                           chameleon_parsec_perm_arena( ws ) | PARSEC_INPUT,
            PARSEC_DTD_ARG_END );
    }
}

/**
 * Reduction of the workspaces in the one of the owner of A(Am, An). With
 * replicated submission, this is called once for all the involved ranks: the
 * rank coordinate of (Wum, Wun) and (Wm, Wn), which designates the calling
 * rank, is replaced by each involved rank.
 */
void
INSERT_TASK_zperm_reduce( const RUNTIME_option_t *options,
                          cham_dir_t              dir,
                          const CHAM_desc_t      *A,
                          int                     Am,
                          int                     An,
                          CHAM_ipiv_t            *ipiv,
                          int                     ipivk,
                          const CHAM_desc_t      *Wu,
                          int                     Wum,
                          int                     Wun,
                          void                   *ws,
                          int                     Wm,
                          int                     Wn )
{
    struct chameleon_pzlaswp_s *tmp = (struct chameleon_pzlaswp_s *)ws;

    chameleon_parsec_zperm_reduce_submit( options, dir, A, Am, An, ipiv, ipivk, Wu, Wum, Wun,
                                          &(tmp->ws), Wm, Wn, &(tmp->reduce), 0 );
}
