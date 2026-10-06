/**
 *
 * @file pzplrnk.c
 *
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zplrnk parallel algorithm
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Florent Pruvost
 * @author Lionel Eyraud-Dubois
 * @date 2025-12-19
 * @precisions normal z -> s d c
 *
 */
#include "control/common.h"

#define WA(m, n) WA, m,  n
#define WB(m, n) WB, m,  n
#define C(m, n)  C,  m,  n

/**
 *  chameleon_pzplrnk - Generate a random rank-k matrix by tiles.
 */
static inline void
chameleon_pzplrnk_generic( CHAM_context_t         *chamctxt,
                           int                     K,
                           CHAM_desc_t            *WA,
                           CHAM_desc_t            *WB,
                           CHAM_desc_t            *C,
                           unsigned long long int  seedA,
                           unsigned long long int  seedB,
                           RUNTIME_option_t       *options )
{
    RUNTIME_sequence_t      *sequence = options->sequence;
    const RUNTIME_request_t *request  = options->request;
    CHAMELEON_Complex64_t zbeta;
    int  m, n, k, KT, r, i;
    int  tempmm, tempnn, tempkk;
    int  myrank  = RUNTIME_comm_rank( chamctxt );
    int  repl    = chameleon_replicated_submission( chamctxt );
    /*
     * The workspaces of a tile are indexed by the owner of C(m, n). Without
     * replicated submission, only the local rank is considered, otherwise the
     * initializations are tracked per rank.
     */
    int  nr      = repl ? RUNTIME_comm_size( chamctxt ) : 1;
    int *initA   = malloc( sizeof(int) * nr );
    int *initB   = malloc( sizeof(int) * nr * C->nt );

    KT = (K + C->mb - 1) / C->mb;

    for (k = 0; k < KT; k++) {
        tempkk = k == KT-1 ? K - k * WA->nb : WA->nb;
        zbeta  = k == 0 ? 0. : 1.;

        memset( initB, 0, sizeof(int) * nr * C->nt );

        for (m = 0; m < C->mt; m++) {
            tempmm = C->get_blkdim( C, m, DIM_m, C->m );

            memset( initA, 0, sizeof(int) * nr );

            for (n = 0; n < C->nt; n++) {
                tempnn = C->get_blkdim( C, n, DIM_n, C->n );

                r = C->get_rankof( C(m, n) );
                if ( !repl && (r != myrank) ) {
                    continue;
                }
                i = repl ? r : 0;

                if ( !initA[i] ) {
                    INSERT_TASK_zplrnt(
                        options,
                        tempmm, tempkk, WA(m, r),
                        WA->m, m * WA->mb, k * WA->nb, seedA );
                    initA[i] = 1;
                }
                if ( !initB[i * C->nt + n] ) {
                    INSERT_TASK_zplrnt(
                        options,
                        tempkk, tempnn, WB(r, n),
                        WB->m, k * WB->mb, n * WB->nb, seedB );
                    initB[i * C->nt + n] = 1;
                }

                INSERT_TASK_zgemm(
                    options,
                    ChamNoTrans, ChamNoTrans,
                    tempmm, tempnn, tempkk, C->mb,
                    1.,    WA(m, r),
                           WB(r, n),
                    zbeta, C(m, n));
            }
            for (i = 0; i < nr; i++) {
                if ( initA[i] ) {
                    r = repl ? i : myrank;
                    chameleon_data_flush( sequence, WA(m, r), request->flush );
                }
            }
        }
        for (i = 0; i < nr; i++) {
            r = repl ? i : myrank;
            for (n = 0; n < C->nt; n++) {
                if ( initB[i * C->nt + n] ) {
                    chameleon_data_flush( sequence, WB(r, n), request->flush );
                }
            }
        }
    }

    free( initA );
    free( initB );
    (void)chamctxt;
}

/**
 *  chameleon_pzplrnk - Generate a random rank-k matrix by tiles on a 2dbc grid.
 */
static inline void
chameleon_pzplrnk_2dbc( CHAM_context_t         *chamctxt,
                        int                     K,
                        CHAM_desc_t            *WA,
                        CHAM_desc_t            *WB,
                        CHAM_desc_t            *C,
                        unsigned long long int  seedA,
                        unsigned long long int  seedB,
                        RUNTIME_option_t       *options )
{
    RUNTIME_sequence_t      *sequence = options->sequence;
    const RUNTIME_request_t *request  = options->request;
    CHAMELEON_Complex64_t    zbeta;
    int m, n, k, KT;
    int tempmm, tempnn, tempkk;
    int p, q, myp, myq, pm, qn, pp, qq, repl;

    KT   = (K + C->mb - 1) / C->mb;
    p    = chameleon_desc_datadist_get_iparam( C, 0 );
    q    = chameleon_desc_datadist_get_iparam( C, 1 );
    myp  = C->myrank / q;
    myq  = C->myrank % q;
    repl = chameleon_replicated_submission( chamctxt );

    for (k = 0; k < KT; k++) {
        tempkk = k == KT-1 ? K - k * WA->nb : WA->nb;
        zbeta  = k == 0 ? 0. : 1.;

        /* WB(pp, n) replicates the row k of B on each row of processes */
        for (n = chameleon_2dbc_first( repl, myq, q ); n < C->nt; n += chameleon_2dbc_stride( repl, q )) {
            tempnn = C->get_blkdim( C, n, DIM_n, C->n );

            for (pp = chameleon_2dbc_first( repl, myp, p ); pp < chameleon_2dbc_last( repl, myp, p ); pp++) {
                INSERT_TASK_zplrnt(
                    options,
                    tempkk, tempnn, WB(pp, n),
                    WB->m, k * WB->mb, n * WB->nb, seedB );
            }
        }

        for (m = chameleon_2dbc_first( repl, myp, p ); m < C->mt; m += chameleon_2dbc_stride( repl, p )) {
            tempmm = C->get_blkdim( C, m, DIM_m, C->m );
            pm     = m % p;

            /* WA(m, qq) replicates the column k of A on each column of processes */
            for (qq = chameleon_2dbc_first( repl, myq, q ); qq < chameleon_2dbc_last( repl, myq, q ); qq++) {
                INSERT_TASK_zplrnt(
                    options,
                    tempmm, tempkk, WA(m, qq),
                    WA->m, m * WA->mb, k * WA->nb, seedA );
            }

            for (n = chameleon_2dbc_first( repl, myq, q ); n < C->nt; n += chameleon_2dbc_stride( repl, q )) {
                tempnn = C->get_blkdim( C, n, DIM_n, C->n );
                qn     = n % q;

                INSERT_TASK_zgemm(
                    options,
                    ChamNoTrans, ChamNoTrans,
                    tempmm, tempnn, tempkk, C->mb,
                    1.,    WA(m, qn),
                           WB(pm, n),
                    zbeta,  C(m, n));
            }
            for (qq = chameleon_2dbc_first( repl, myq, q ); qq < chameleon_2dbc_last( repl, myq, q ); qq++) {
                chameleon_data_flush( sequence, WA(m, qq), request->flush );
            }
        }
        for (n = chameleon_2dbc_first( repl, myq, q ); n < C->nt; n += chameleon_2dbc_stride( repl, q )) {
            for (pp = chameleon_2dbc_first( repl, myp, p ); pp < chameleon_2dbc_last( repl, myp, p ); pp++) {
                chameleon_data_flush( sequence, WB(pp, n), request->flush );
            }
        }
    }
    (void)chamctxt;
}

/**
 * Rank-k matrix generator.
 */
void
chameleon_pzplrnk( int                         K,
                   CHAM_desc_t                *C,
                   unsigned long long int      seedA,
                   unsigned long long int      seedB,
                   RUNTIME_sequence_t         *sequence,
                   RUNTIME_request_t          *request )
{
    CHAM_context_t *chamctxt;
    RUNTIME_option_t options;
    CHAM_desc_t WA, WB;
    int P, Q;

    chamctxt = chameleon_context_self();
    if (sequence->status != CHAMELEON_SUCCESS) {
        return;
    }
    RUNTIME_options_init( &options, chamctxt, sequence, request );
    RUNTIME_options_set_taskcolor( &options, CHAMELEON_DAG_COLOR_ALGORITHM( plrnk ) );

    P = chameleon_desc_datadist_get_iparam( C, 0 );
    Q = chameleon_desc_datadist_get_iparam( C, 1 );
    if ( ( chamctxt->generic_enabled != CHAMELEON_TRUE )  &&
         ( C->get_rankof_init == chameleon_getrankof_2d ) &&
         ( (chameleon_desc_datadist_get_iparam(C, 0) != 1) ||
           (chameleon_desc_datadist_get_iparam(C, 1) != 1) ) )
    {
        chameleon_desc_init_2dtile( chamctxt, &WA, "PLRNK_WA", ChamComplexDouble,
                                    C->mb, C->nb, C->m, C->nb * Q, P, Q );
        chameleon_desc_init_2dtile( chamctxt, &WB, "PLRNK_WB", ChamComplexDouble,
                                    C->mb, C->nb, C->mb * P, C->n, P, Q );

        chameleon_pzplrnk_2dbc( chamctxt, K, &WA, &WB, C, seedA, seedB, &options );
    }
    else {
        int np = P * Q;
        chameleon_desc_init_2dtile( chamctxt, &WA, "PLRNK_WA", ChamComplexDouble,
                                    C->mb, C->nb, C->m, C->nb * np, 1, np );
        chameleon_desc_init_2dtile( chamctxt, &WB, "PLRNK_WB", ChamComplexDouble,
                                    C->mb, C->nb, C->mb * np, C->n, np, 1 );

        chameleon_pzplrnk_generic( chamctxt, K, &WA, &WB, C, seedA, seedB, &options );
    }
    /* Mark written data for synchronization */
    C->sync = 1;

    RUNTIME_desc_flush( &WA, sequence );
    RUNTIME_desc_flush( &WB, sequence );
    RUNTIME_desc_flush(  C,  sequence );

    chameleon_sequence_wait( chamctxt, sequence );
    chameleon_desc_destroy( &WA );
    chameleon_desc_destroy( &WB );

    RUNTIME_options_finalize( &options, chamctxt );
}
