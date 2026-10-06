/**
 *
 * @file parsec/codelet_zgetrf_blocked.c
 *
 * @copyright 2023-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zgetrf_blocked Parsec codelets
 *
 * @version 1.4.0
 * @comment Codelets to perform panel factorization with partial pivoting
 *
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2026-10-05
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

/*
 * Panel factorization tasks with partial pivoting.
 *
 * The candidate pivots of a step h are accumulated in the buffer of the rank
 * executing the task (pivot(rank, h)), reset by the first task of the step on
 * this rank. The previous step is read from the buffer of the root (owner of
 * the diagonal tile) where the candidates of all the ranks have been merged
 * by INSERT_TASK_zpivot_allreduce().
 */
static inline int
CORE_zgetrf_blocked_diag_parsec( parsec_execution_stream_t *context,
                                 parsec_task_t             *this_task )
{
    int m, n, h, m0, ib, nb, reset, lda, ldu;
    cham_flttype_t dtyp;
    CHAMELEON_Complex64_t *A, *U;
    int  *ipiv;
    void *nextbuf, *prevbuf;
    CHAM_pivot_t nextpiv, prevpiv;
    chameleon_parsec_pivot_hdr_t *hdr;

    parsec_dtd_unpack_args(
        this_task, &m, &n, &h, &m0, &ib, &nb, &dtyp, &reset,
        &A, &lda, &ipiv, &U, &ldu, &nextbuf, &prevbuf );

    hdr = (chameleon_parsec_pivot_hdr_t *)nextbuf;
    if ( reset ) {
        chameleon_parsec_pivot_reset( nextbuf, nb, dtyp );
    }
    hdr->h        = h;
    hdr->has_diag = 1;

    chameleon_parsec_pivot_load( nextbuf, nb, dtyp, &nextpiv );
    if ( prevbuf != NULL ) {
        chameleon_parsec_pivot_load( prevbuf, nb, dtyp, &prevpiv );
    }

    CORE_zgetrf_panel_diag( m, n, h, m0, ib, A, lda, U, ldu,
                            ipiv, &nextpiv, ( prevbuf != NULL ) ? &prevpiv : NULL );

    chameleon_parsec_pivot_store( nextbuf, &nextpiv );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

void INSERT_TASK_zgetrf_blocked_diag( const RUNTIME_option_t *options,
                                      int m, int n, int h, int m0, int ib, int readUp,
                                      CHAM_desc_t *A, int Am, int An,
                                      CHAM_desc_t *U, int Um, int Un,
                                      CHAM_ipiv_t *ipiv,
                                      CHAM_desc_pivot_t *pivot )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileA = A->get_blktile( A, Am, An );
    int                rankA = A->get_rankof( A, Am, An );
    int                reset = chameleon_parsec_pivot_contrib( pivot, rankA, 1 );
    int                nb    = pivot->nb;
    cham_flttype_t     dtyp  = pivot->dtyp;
    int                ldu   = -1;
    parsec_dtd_tile_t *tileUp = NULL;
    int                arenaU = 0;
    int                access_ipiv = ( h == 0 ) ? PARSEC_OUTPUT : PARSEC_INOUT;

    if ( readUp ) {
        ldu    = U->get_blktile( U, Um, Un )->ld;
        tileUp = RTBLKADDR( U, CHAMELEON_Complex64_t, Um, Un );
        arenaU = chameleon_parsec_get_arena_index( U, Um, Un );
    }

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zgetrf_blocked_diag_parsec, options->priority, PARSEC_DEV_CPU, "getrf_diag",
        sizeof(int),            &m,     PARSEC_VALUE,
        sizeof(int),            &n,     PARSEC_VALUE,
        sizeof(int),            &h,     PARSEC_VALUE,
        sizeof(int),            &m0,    PARSEC_VALUE,
        sizeof(int),            &ib,    PARSEC_VALUE,
        sizeof(int),            &nb,    PARSEC_VALUE,
        sizeof(cham_flttype_t), &dtyp,  PARSEC_VALUE,
        sizeof(int),            &reset, PARSEC_VALUE,
        PASSED_BY_REF, RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
                       chameleon_parsec_get_arena_index( A, Am, An ) | PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),            &(tileA->ld), PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipiv, ChamParsecIpiv, An ),
                       chameleon_parsec_ipiv_arena( ipiv, ChamParsecIpiv, An ) | access_ipiv,
        PASSED_BY_REF, tileUp, arenaU | PARSEC_INPUT,
        sizeof(int),            &ldu,   PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_pivot_tile( options, pivot, rankA, h ),
                       chameleon_parsec_pivot_arena( pivot ) | PARSEC_INOUT,
        PASSED_BY_REF, ( h > 0 ) ? chameleon_parsec_pivot_tile( options, pivot, rankA, h-1 ) : NULL,
                       chameleon_parsec_pivot_arena( pivot ) | PARSEC_INPUT,
        PARSEC_DTD_ARG_END );
}

static inline int
CORE_zgetrf_blocked_offdiag_parsec( parsec_execution_stream_t *context,
                                    parsec_task_t             *this_task )
{
    int m, n, h, m0, ib, nb, reset, lda, ldu;
    cham_flttype_t dtyp;
    CHAMELEON_Complex64_t *A, *U;
    void *nextbuf, *prevbuf;
    CHAM_pivot_t nextpiv, prevpiv;
    chameleon_parsec_pivot_hdr_t *hdr;

    parsec_dtd_unpack_args(
        this_task, &m, &n, &h, &m0, &ib, &nb, &dtyp, &reset,
        &A, &lda, &U, &ldu, &nextbuf, &prevbuf );

    hdr = (chameleon_parsec_pivot_hdr_t *)nextbuf;
    if ( reset ) {
        chameleon_parsec_pivot_reset( nextbuf, nb, dtyp );
    }
    hdr->h = h;

    chameleon_parsec_pivot_load( nextbuf, nb, dtyp, &nextpiv );
    if ( prevbuf != NULL ) {
        chameleon_parsec_pivot_load( prevbuf, nb, dtyp, &prevpiv );
    }

    CORE_zgetrf_panel_offdiag( m, n, h, m0, ib, A, lda, U, ldu,
                               &nextpiv, ( prevbuf != NULL ) ? &prevpiv : NULL );

    chameleon_parsec_pivot_store( nextbuf, &nextpiv );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

void INSERT_TASK_zgetrf_blocked_offdiag( const RUNTIME_option_t *options,
                                         int m, int n, int h, int m0, int ib, int readUp,
                                         CHAM_desc_t *A, int Am, int An,
                                         CHAM_desc_t *U, int Um, int Un,
                                         CHAM_desc_pivot_t *pivot )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileA = A->get_blktile( A, Am, An );
    int                rankA = A->get_rankof( A, Am, An );
    int                reset = chameleon_parsec_pivot_contrib( pivot, rankA, 0 );
    int                root  = chameleon_parsec_pivot( pivot )->root;
    int                nb    = pivot->nb;
    cham_flttype_t     dtyp  = pivot->dtyp;
    int                ldu   = -1;
    parsec_dtd_tile_t *tileUp = NULL;
    int                arenaU = 0;

    if ( readUp ) {
        ldu    = U->get_blktile( U, Um, Un )->ld;
        tileUp = RTBLKADDR( U, CHAMELEON_Complex64_t, Um, Un );
        arenaU = chameleon_parsec_get_arena_index( U, Um, Un );
    }

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zgetrf_blocked_offdiag_parsec, options->priority, PARSEC_DEV_CPU, "getrf_offdiag",
        sizeof(int),            &m,     PARSEC_VALUE,
        sizeof(int),            &n,     PARSEC_VALUE,
        sizeof(int),            &h,     PARSEC_VALUE,
        sizeof(int),            &m0,    PARSEC_VALUE,
        sizeof(int),            &ib,    PARSEC_VALUE,
        sizeof(int),            &nb,    PARSEC_VALUE,
        sizeof(cham_flttype_t), &dtyp,  PARSEC_VALUE,
        sizeof(int),            &reset, PARSEC_VALUE,
        PASSED_BY_REF, RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
                       chameleon_parsec_get_arena_index( A, Am, An ) | PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),            &(tileA->ld), PARSEC_VALUE,
        PASSED_BY_REF, tileUp, arenaU | PARSEC_INPUT,
        sizeof(int),            &ldu,   PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_pivot_tile( options, pivot, rankA, h ),
                       chameleon_parsec_pivot_arena( pivot ) | PARSEC_INOUT,
        PASSED_BY_REF, ( h > 0 ) ? chameleon_parsec_pivot_tile( options, pivot, root, h-1 ) : NULL,
                       chameleon_parsec_pivot_arena( pivot ) | PARSEC_INPUT,
        PARSEC_DTD_ARG_END );
}
