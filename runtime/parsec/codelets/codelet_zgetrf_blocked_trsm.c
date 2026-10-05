/**
 *
 * @file starpu/codelet_zgetrf_blocked_trsm.c
 *
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zpanel PaRSEC codelets
 *
 * @version 1.4.0
 * @comment Codelets to perform panel factorization with partial pivoting
 *
 * @author Alycia Lisito
 * @date 2026-10-05
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"
#include <coreblas/cblas_wrapper.h>

static const CHAMELEON_Complex64_t zone = (CHAMELEON_Complex64_t)1.0;

/*
 * The selected pivot rows are read from the buffer of the root, where the
 * candidates of all the ranks have been merged.
 */
static inline int
CORE_zgetrf_cpy_pivrow_in_Up_parsec( parsec_execution_stream_t *context,
                                     parsec_task_t             *this_task )
{
    int h, n, ib, nb, ldu;
    cham_flttype_t dtyp;
    CHAMELEON_Complex64_t *U;
    void *pivbuf;
    CHAM_pivot_t piv;

    parsec_dtd_unpack_args( this_task, &h, &n, &ib, &nb, &dtyp, &U, &ldu, &pivbuf );

    chameleon_parsec_pivot_load( pivbuf, nb, dtyp, &piv );
    cblas_zcopy( n, piv.pivrow, 1, U + ( h % ib ), ldu );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

void INSERT_TASK_zgetrf_cpy_pivrow_in_Up( const RUNTIME_option_t *options,
                                          CHAM_desc_t            *Up,
                                          int                     Upm,
                                          int                     k,
                                          int                     h,
                                          int                     ib,
                                          int                     n,
                                          CHAM_desc_pivot_t      *pivot )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileU    = Up->get_blktile( Up, Upm, 0 );
    int                root     = chameleon_parsec_pivot( pivot )->root;
    int                accessUp = ( ( ( h % ib ) == 0 ) || ( ib == 1 ) ) ? PARSEC_OUTPUT : PARSEC_INOUT;
    int                nb       = pivot->nb;
    cham_flttype_t     dtyp     = pivot->dtyp;

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zgetrf_cpy_pivrow_in_Up_parsec, options->priority, PARSEC_DEV_CPU, "getrf_cpy_pivrow",
        sizeof(int),            &h,    PARSEC_VALUE,
        sizeof(int),            &n,    PARSEC_VALUE,
        sizeof(int),            &ib,   PARSEC_VALUE,
        sizeof(int),            &nb,   PARSEC_VALUE,
        sizeof(cham_flttype_t), &dtyp, PARSEC_VALUE,
        PASSED_BY_REF, RTBLKADDR( Up, CHAMELEON_Complex64_t, Upm, 0 ),
                       chameleon_parsec_get_arena_index( Up, Upm, 0 ) | accessUp | PARSEC_AFFINITY,
        sizeof(int),            &(tileU->ld), PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_pivot_tile( options, pivot, root, h ),
                       chameleon_parsec_pivot_arena( pivot ) | PARSEC_INPUT,
        PARSEC_DTD_ARG_END );

    (void)k;
}

static inline int
CORE_zgetrf_blocked_trsm_parsec( parsec_execution_stream_t *context,
                                 parsec_task_t             *this_task )
{
    int m, n, h, ib, nb, ldu;
    cham_flttype_t dtyp;
    CHAMELEON_Complex64_t *U;
    void *pivbuf;
    CHAM_pivot_t piv;

    parsec_dtd_unpack_args( this_task, &m, &n, &h, &ib, &nb, &dtyp, &U, &ldu, &pivbuf );

    chameleon_parsec_pivot_load( pivbuf, nb, dtyp, &piv );

    /* Copy the final max line of the block and solve */
    cblas_zcopy( n, piv.pivrow, 1, U + m - 1, ldu );

    if ( ( n - h ) > 0 ) {
        cblas_ztrsm( CblasColMajor,
                     CblasLeft, CblasLower,
                     CblasNoTrans, CblasUnit,
                     ib, n - h,
                     CBLAS_SADDR(zone), U + (h-ib) * ldu, ldu,
                                        U +  h     * ldu, ldu );
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

void INSERT_TASK_zgetrf_blocked_trsm( const RUNTIME_option_t *options,
                                      int                     m,
                                      int                     n,
                                      int                     h,
                                      int                     ib,
                                      CHAM_desc_t            *U,
                                      int                     Um,
                                      int                     Un,
                                      CHAM_desc_pivot_t      *pivot )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileU = U->get_blktile( U, Um, Un );
    int                root  = chameleon_parsec_pivot( pivot )->root;
    int                nb    = pivot->nb;
    cham_flttype_t     dtyp  = pivot->dtyp;

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zgetrf_blocked_trsm_parsec, options->priority, PARSEC_DEV_CPU, "getrf_blocked_trsm",
        sizeof(int),            &m,    PARSEC_VALUE,
        sizeof(int),            &n,    PARSEC_VALUE,
        sizeof(int),            &h,    PARSEC_VALUE,
        sizeof(int),            &ib,   PARSEC_VALUE,
        sizeof(int),            &nb,   PARSEC_VALUE,
        sizeof(cham_flttype_t), &dtyp, PARSEC_VALUE,
        PASSED_BY_REF, RTBLKADDR( U, CHAMELEON_Complex64_t, Um, Un ),
                       chameleon_parsec_get_arena_index( U, Um, Un ) | PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),            &(tileU->ld), PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_pivot_tile( options, pivot, root, h-1 ),
                       chameleon_parsec_pivot_arena( pivot ) | PARSEC_INPUT,
        PARSEC_DTD_ARG_END );
}
