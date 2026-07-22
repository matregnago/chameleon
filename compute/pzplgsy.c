/**
 *
 * @file pzplgsy.c
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zplgsy parallel algorithm
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Emmanuel Agullo
 * @author Cedric Castagnede
 * @author Mathis Rade
 * @author Florent Pruvost
 * @date 2025-01-24
 * @precisions normal z -> c d s
 *
 */
#include "control/common.h"

#define A(m,n) A,  m,  n

/**
 * @brief Generic tile algorithm to generate a symmetric random matrix
 *
 * This is the version to use by default.
 */
void chameleon_pzplgsy_generic( CHAMELEON_Complex64_t  bump,
                                cham_uplo_t            uplo,
                                CHAM_desc_t           *A,
                                int                    bigM,
                                int                    m0,
                                int                    n0,
                                unsigned long long int seed,
                                RUNTIME_sequence_t    *sequence,
                                RUNTIME_request_t     *request )
{
    CHAM_context_t *chamctxt;
    RUNTIME_option_t options;

    int m, n;
    int tile_m0, tile_n0;
    int tempmm, tempnn;

    chamctxt = chameleon_context_self();
    if (sequence->status != CHAMELEON_SUCCESS) {
        return;
    }
    RUNTIME_options_init(&options, chamctxt, sequence, request);
    RUNTIME_options_set_taskcolor( &options, CHAMELEON_DAG_COLOR_ALGORITHM( plgsy ) );

    for (m = 0; m < A->mt; m++) {
        tempmm  = A->get_blkdim( A, m, DIM_m, A->m );
        tile_m0 = m * A->mb + m0;

        for (n = 0; n < A->nt; n++) {
            tempnn  = A->get_blkdim( A, n, DIM_n, A->n );
            tile_n0 = n * A->nb + n0;

            /* Let's skip the upper part */
            if ( ( uplo == ChamLower ) && ( tile_n0 >= (tile_m0 + tempmm) ) ) {
                continue;
            }
            /* Let's skip the lower part */
            else if ( ( uplo == ChamUpper ) && (tile_m0 >= (tile_n0 + tempnn) ) ) {
                continue;
            }

            INSERT_TASK_zplgsy(
                &options,
                bump, uplo, tempmm, tempnn, A(m, n),
                bigM, tile_m0, tile_n0, seed );
        }
    }
    RUNTIME_options_finalize(&options, chamctxt);
}

void chameleon_pzplgsy( CHAMELEON_Complex64_t bump, cham_uplo_t uplo, CHAM_desc_t *A,
                        int bigM, int m0, int n0, unsigned long long int seed,
                        RUNTIME_sequence_t *sequence, RUNTIME_request_t *request )
{
    chameleon_pzplgsy_generic( bump, uplo, A, bigM, m0, n0, seed, sequence, request );

    /* Mark written data for synchronization */
    A->sync = 1;
}
