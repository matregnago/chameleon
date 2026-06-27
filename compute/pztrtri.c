/**
 *
 * @file pztrtri.c
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon ztrtri parallel algorithm
 *
 * @version 1.4.0
 * @comment This file has been automatically generated
 *          from Plasma 2.5.0 for CHAMELEON 0.9.2
 * @author Julien Langou
 * @author Henricus Bouwmeester
 * @author Mathieu Faverge
 * @author Emmanuel Agullo
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @author Samuel Thibault
 * @date 2025-12-19
 * @precisions normal z -> s d c
 *
 */
#include "control/common.h"

#define A(m,n) A,  m,  n

static CHAMELEON_Complex64_t zone  = (CHAMELEON_Complex64_t)1.0;
static CHAMELEON_Complex64_t mzone = (CHAMELEON_Complex64_t)-1.0;

/**
 * Parallel tile triangular matrix inverse - Variant 1 from
 *  - Bientinesi et al., Families of algorithms related to the inversion of a Symmetric Positive Definite matrix.
 *  - H. Bouwmeester, J. Langou. A Critical Path Approach to Analyzing Parallelism of Algorithmic Variants. Application to Cholesky Inversion
 */
static inline void
chameleon_pztrtri_v1( CHAM_context_t *chamctxt, cham_uplo_t uplo, cham_diag_t diag,
                      CHAM_desc_t *A, RUNTIME_option_t *options )
{
    RUNTIME_sequence_t *sequence = options->sequence;
    RUNTIME_request_t  *request  = options->request;

    int k, m, n;
    int tempkn, tempkm, tempmm, tempnn;

    /*
     *  ChamLower
     */
    if (uplo == ChamLower) {
        for (k = 0; k < A->nt; k++) {
            RUNTIME_iteration_push(chamctxt, k);

            tempkn = A->get_blkdim( A, k, DIM_n, A->n );
            for (n = 0; n < k; n++) {
                tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                INSERT_TASK_ztrmm(
                    options,
                    ChamRight, uplo, ChamNoTrans, diag,
                    tempkn, tempnn, A->mb,
                    mzone, A(n, n),
                           A(k, n));

                for (m = n+1; m < k; m++) {
                    tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                    INSERT_TASK_zgemm(
                        options,
                        ChamNoTrans, ChamNoTrans,
                        tempkn, tempnn, tempmm, A->mb,
                        mzone, A(k, m),
                               A(m, n),
                        zone,  A(k, n));
                }
            }

            for (n = 0; n < k; n++) {
                chameleon_data_flush( sequence, A(k, n), request->flush );
                INSERT_TASK_ztrsm(
                    options,
                    ChamLeft, uplo, ChamNoTrans, diag,
                    tempkn, A->nb, A->mb,
                    zone, A(k, k),
                          A(k, n));
            }

            chameleon_data_flush( sequence, A(k, k), request->flush );
            INSERT_TASK_ztrtri(
                options,
                uplo, diag,
                tempkn, A->mb,
                A(k, k), A->nb*k);

            RUNTIME_iteration_pop(chamctxt);
        }
    }
    /*
     *  ChamUpper
     */
    else {
        for (k = 0; k < A->mt; k++) {
            RUNTIME_iteration_push(chamctxt, k);

            tempkm = A->get_blkdim( A, k, DIM_m, A->m );
            for (m = 0; m < k; m++) {
                tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                INSERT_TASK_ztrmm(
                    options,
                    ChamLeft, uplo, ChamNoTrans, diag,
                    tempmm, tempkm, A->mb,
                    mzone, A(m, m),
                           A(m, k));

                for (n = m+1; n < k; n++) {
                    tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                    INSERT_TASK_zgemm(
                        options,
                        ChamNoTrans, ChamNoTrans,
                        tempmm, tempkm, tempnn, A->mb,
                        mzone, A(m, n),
                               A(n, k),
                        zone,  A(m, k));
                }
            }

            for (m = 0; m < k; m++) {
                chameleon_data_flush( sequence, A(m, k), request->flush );
                INSERT_TASK_ztrsm(
                    options,
                    ChamRight, uplo, ChamNoTrans, diag,
                    A->mb, tempkm, A->mb,
                    zone, A(k, k),
                          A(m, k));
            }

            chameleon_data_flush( sequence, A(k, k), request->flush );
            INSERT_TASK_ztrtri(
                options,
                uplo, diag,
                tempkm, A->mb,
                A(k, k), A->mb*k);

            RUNTIME_iteration_pop(chamctxt);
        }
    }
}

/**
 * Parallel tile triangular matrix inverse - Variant 2 from
 *  - Bientinesi et al., Families of algorithms related to the inversion of a Symmetric Positive Definite matrix.
 *  - H. Bouwmeester, J. Langou. A Critical Path Approach to Analyzing Parallelism of Algorithmic Variants. Application to Cholesky Inversion
 */
static inline void
chameleon_pztrtri_v2( CHAM_context_t *chamctxt, cham_uplo_t uplo, cham_diag_t diag,
                      CHAM_desc_t *A, RUNTIME_option_t *options )
{
    RUNTIME_sequence_t *sequence = options->sequence;
    RUNTIME_request_t  *request  = options->request;

    int k, m, n;
    int tempkn, tempkm, tempmm, tempnn;

    /*
     *  ChamLower
     */
    if (uplo == ChamLower) {
        for (k = 0; k < A->nt; k++) {
            RUNTIME_iteration_push(chamctxt, k);

            tempkn = A->get_blkdim( A, k, DIM_n, A->n );
            for (m = k+1; m < A->mt; m++) {
                tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                INSERT_TASK_ztrsm(
                    options,
                    ChamLeft, uplo, ChamNoTrans, diag,
                    tempmm, tempkn, A->mb,
                    (m == k+1) ? mzone : zone, A(m, m),
                           A(m, k));

                for (n = m+1; n < A->mt; n++) {
                    tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                    INSERT_TASK_zgemm(
                        options,
                        ChamNoTrans, ChamNoTrans,
                        tempnn, tempkn, tempmm, A->mb,
                        mzone, A(n, m),
                                A(m, k),
                        (m == k+1) ? mzone : zone, A(n, k));
                }
                chameleon_data_flush( sequence, A(m, k), request->flush );
            }

            for (m = k+1; m < A->mt; m++) {
                tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                INSERT_TASK_ztrsm(
                    options,
                    ChamRight, uplo, ChamNoTrans, diag,
                    tempmm, tempkn, A->mb,
                    zone, A(k, k),
                          A(m, k));
            }

            chameleon_data_flush( sequence, A(k, k), request->flush );
            INSERT_TASK_ztrtri(
                options,
                uplo, diag,
                tempkn, A->mb,
                A(k, k), A->nb*k);

            RUNTIME_iteration_pop(chamctxt);
        }
    }
    /*
     *  ChamUpper
     */
    else {
        for (k = 0; k < A->mt; k++) {
            RUNTIME_iteration_push(chamctxt, k);

            tempkm = A->get_blkdim( A, k, DIM_m, A->m );
            for (n = k+1; n < A->nt; n++) {
                tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                INSERT_TASK_ztrsm(
                    options,
                    ChamRight, uplo, ChamNoTrans, diag,
                    tempkm, tempnn, A->mb,
                    (n == k+1) ? mzone : zone, A(n, n),
                           A(k, n));

                for (m = n+1; m < A->nt; m++) {
                    tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                    INSERT_TASK_zgemm(
                        options,
                        ChamNoTrans, ChamNoTrans,
                        tempkm, tempmm, tempnn, A->mb,
                        mzone, A(k, n),
                                A(n, m),
                        (n == k+1) ? mzone : zone, A(k, m));
                }
                chameleon_data_flush( sequence, A(k, n), request->flush );
            }

            for (n = k+1; n < A->nt; n++) {
                tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                INSERT_TASK_ztrsm(
                    options,
                    ChamLeft, uplo, ChamNoTrans, diag,
                    tempkm, tempnn, A->mb,
                    zone, A(k, k),
                          A(k, n));
            }

            chameleon_data_flush( sequence, A(k, k), request->flush );
            INSERT_TASK_ztrtri(
                options,
                uplo, diag,
                tempkm, A->mb,
                A(k, k), A->mb*k);

            RUNTIME_iteration_pop(chamctxt);
        }
    }
}

/**
 * Parallel tile triangular matrix inverse - Variant 3 from
 *  - Bientinesi et al., Families of algorithms related to the inversion of a Symmetric Positive Definite matrix.
 *  - H. Bouwmeester, J. Langou. A Critical Path Approach to Analyzing Parallelism of Algorithmic Variants.
 */
static inline void
chameleon_pztrtri_v3( CHAM_context_t *chamctxt, cham_uplo_t uplo, cham_diag_t diag,
                      CHAM_desc_t *A, RUNTIME_option_t *options )
{
    RUNTIME_sequence_t *sequence = options->sequence;
    RUNTIME_request_t  *request  = options->request;

    int k, m, n;
    int tempkn, tempkm, tempmm, tempnn;

    /*
     *  ChamLower
     */
    if (uplo == ChamLower) {
        for (k = 0; k < A->nt; k++) {
            RUNTIME_iteration_push(chamctxt, k);

            tempkn = A->get_blkdim( A, k, DIM_n, A->n );
            for (m = k+1; m < A->mt; m++) {
                tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                INSERT_TASK_ztrsm(
                    options,
                    ChamRight, uplo, ChamNoTrans, diag,
                    tempmm, tempkn, A->mb,
                    mzone, A(k, k),
                           A(m, k));
            }
            for (m = k+1; m < A->mt; m++) {
                tempmm = A->get_blkdim( A, m, DIM_m, A->m );
                for (n = 0; n < k; n++) {
                    INSERT_TASK_zgemm(
                        options,
                        ChamNoTrans, ChamNoTrans,
                        tempmm, A->nb, tempkn, A->mb,
                        zone, A(m, k),
                              A(k, n),
                        zone, A(m, n));
                }
                chameleon_data_flush( sequence, A(m, k), request->flush );
            }

            for (n = 0; n < k; n++) {
                chameleon_data_flush( sequence, A(k, n), request->flush );
                INSERT_TASK_ztrsm(
                    options,
                    ChamLeft, uplo, ChamNoTrans, diag,
                    tempkn, A->nb, A->mb,
                    zone, A(k, k),
                          A(k, n));
            }

            chameleon_data_flush( sequence, A(k, k), request->flush );
            INSERT_TASK_ztrtri(
                options,
                uplo, diag,
                tempkn, A->mb,
                A(k, k), A->nb*k);

            RUNTIME_iteration_pop(chamctxt);
        }
    }
    /*
     *  ChamUpper
     */
    else {
        for (k = 0; k < A->mt; k++) {
            RUNTIME_iteration_push(chamctxt, k);

            tempkm = A->get_blkdim( A, k, DIM_m, A->m );
            for (n = k+1; n < A->nt; n++) {
                tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                INSERT_TASK_ztrsm(
                    options,
                    ChamLeft, uplo, ChamNoTrans, diag,
                    tempkm, tempnn, A->mb,
                    mzone, A(k, k),
                           A(k, n));
            }
            for (n = k+1; n < A->nt; n++) {
                tempnn = A->get_blkdim( A, n, DIM_n, A->n );
                for (m = 0; m < k; m++) {
                    INSERT_TASK_zgemm(
                        options,
                        ChamNoTrans, ChamNoTrans,
                        A->mb, tempnn, tempkm, A->mb,
                        zone, A(m, k),
                              A(k, n),
                        zone, A(m, n));
                }
                chameleon_data_flush( sequence, A(k, n), request->flush );
            }

            for (m = 0; m < k; m++) {
                chameleon_data_flush( sequence, A(m, k), request->flush );
                INSERT_TASK_ztrsm(
                    options,
                    ChamRight, uplo, ChamNoTrans, diag,
                    A->mb, tempkm, A->mb,
                    zone, A(k, k),
                          A(m, k));
            }

            chameleon_data_flush( sequence, A(k, k), request->flush );
            INSERT_TASK_ztrtri(
                options,
                uplo, diag,
                tempkm, A->mb,
                A(k, k), A->mb*k);

            RUNTIME_iteration_pop(chamctxt);
        }
    }
}

/**
 *  Parallel tile triangular matrix inverse wrapper.
 */
void
chameleon_pztrtri( cham_uplo_t uplo, cham_diag_t diag, CHAM_desc_t *A,
                   RUNTIME_sequence_t *sequence, RUNTIME_request_t *request )
{
    CHAM_context_t   *chamctxt;
    RUNTIME_option_t  options;
    cham_trtri_t      alg = ChamTrtriVariant3;
    char             *algostr;

    chamctxt = chameleon_context_self();
    if (sequence->status != CHAMELEON_SUCCESS) {
        return;
    }

    algostr = chameleon_getenv( "CHAMELEON_TRTRI_ALGO" );
    if ( algostr ) {
        if ( strcasecmp( algostr, "1" ) == 0 ||
             strcasecmp( algostr, "v1" ) == 0 ||
             strcasecmp( algostr, "variant1" ) == 0 )
        {
            alg = ChamTrtriVariant1;
        }
        else if ( strcasecmp( algostr, "2" ) == 0 ||
                  strcasecmp( algostr, "v2" ) == 0 ||
                  strcasecmp( algostr, "variant2" ) == 0 )
        {
            alg = ChamTrtriVariant2;
        }
        else if ( strcasecmp( algostr, "3" ) == 0 ||
                  strcasecmp( algostr, "v3" ) == 0 ||
                  strcasecmp( algostr, "variant3" ) == 0 )
        {
            alg = ChamTrtriVariant3;
        }
        else {
            fprintf( stderr, "ERROR: CHAMELEON_TRTRI_ALGO is not one of 1, 2, 3, V1, V2, V3, VARIANT1, VARIANT2, VARIANT3 => Switch back to Variant 3\n" );
        }
    }
    chameleon_cleanenv( algostr );

    RUNTIME_options_init( &options, chamctxt, sequence, request );

    switch( alg ) {
    case ChamTrtriVariant1:
        chameleon_pztrtri_v1( chamctxt, uplo, diag, A, &options );
        break;

    case ChamTrtriVariant2:
        chameleon_pztrtri_v2( chamctxt, uplo, diag, A, &options );
        break;

    case ChamTrtriAuto:
    case ChamTrtriVariant3:
    default:
        chameleon_pztrtri_v3( chamctxt, uplo, diag, A, &options );
        break;
    }

    /* Mark written data for synchronization */
    A->sync = 1;

    RUNTIME_options_finalize( &options, chamctxt );
}
