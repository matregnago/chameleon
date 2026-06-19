/**
 *
 * @file hip_zgeadd.c
 *
 * @copyright 2026-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon hip_zgeadd GPU kernel
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2026-05-27
 * @precisions normal z -> c d s
 *
 */
#include "gpuhipblas.h"

int
HIP_zgeadd( cham_trans_t trans,
            int m, int n,
            const hipDoubleComplex *alpha,
            const hipDoubleComplex *A, int lda,
            const hipDoubleComplex *beta,
            hipDoubleComplex *B, int ldb,
            hipblasHandle_t handle )
{
    hipblasStatus_t rc;

    rc = hipblasZgeam( handle,
                       (hipblasOperation_t) chameleon_hipblas_const(trans),
                       (hipblasOperation_t) chameleon_hipblas_const(ChamNoTrans),
                       m, n,
                       HIPBLAS_VALUE(alpha), A, lda,
                       HIPBLAS_VALUE(beta),  B, ldb,
                       B, ldb );

    assert( rc == HIPBLAS_STATUS_SUCCESS );
    (void)rc;
    return CHAMELEON_SUCCESS;
}
