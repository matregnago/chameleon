/**
 *
 * @file hip_zlacpy.cpp
 *
 * @copyright 2026-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon hip_zlacpy GPU kernel
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2026-05-27
 * @precisions normal z -> c d s
 *
 */
#include "gpuhipblas.h"

#define BLK_X 32
#define BLK_Y 32

__global__
void HIP_zlacpy_upper( int m, int n,
                       const hipDoubleComplex *A, int lda,
                       hipDoubleComplex *B, int ldb )
{
    int x = threadIdx.x + blockIdx.x * BLK_X;
    int y = threadIdx.y + blockIdx.y * BLK_Y;

    if ( ( x > y ) || ( x >= m ) || ( y >= n ) ) {
        return;
    }
    B[ldb * y + x] = A[lda * y + x];
}

__global__
void HIP_zlacpy_lower( int m, int n,
                       const hipDoubleComplex *A, int lda,
                       hipDoubleComplex *B, int ldb )
{
    int x = threadIdx.x + blockIdx.x * BLK_X;
    int y = threadIdx.y + blockIdx.y * BLK_Y;

    if ( ( x < y ) || ( x >= m ) || ( y >= n ) ) {
        return;
    }
    B[ldb * y + x] = A[lda * y + x];
}

extern "C"
int HIP_zlacpy( cham_uplo_t uplo,
                int m, int n,
                const hipDoubleComplex *A, int lda,
                hipDoubleComplex *B, int ldb,
                hipblasHandle_t handle )
{
    hipError_t err;
    hipblasStatus_t rc = HIPBLAS_STATUS_SUCCESS;
    hipStream_t stream;

#if defined(PRECISION_z) || defined(PRECISION_c)
    hipDoubleComplex one  = make_hipDoubleComplex( 1., 0. );
    hipDoubleComplex zero = make_hipDoubleComplex( 0., 0. );
#else
    double one  = 1.;
    double zero = 0.;
#endif

    dim3 threads( BLK_X, BLK_Y );
    dim3 grid( chameleon_ceil( m, BLK_X ), chameleon_ceil( n, BLK_Y ) );

    hipblasGetStream( handle, &stream );

    switch ( uplo ) {
    case ChamUpperLower:
        rc = hipblasZgeam( handle,
                           (hipblasOperation_t) chameleon_hipblas_const(ChamNoTrans),
                           (hipblasOperation_t) chameleon_hipblas_const(ChamNoTrans),
                           m, n,
                           &one,  A, lda,
                           &zero, B, ldb,
                           B, ldb );
        break;
    case ChamUpper:
        HIP_zlacpy_upper<<<grid, threads, 0, stream>>>
            ( m, n, A, lda, B, ldb );
        break;
    case ChamLower:
        HIP_zlacpy_lower<<<grid, threads, 0, stream>>>
            ( m, n, A, lda, B, ldb );
        break;
    }

    assert( rc == HIPBLAS_STATUS_SUCCESS );
    err = hipGetLastError();
    if ( err != hipSuccess ) {
        fprintf( stderr, "HIP_zlacpy failed to launch HIP kernel %s\n", hipGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }
    return CHAMELEON_SUCCESS;
}
