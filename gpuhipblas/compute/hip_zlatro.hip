/**
 *
 * @file hip_zlatro.hip
 *
 * @copyright 2026-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon hip_zlatro GPU kernel
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
void HIP_zlatro_upper_trans( int m, int n,
                             const hipDoubleComplex *A, int lda,
                             hipDoubleComplex *B, int ldb )
{
    __shared__ hipDoubleComplex wA[BLK_X * BLK_Y];
    int i = threadIdx.x;
    int j = threadIdx.y;
    int x = threadIdx.x + blockIdx.x * BLK_X;
    int y = threadIdx.y + blockIdx.y * BLK_Y;

    if ( ( x > y ) || ( x >= m ) || ( y >= n ) ) {
        return;
    }
    wA[j + BLK_Y * i] = A[x + y * lda];
    __syncthreads();
    B[y + ldb * x] = wA[j + BLK_Y * i];
}

__global__
void HIP_zlatro_lower_trans( int m, int n,
                             const hipDoubleComplex *A, int lda,
                             hipDoubleComplex *B, int ldb )
{
    __shared__ hipDoubleComplex wA[BLK_X * BLK_Y];
    int i = threadIdx.x;
    int j = threadIdx.y;
    int x = threadIdx.x + blockIdx.x * BLK_X;
    int y = threadIdx.y + blockIdx.y * BLK_Y;

    if ( ( x < y ) || ( x >= m ) || ( y >= n ) ) {
        return;
    }
    wA[j + BLK_Y * i] = A[x + y * lda];
    __syncthreads();
    B[y + ldb * x] = wA[j + BLK_Y * i];
}

#if defined(PRECISION_z) || defined(PRECISION_c)
__global__
void HIP_zlatro_upper_conjtrans( int m, int n,
                                 const hipDoubleComplex *A, int lda,
                                 hipDoubleComplex *B, int ldb )
{
    __shared__ hipDoubleComplex wA[BLK_X * BLK_Y];
    int i = threadIdx.x;
    int j = threadIdx.y;
    int x = threadIdx.x + blockIdx.x * BLK_X;
    int y = threadIdx.y + blockIdx.y * BLK_Y;

    if ( ( x > y ) || ( x >= m ) || ( y >= n ) ) {
        return;
    }
    wA[j + BLK_Y * i] = A[x + y * lda];
    __syncthreads();
    B[y + ldb * x] = hipConj( wA[j + BLK_Y * i] );
}

__global__
void HIP_zlatro_lower_conjtrans( int m, int n,
                                 const hipDoubleComplex *A, int lda,
                                 hipDoubleComplex *B, int ldb )
{
    __shared__ hipDoubleComplex wA[BLK_X * BLK_Y];
    int i = threadIdx.x;
    int j = threadIdx.y;
    int x = threadIdx.x + blockIdx.x * BLK_X;
    int y = threadIdx.y + blockIdx.y * BLK_Y;

    if ( ( x < y ) || ( x >= m ) || ( y >= n ) ) {
        return;
    }
    wA[j + BLK_Y * i] = A[x + y * lda];
    __syncthreads();
    B[y + ldb * x] = hipConj( wA[j + BLK_Y * i] );
}
#endif

extern "C"
int HIP_zlatro( cham_uplo_t uplo, cham_trans_t trans,
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
                           (hipblasOperation_t) chameleon_hipblas_const(trans),
                           (hipblasOperation_t) chameleon_hipblas_const(ChamNoTrans),
                           n, m,
                           &one,  A, lda,
                           &zero, B, ldb,
                           B, ldb );
        break;
    case ChamUpper:
#if defined(PRECISION_z) || defined(PRECISION_c)
        if ( trans == ChamConjTrans ) {
            HIP_zlatro_upper_conjtrans<<<grid, threads, 0, stream>>>
                ( m, n, A, lda, B, ldb );
        }
        else
#endif
        {
            HIP_zlatro_upper_trans<<<grid, threads, 0, stream>>>
                ( m, n, A, lda, B, ldb );
        }
        break;
    case ChamLower:
#if defined(PRECISION_z) || defined(PRECISION_c)
        if ( trans == ChamConjTrans ) {
            HIP_zlatro_lower_conjtrans<<<grid, threads, 0, stream>>>
                ( m, n, A, lda, B, ldb );
        }
        else
#endif
        {
            HIP_zlatro_lower_trans<<<grid, threads, 0, stream>>>
                ( m, n, A, lda, B, ldb );
        }
        break;
    }

    assert( rc == HIPBLAS_STATUS_SUCCESS );
    err = hipGetLastError();
    if ( err != hipSuccess ) {
        fprintf( stderr, "HIP_zlatro failed to launch HIP kernel %s\n", hipGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }
    return CHAMELEON_SUCCESS;
}
