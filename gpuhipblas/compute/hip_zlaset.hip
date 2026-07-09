/**
 *
 * @file hip_zlaset.hip
 *
 * @copyright 2026-2026 Bordeaux INP, CNRS (LaBRI UMR 5800 ), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon hip_zlaset GPU kernel
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
void HIP_zlaset_upper( int m, int n,
                       hipDoubleComplex alpha,
                       hipDoubleComplex beta,
                       hipDoubleComplex *A, int lda )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ( i >= m ) || ( j >= n ) || ( j < i ) ) {
        return;
    }

    if ( j == i ) {
        A[i + j*lda] = beta;
    }
    else {
        A[i + j*lda] = alpha;
    }
}

__global__
void HIP_zlaset_lower( int m, int n,
                       hipDoubleComplex alpha,
                       hipDoubleComplex beta,
                       hipDoubleComplex *A, int lda )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ( i >= m ) || ( j >= n ) || ( i < j ) ) {
        return;
    }

    if ( j == i ) {
        A[i + j*lda] = beta;
    }
    else {
        A[i + j*lda] = alpha;
    }
}

__global__
void HIP_zlaset_upperlower( int m, int n,
                            hipDoubleComplex alpha,
                            hipDoubleComplex beta,
                            hipDoubleComplex *A, int lda )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ( i >= m ) || ( j >= n ) ) {
        return;
    }

    if ( j == i ) {
        A[i + j*lda] = beta;
    }
    else {
        A[i + j*lda] = alpha;
    }
}

extern "C"
int HIP_zlaset( cham_uplo_t uplo, int m, int n,
                const hipDoubleComplex *alpha, const hipDoubleComplex *beta,
                hipDoubleComplex *A, int lda,
                hipblasHandle_t handle )
{
    hipError_t err;
    hipStream_t stream;

    hipblasGetStream( handle, &stream );

    if ( ( uplo == ChamUpperLower ) &&
#if defined(PRECISION_z) || defined(PRECISION_c)
         ( hipCreal(*alpha) == 0. ) && ( hipCimag(*alpha) == 0. ) &&
         ( hipCreal(*beta)  == 0. ) && ( hipCimag(*beta)  == 0. )
#else
         ( *alpha == 0. ) && ( *beta == 0. )
#endif
         )
    {
        err = hipMemset2DAsync( A, sizeof(hipDoubleComplex) * lda, 0,
                                sizeof(hipDoubleComplex) * m, n, stream );
    }
    else
    {
        dim3 threads( BLK_X, BLK_Y );
        dim3 grid( chameleon_ceil( m, BLK_X ), chameleon_ceil( n, BLK_Y ) );

        switch ( uplo ) {
        case ChamUpperLower:
            HIP_zlaset_upperlower<<<grid, threads, 0, stream>>>( m, n,
                                                                 *alpha, *beta,
                                                                 A, lda );
            break;
        case ChamUpper:
            HIP_zlaset_upper<<<grid, threads, 0, stream>>>( m, n,
                                                            *alpha, *beta,
                                                            A, lda );
            break;
        case ChamLower:
            HIP_zlaset_lower<<<grid, threads, 0, stream>>>( m, n,
                                                            *alpha, *beta,
                                                            A, lda );
            break;
        }
    }

    err = hipGetLastError();
    if ( err != hipSuccess ) {
        fprintf( stderr, "HIP_zlaset failed to launch HIP kernel %s\n", hipGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }
    return CHAMELEON_SUCCESS;
}
