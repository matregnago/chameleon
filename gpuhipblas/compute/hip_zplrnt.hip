/**
 *
 * @file hip_zplrnt.cpp
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon hip_zplrnt GPU kernel
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @author Loris Lucido
 * @date 2026-05-27
 * @precisions normal z -> d s c
 *
 */
#include "gpuhipblas.h"
#include "gpuhipblas/gpuhipblas_random.h"

#if defined(PRECISION_z) || defined(PRECISION_c)
#define NBELEM 2
#else
#define NBELEM 1
#endif

#define BLK_X 32
#define BLK_Y 64

__global__
void hip_zplrnt_kernel( int m, int n, hipDoubleComplex *A, int lda,
                        int bigM, int m0, int n0, unsigned long long int seed )
{
    int i;
    hipDoubleComplex *tmp;
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;
    unsigned long long int jump = (unsigned long long int)(m0 + x)
                                + (unsigned long long int)(n0 + y) * (unsigned long long int)bigM;
    unsigned long long int ran = HIP_rnd64_jump( NBELEM * jump, seed );
    bool full = (x + BLK_X <= m);

    if ( y >= n ) {
        return;
    }

    tmp = A + x + y * lda;
    if ( full ) {
#pragma unroll
        for ( i = 0; i < BLK_X; ++i, tmp++ ) {
            *tmp = HIP_zlaran( &ran );
        }
    }
    else {
        /*
         * Don't need to check against BLK_X in this branch,
         * and needed for unroll
         */
#pragma unroll
        for ( i = 0; x + i < m; ++i, tmp++ ) {
            *tmp = HIP_zlaran( &ran );
        }
    }
}

extern "C"
int HIP_zplrnt( int m, int n, hipDoubleComplex *A, int lda,
                int bigM, int m0, int n0, unsigned long long int seed,
                hipblasHandle_t handle )
{
    hipStream_t stream;
    hipError_t err;

    if ( m < 0 ) {
        return -1;
    }
    else if ( n < 0 ) {
        return -2;
    }
    else if ( lda < chameleon_max(1, m) ) {
        return -4;
    }
    else if ( bigM < 0 ) {
        return -5;
    }
    else if ( m0 < 0 ) {
        return -6;
    }
    else if ( n0 < 0 ) {
        return -7;
    }

    if ( ( m == 0 ) || ( n == 0 ) ) {
        return CHAMELEON_SUCCESS;
    }

    dim3 threads( 1, BLK_Y );
    dim3 grid( chameleon_ceil( m, BLK_X ), chameleon_ceil( n, BLK_Y ) );

    hipblasGetStream( handle, &stream );
    hip_zplrnt_kernel<<<grid, threads, 0, stream>>>
        ( m, n, A, lda, bigM, m0, n0, seed );

    err = hipGetLastError();
    if ( err != hipSuccess ) {
        fprintf( stderr, "HIP_zplrnt failed to launch HIP kernel %s\n", hipGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }

    return CHAMELEON_SUCCESS;
}
