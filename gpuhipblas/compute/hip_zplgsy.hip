/**
 *
 * @file hip_zplgsy.cpp
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon hip_zplgsy GPU kernel
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

#if defined(PRECISION_z)
#define addBump( val, bump ) make_hipDoubleComplex( hipCreal(val) + hipCreal(bump), hipCimag(val) + hipCimag(bump) )
#elif defined(PRECISION_c)
#define addBump( val, bump ) make_hipFloatComplex( hipCrealf(val) + hipCrealf(bump), hipCimagf(val) + hipCimagf(bump) )
#else
#define addBump( val, bump ) ((val) + (bump))
#endif

#define BLK_X 32
#define BLK_Y 32

__global__
void hip_zplgsy_diag_kernel( hipDoubleComplex bump, int m, int n, hipDoubleComplex *A, int lda,
                             int bigM, int m0, int n0, unsigned long long int seed )
{
    int minmn = (m < n) ? m : n;
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    if ( (x >= m) || (y >= n) ) {
        return;
    }

    unsigned long long int jump = (unsigned long long int)(m0)
                                + (unsigned long long int)(n0) * (unsigned long long int)bigM;

    if ( (x == y) && (x < minmn) ) {
        unsigned long long int ran = HIP_rnd64_jump( NBELEM * (jump + x + (unsigned long long int)x * (unsigned long long int)bigM), seed );
        hipDoubleComplex val = HIP_zlaran( &ran );
        A[x + x * lda] = addBump( val, bump );
    }
    else if ( (x > y) && (y < minmn) ) {
        unsigned long long int diag_jump = jump + y + (unsigned long long int)y * (unsigned long long int)bigM;
        unsigned long long int ran = HIP_rnd64_jump( NBELEM * (diag_jump + (x - y)), seed );
        A[x + y * lda] = HIP_zlaran( &ran );
    }
    else if ( (x < y) && (x < minmn) ) {
        unsigned long long int diag_jump = jump + x + (unsigned long long int)x * (unsigned long long int)bigM;
        unsigned long long int ran = HIP_rnd64_jump( NBELEM * (diag_jump + (y - x)), seed );
        A[x + y * lda] = HIP_zlaran( &ran );
    }
}

__global__
void hip_zplgsy_lower_kernel( int m, int n, hipDoubleComplex *A, int lda,
                              int bigM, int m0, int n0, unsigned long long int seed )
{
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    if ( (x >= m) || (y >= n) ) {
        return;
    }

    unsigned long long int jump = (unsigned long long int)(m0 + x)
                                + (unsigned long long int)(n0 + y) * (unsigned long long int)bigM;
    unsigned long long int ran = HIP_rnd64_jump( NBELEM * jump, seed );
    A[x + y * lda] = HIP_zlaran( &ran );
}

__global__
void hip_zplgsy_upper_kernel( int m, int n, hipDoubleComplex *A, int lda,
                              int bigM, int m0, int n0, unsigned long long int seed )
{
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    if ( (x >= m) || (y >= n) ) {
        return;
    }

    unsigned long long int jump = (unsigned long long int)(n0 + y)
                                + (unsigned long long int)(m0 + x) * (unsigned long long int)bigM;
    unsigned long long int ran = HIP_rnd64_jump( NBELEM * jump, seed );
    A[x + y * lda] = HIP_zlaran( &ran );
}

extern "C"
int HIP_zplgsy( const hipDoubleComplex *bump, int m, int n, hipDoubleComplex *A, int lda,
                int bigM, int m0, int n0, unsigned long long int seed, hipblasHandle_t handle )
{
    hipStream_t stream;
    hipError_t err;

    if ( m < 0 ) {
        return -2;
    }
    else if ( n < 0 ) {
        return -3;
    }
    else if ( lda < chameleon_max(1, m) ) {
        return -5;
    }
    else if ( bigM < 0 ) {
        return -6;
    }
    else if ( m0 < 0 ) {
        return -7;
    }
    else if ( n0 < 0 ) {
        return -8;
    }

    if ( ( m == 0 ) || ( n == 0 ) ) {
        return CHAMELEON_SUCCESS;
    }

    dim3 threads( BLK_X, BLK_Y );
    dim3 grid( chameleon_ceil( m, BLK_X ), chameleon_ceil( n, BLK_Y ) );

    hipblasGetStream( handle, &stream );

    if ( m0 == n0 ) {
        hip_zplgsy_diag_kernel<<<grid, threads, 0, stream>>>
            ( *bump, m, n, A, lda, bigM, m0, n0, seed );
    }
    else if ( m0 > n0 ) {
        hip_zplgsy_lower_kernel<<<grid, threads, 0, stream>>>
            ( m, n, A, lda, bigM, m0, n0, seed );
    }
    else {
        hip_zplgsy_upper_kernel<<<grid, threads, 0, stream>>>
            ( m, n, A, lda, bigM, m0, n0, seed );
    }

    err = hipGetLastError();
    if ( err != hipSuccess ) {
        fprintf( stderr, "HIP_zplgsy failed to launch HIP kernel %s\n", hipGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }
    return CHAMELEON_SUCCESS;
}
