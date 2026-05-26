
/**
 *
 * @file cuda_zplgsy.cu
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon cuda_zplgsy GPU kernel
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2025-12-18
 * @precisions normal z -> d s c
 *
 */
#include "gpucublas.h"
#include "gpucublas/gpucublas_random.h"

#if defined(PRECISION_z) || defined(PRECISION_c)
#define NBELEM   2
#else
#define NBELEM   1
#endif

#define BLK_X 32
#define BLK_Y 32

#if defined(PRECISION_z)
#define addBump( val, bump ) cuCadd( val, bump )
#elif defined(PRECISION_c)
#define addBump( val, bump ) cuCaddf( val, bump )
#else
#define addBump( val, bump ) val + bump
#endif

/*
 * CUDA kernel for diagonal tiles
 * Generates both lower and upper triangular parts
 */
__global__
void cuda_zplgsy_diag_kernel( cuDoubleComplex bump, int m, int n, cuDoubleComplex *A, int lda,
                              int bigM, int m0, int n0, unsigned long long int seed )
{
    int minmn = min(m, n);

    /* Tile indexes */
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    if (x >= m || y >= n) {
        return;
    }

    unsigned long long int jump = (unsigned long long int)(m0)
                                + (unsigned long long int)(n0) * (unsigned long long int)bigM;

    /* On diagonal tile */
    if (x == y && x < minmn) {
        /* Diagonal element */
        unsigned long long int ran = CUDA_rnd64_jump( NBELEM * (jump + x + (unsigned long long int)x * (unsigned long long int)bigM), seed );
        cuDoubleComplex val = CUDA_zlaran( &ran );
        A[x + x * lda] = addBump(val, bump);
    }
    else if (x > y && y < minmn) {
        /* Lower triangular part */
        unsigned long long int diag_jump = jump + y + (unsigned long long int)y * (unsigned long long int)bigM;
        unsigned long long int ran = CUDA_rnd64_jump( NBELEM * (diag_jump + (x - y)), seed );
        A[x + y * lda] = CUDA_zlaran( &ran );
    }
    else if (x < y && x < minmn) {
        /* Upper triangular part - mirror of lower part */
        unsigned long long int diag_jump = jump + x + (unsigned long long int)x * (unsigned long long int)bigM;
        unsigned long long int ran = CUDA_rnd64_jump( NBELEM * (diag_jump + (y - x)), seed );
        A[x + y * lda] = CUDA_zlaran( &ran );
    }
}

/*
 * CUDA kernel for lower triangular tiles (m0 > n0)
 */
__global__
void cuda_zplgsy_lower_kernel( int m, int n, cuDoubleComplex *A, int lda,
                               int bigM, int m0, int n0, unsigned long long int seed )
{
    /* Tile indexes */
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    if (x >= m || y >= n) {
        return;
    }

    unsigned long long int jump = (unsigned long long int)(m0 + x)
                                + (unsigned long long int)(n0 + y) * (unsigned long long int)bigM;

    unsigned long long int ran = CUDA_rnd64_jump( NBELEM * jump, seed );

    A[x + y * lda] = CUDA_zlaran( &ran );
}

/*
 * CUDA kernel for upper triangular tiles (m0 < n0)
 */
__global__
void cuda_zplgsy_upper_kernel( int m, int n, cuDoubleComplex *A, int lda,
                               int bigM, int m0, int n0, unsigned long long int seed )
{
    /* Tile indexes */
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    if (x >= m || y >= n) {
        return;
    }

    /* For upper part, use transposed coordinates */
    unsigned long long int jump = (unsigned long long int)(n0 + y)
                                + (unsigned long long int)(m0 + x) * (unsigned long long int)bigM;

    unsigned long long int ran = CUDA_rnd64_jump( NBELEM * jump, seed );

    A[x + y * lda] = CUDA_zlaran( &ran );
}

/**
 *
 * @ingroup CUDA_CHAMELEON_Complex64_t
 *
 * CUDA_zplgsy Generate a random symmetric matrix
 *
 * @param[in] bump
 *          The value to add to diagonal elements to ensure positive definiteness
 *
 * @param[in] m
 *          The number of rows of the tile A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the tile A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, uninitialized tile of size m-by-n.
 *          On exit, random symmetric tile.
 *
 * @param[in] lda
 *          Leading dimension of the array A. LDA >= max(1,M).
 *
 * @param[in] bigM
 *          Number of rows of the full matrix containing A
 *
 * @param[in] m0
 *          Row tile offset in the full matrix
 *
 * @param[in] n0
 *          Column tile offset in the full matrix
 *
 * @param[in] seed
 *          Random seed
 *
 * @param[in] handle
 *          Cublas handle to execute in.
 *
 * @retval CHAMELEON_SUCCESS on success
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 **/
extern "C" int
CUDA_zplgsy( const cuDoubleComplex *bump, int m, int n, cuDoubleComplex *A, int lda,
             int bigM, int m0, int n0, unsigned long long int seed,
             cublasHandle_t handle)
{
    cudaStream_t stream;
    cudaError_t  err;

    if ( m < 0 ) {
        return -2;
    }
    else if ( n < 0 ) {
        return -3;
    }
    else if ( lda < chameleon_max(1,m) ) {
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

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return CHAMELEON_SUCCESS;
    }

    dim3 threads(BLK_X, BLK_Y);

    dim3 grid( chameleon_ceil( m, BLK_X ), chameleon_ceil( n, BLK_Y ) );

    cublasGetStream( handle, &stream );

    /* Diagonal tile */
    if ( m0 == n0 ) {
        cuda_zplgsy_diag_kernel<<< grid, threads, 0, stream >>>( *bump, m, n, A, lda, bigM, m0, n0, seed );
    }
    /* Lower triangular tiles */
    else if ( m0 > n0 ) {
        cuda_zplgsy_lower_kernel<<< grid, threads, 0, stream >>>( m, n, A, lda, bigM, m0, n0, seed );
    }
    /* Upper triangular tiles */
    else {
        cuda_zplgsy_upper_kernel<<< grid, threads, 0, stream >>>( m, n, A, lda, bigM, m0, n0, seed );
    }

    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        fprintf( stderr, "CUDA_zplgsy failed to launch CUDA kernel %s\n", cudaGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }

    return CHAMELEON_SUCCESS;
}
