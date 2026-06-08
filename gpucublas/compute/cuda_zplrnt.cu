/**
 *
 * @file cuda_zplrnt.cu
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon cuda_zplrnt GPU kernel
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2025-12-17
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
#define BLK_Y 64

/*
 * Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
 * Each block has BLK_Y threads.
 * Each thread loops across one column, updating BLK_X entries.
 */
__global__
void cuda_zplrnt_kernel( int m, int n, cuDoubleComplex *A, int lda,
                         int bigM, int m0, int n0, unsigned long long int seed )
{
    int i;
    cuDoubleComplex *tmp;

    /* Tile indexes */
    /* threadIdx.x is very likely to be 0 */
    int x = blockIdx.x * BLK_X + threadIdx.x;
    int y = blockIdx.y * BLK_Y + threadIdx.y;

    unsigned long long int jump = (unsigned long long int)(m0 + x)
                                + (unsigned long long int)(n0 + y) * (unsigned long long int)bigM;

    unsigned long long int ran = CUDA_rnd64_jump( NBELEM * jump, seed );

    /* check if full block-column */
    bool full = (x + BLK_X <= m);

    /* do only cols inside matrix */
    if ( y >= n ) {
        return;
    }

    tmp = A + x + y * lda;
    if ( full ) {
        /* full block-column */
        #pragma unroll
        for( i=0; i < BLK_X; ++i, tmp++ ) {
            *tmp = CUDA_zlaran( &ran );
        }
    }
    else {
        /* partial block-column */
        #pragma unroll
        /* Don't need to check against BLK_X in this branch,
           and needed for unroll */
        for( i=0; x+i < m; ++i, tmp++ ) {
            *tmp = CUDA_zlaran( &ran );
        }
    }
}

/**
 *
 * @ingroup CUDA_CHAMELEON_Complex64_t
 *
 * CUDA_zplrnt Generate a random matrix
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] a
 *          Matrix of size LDA-by-N
 *
 * @param[in] lda
 *          Leading dimension of the array A. LDA >= max(1,k), with k=M, if
 *          trans = ChamNoTrans, and k=N otherwise.
 *
 * @param[in] bigM
 *          number of rows of matrix containing A
 *
 * @param[in] m0
 *          row block offset of matrix containing A
 *
 * @param[in] n0
 *          column block offset of matrix containing A
 *
 * @param[in] seed
 *          random seed
 *
 *  @param[out]
 *    -     = 0:  successful exit.
 *    -     < 0:  if INFO = -i, the i-th argument had an illegal value
 *    -     = 1:  an entry of the matrix A is greater than the COMPLEX
 *                overflow threshold, in this case, the content
 *                of SA on exit is unspecified.
 *
 *  @param[in] handle
 *          Cublas handle to execute in.
 *
 **/
extern "C" int
CUDA_zplrnt( int m, int n, cuDoubleComplex *A, int lda,
             int bigM, int m0, int n0, unsigned long long int seed,
             cublasHandle_t handle )
{
    cudaStream_t stream;
    cudaError_t  err;

    if ( m < 0 ) {
        return -1;
    }
    else if ( n < 0 ) {
        return -2;
    }
    else if ( lda < chameleon_max(1,m) ) {
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

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return CHAMELEON_SUCCESS;
    }

    dim3 threads( 1, BLK_Y ); /* Distribution by column */
    dim3 grid( chameleon_ceil( m, BLK_X ), chameleon_ceil( n, BLK_Y ) );

    cublasGetStream( handle, &stream );

    cuda_zplrnt_kernel<<< grid, threads, 0, stream >>>( m, n, A, lda, bigM, m0, n0, seed );

    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        fprintf( stderr, "CUDA_zplrnt failed to launch CUDA kernel %s\n", cudaGetErrorString(err) );
        return CHAMELEON_ERR_UNEXPECTED;
    }

    return CHAMELEON_SUCCESS;
}
