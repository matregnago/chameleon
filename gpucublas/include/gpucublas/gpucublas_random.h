/**
 *
 * @file gpucublas_random.h
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon gpucublas random number generator
 *
 * @version 1.4.0
 * @author Piotr Luszczek
 * @author Mathieu Faverge
 * @author Brieuc Nicolas
 * @date 2025-12-16
 *
 */
#ifndef _chameleon_gpucublas_random_h_
#define _chameleon_gpucublas_random_h_

/* Can't include coreblas/random.h because complex.h is not supported by nvcc */
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

__device__ __forceinline__ unsigned long long
CUDA_rnd64_jump(unsigned long long int n, unsigned long long int seed ) {
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    for (i = 0; n; n >>= 1, ++i) {
        if (n & 1) {
            ran = a_k * ran + c_k;
        }
        c_k *= (a_k + 1);
        a_k *= a_k;
    }

    return ran;
}

__device__ __forceinline__ float
CUDA_slaran( unsigned long long int *ran )
{
    float value = 0.5 - (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return value;
}

__device__ __forceinline__ double
CUDA_dlaran( unsigned long long int *ran )
{
    double value = 0.5 - (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return value;
}

__device__ __forceinline__ cuFloatComplex
CUDA_claran( unsigned long long int *ran )
{
    float real = 0.5 - (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    float imag = 0.5 - (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return make_cuFloatComplex( real, imag );
}

__device__ __forceinline__ cuDoubleComplex
CUDA_zlaran( unsigned long long int *ran )
{
    double real = 0.5 - (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    double imag = 0.5 - (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return make_cuDoubleComplex( real, imag );
}

#endif /* _chameleon_gpucublas_random_h_ */
