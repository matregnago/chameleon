/**
 *
 * @file gpuhipblas_random.h
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon gpuhipblas random number generator
 *
 * @version 1.4.0
 * @author Piotr Luszczek
 * @author Mathieu Faverge
 * @author Brieuc Nicolas
 * @author Loris Lucido
 * @date 2026-05-27
 *
 */
#ifndef _chameleon_gpuhipblas_random_h_
#define _chameleon_gpuhipblas_random_h_

#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

__device__ __forceinline__ unsigned long long
HIP_rnd64_jump( unsigned long long int n, unsigned long long int seed )
{
    unsigned long long int a_k, c_k, ran;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    while ( n ) {
        if ( n & 1 ) {
            ran = a_k * ran + c_k;
        }
        c_k *= (a_k + 1);
        a_k *= a_k;
        n >>= 1;
    }

    return ran;
}

__device__ __forceinline__ float
HIP_slaran( unsigned long long int *ran )
{
    float value = 0.5f - (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return value;
}

__device__ __forceinline__ double
HIP_dlaran( unsigned long long int *ran )
{
    double value = 0.5 - (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return value;
}

__device__ __forceinline__ hipFloatComplex
HIP_claran( unsigned long long int *ran )
{
    float real = 0.5f - (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    float imag = 0.5f - (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return make_hipFloatComplex( real, imag );
}

__device__ __forceinline__ hipDoubleComplex
HIP_zlaran( unsigned long long int *ran )
{
    double real = 0.5 - (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    double imag = 0.5 - (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return make_hipDoubleComplex( real, imag );
}

#endif /* _chameleon_gpuhipblas_random_h_ */
