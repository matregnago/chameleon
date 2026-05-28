/**
 *
 * @file gpuhipblas_ztile.h
 *
 * @copyright 2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Chameleon HIP tile-based kernel interface header
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @author Florent Pruvost
 * @date 2026-05-05
 * @precisions normal z -> c d s
 *
 */
#ifndef _gpuhipblas_ztile_h_
#define _gpuhipblas_ztile_h_

#include "chameleon/struct.h"
#include <hipblas/hipblas.h>

/**
 *  Declarations of HIP tile-based kernels - alphabetical order
 */
int THIP_zgeadd( cham_trans_t trans, int m, int n, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const hipDoubleComplex *beta, CHAM_tile_t *B, hipblasHandle_t handle );
int THIP_zgemm( cham_trans_t transa, cham_trans_t transb, int m, int n, int k, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const CHAM_tile_t *B, const hipDoubleComplex *beta, CHAM_tile_t *C, hipblasHandle_t handle );
#if defined(PRECISION_z) || defined(PRECISION_c)
int THIP_zhemm( cham_side_t side, cham_uplo_t uplo, int m, int n, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const CHAM_tile_t *B, const hipDoubleComplex *beta, CHAM_tile_t *C, hipblasHandle_t handle );
int THIP_zher2k( cham_uplo_t uplo, cham_trans_t trans, int n, int k, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const CHAM_tile_t *B, const double *beta, CHAM_tile_t *C, hipblasHandle_t handle );
int THIP_zherk( cham_uplo_t uplo, cham_trans_t trans, int n, int k, const double *alpha, const CHAM_tile_t *A, const double *beta, CHAM_tile_t *C, hipblasHandle_t handle );
#endif
int THIP_zlacpy( cham_uplo_t uplo, int M, int N, const CHAM_tile_t *A, CHAM_tile_t *B, hipblasHandle_t handle );
int THIP_zlacpyx( cham_uplo_t uplo, int M, int N, int displA, const CHAM_tile_t *A, int LDA, int displB, CHAM_tile_t *B, int LDB, hipblasHandle_t handle );
int THIP_zlaset( cham_uplo_t uplo, int n1, int n2, const hipDoubleComplex *alpha, const hipDoubleComplex *beta, CHAM_tile_t *A, hipblasHandle_t handle );
int THIP_zlatro( cham_uplo_t uplo, cham_trans_t trans, int M, int N, const CHAM_tile_t *A, CHAM_tile_t *B, hipblasHandle_t handle );
#if defined( PRECISION_z ) || defined( PRECISION_c )
int THIP_zplghe( const double *bump, int m, int n, CHAM_tile_t *A, int bigM, int m0, int n0, unsigned long long int seed, hipblasHandle_t handle );
#endif
int THIP_zplgsy( const hipDoubleComplex *bump, int m, int n, CHAM_tile_t *A, int bigM, int m0, int n0, unsigned long long int seed, hipblasHandle_t handle );
int THIP_zplrnt( int m, int n, CHAM_tile_t *A, int bigM, int m0, int n0, unsigned long long int seed, hipblasHandle_t handle );
int THIP_zsymm( cham_side_t side, cham_uplo_t uplo, int m, int n, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const CHAM_tile_t *B, const hipDoubleComplex *beta, CHAM_tile_t *C, hipblasHandle_t handle );
int THIP_zsyr2k( cham_uplo_t uplo, cham_trans_t trans, int n, int k, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const CHAM_tile_t *B, const hipDoubleComplex *beta, CHAM_tile_t *C, hipblasHandle_t handle );
int THIP_zsyrk( cham_uplo_t uplo, cham_trans_t trans, int n, int k, const hipDoubleComplex *alpha, const CHAM_tile_t *A, const hipDoubleComplex *beta, CHAM_tile_t *C, hipblasHandle_t handle );
int THIP_ztrmm( cham_side_t side, cham_uplo_t uplo, cham_trans_t transa, cham_diag_t diag, int m, int n, const hipDoubleComplex *alpha, const CHAM_tile_t *A, CHAM_tile_t *B, hipblasHandle_t handle );
int THIP_ztrsm( cham_side_t side, cham_uplo_t uplo, cham_trans_t transa, cham_diag_t diag, int m, int n, const hipDoubleComplex *alpha, const CHAM_tile_t *A, CHAM_tile_t *B, hipblasHandle_t handle );

#endif /* _gpuhipblas_ztile_h_ */
