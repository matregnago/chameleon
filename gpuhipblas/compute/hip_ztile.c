/**
 *
 * @file hip_ztile.c
 *
 * @copyright 2026-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Chameleon HIP kernel interface from CHAM_tile_t layout.
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2026-05-05
 * @precisions normal z -> c d s
 *
 */
#include "gpuhipblas.h"
#include "gpuhipblas/gpuhipblas_ztile.h"

int
THIP_zgeadd( cham_trans_t            trans,
             int                     m,
             int                     n,
             const hipDoubleComplex *alpha,
             const CHAM_tile_t      *A,
             const hipDoubleComplex *beta,
             CHAM_tile_t            *B,
             hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zgeadd( trans, m, n, alpha, (const hipDoubleComplex *)A->mat, A->ld, beta, (hipDoubleComplex *)B->mat, B->ld, handle );
}

int
THIP_zgemm( cham_trans_t            transA,
            cham_trans_t            transB,
            int                     m,
            int                     n,
            int                     k,
            const hipDoubleComplex *alpha,
            const CHAM_tile_t      *A,
            const CHAM_tile_t      *B,
            const hipDoubleComplex *beta,
            CHAM_tile_t            *C,
            hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zgemm( transA, transB, m, n, k, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      (const hipDoubleComplex *)B->mat, B->ld,
                      beta,
                      (hipDoubleComplex *)C->mat, C->ld,
                      handle );
}

#if defined( PRECISION_z ) || defined( PRECISION_c )
int
THIP_zhemm( cham_side_t             side,
            cham_uplo_t             uplo,
            int                     m,
            int                     n,
            const hipDoubleComplex *alpha,
            const CHAM_tile_t      *A,
            const CHAM_tile_t      *B,
            const hipDoubleComplex *beta,
            CHAM_tile_t            *C,
            hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zhemm( side, uplo, m, n, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      (const hipDoubleComplex *)B->mat, B->ld,
                      beta,
                      (hipDoubleComplex *)C->mat, C->ld,
                      handle );
}

int
THIP_zher2k( cham_uplo_t             uplo,
             cham_trans_t            trans,
             int                     n,
             int                     k,
             const hipDoubleComplex *alpha,
             const CHAM_tile_t      *A,
             const CHAM_tile_t      *B,
             const double           *beta,
             CHAM_tile_t            *C,
             hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zher2k( uplo, trans, n, k, alpha,
                       (const hipDoubleComplex *)A->mat, A->ld,
                       (const hipDoubleComplex *)B->mat, B->ld,
                       beta,
                       (hipDoubleComplex *)C->mat, C->ld,
                       handle );
}

int
THIP_zherk( cham_uplo_t        uplo,
            cham_trans_t       trans,
            int                n,
            int                k,
            const double      *alpha,
            const CHAM_tile_t *A,
            const double      *beta,
            CHAM_tile_t       *C,
            hipblasHandle_t    handle )
{
    gpuhipblas_kernel_trace( A, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zherk( uplo, trans, n, k, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      beta,
                      (hipDoubleComplex *)C->mat, C->ld,
                      handle );
}
#endif

int
THIP_zsymm( cham_side_t             side,
            cham_uplo_t             uplo,
            int                     m,
            int                     n,
            const hipDoubleComplex *alpha,
            const CHAM_tile_t      *A,
            const CHAM_tile_t      *B,
            const hipDoubleComplex *beta,
            CHAM_tile_t            *C,
            hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zsymm( side, uplo, m, n, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      (const hipDoubleComplex *)B->mat, B->ld,
                      beta,
                      (hipDoubleComplex *)C->mat, C->ld,
                      handle );
}

int
THIP_zsyr2k( cham_uplo_t             uplo,
             cham_trans_t            trans,
             int                     n,
             int                     k,
             const hipDoubleComplex *alpha,
             const CHAM_tile_t      *A,
             const CHAM_tile_t      *B,
             const hipDoubleComplex *beta,
             CHAM_tile_t            *C,
             hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zsyr2k( uplo, trans, n, k, alpha,
                       (const hipDoubleComplex *)A->mat, A->ld,
                       (const hipDoubleComplex *)B->mat, B->ld,
                       beta,
                       (hipDoubleComplex *)C->mat, C->ld,
                       handle );
}

int
THIP_zsyrk( cham_uplo_t             uplo,
            cham_trans_t            trans,
            int                     n,
            int                     k,
            const hipDoubleComplex *alpha,
            const CHAM_tile_t      *A,
            const hipDoubleComplex *beta,
            CHAM_tile_t            *C,
            hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_zsyrk( uplo, trans, n, k, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      beta,
                      (hipDoubleComplex *)C->mat, C->ld,
                      handle );
}

int
THIP_ztrmm( cham_side_t             side,
            cham_uplo_t             uplo,
            cham_trans_t            transA,
            cham_diag_t             diag,
            int                     m,
            int                     n,
            const hipDoubleComplex *alpha,
            const CHAM_tile_t      *A,
            CHAM_tile_t            *B,
            hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    return HIP_ztrmm( side, uplo, transA, diag, m, n, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      (hipDoubleComplex *)B->mat, B->ld,
                      handle );
}

int
THIP_ztrsm( cham_side_t             side,
            cham_uplo_t             uplo,
            cham_trans_t            transA,
            cham_diag_t             diag,
            int                     m,
            int                     n,
            const hipDoubleComplex *alpha,
            const CHAM_tile_t      *A,
            CHAM_tile_t            *B,
            hipblasHandle_t         handle )
{
    gpuhipblas_kernel_trace( A, B );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    return HIP_ztrsm( side, uplo, transA, diag, m, n, alpha,
                      (const hipDoubleComplex *)A->mat, A->ld,
                      (hipDoubleComplex *)B->mat, B->ld,
                      handle );
}

/* Not a Double Complex Function, but must only be defined once */
#if defined(PRECISION_z)
int
THIP_hgemm( cham_trans_t       transA,
            cham_trans_t       transB,
            int                m,
            int                n,
            int                k,
            const hipblasHalf *alpha,
            const CHAM_tile_t *A,
            const CHAM_tile_t *B,
            const hipblasHalf *beta,
            CHAM_tile_t       *C,
            hipblasHandle_t    handle )
{
    gpuhipblas_kernel_trace( A, B, C );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
    assert( C->format & CHAMELEON_TILE_FULLRANK );
    return HIP_hgemm( transA, transB, m, n, k, alpha,
                      (const hipblasHalf *)A->mat, A->ld,
                      (const hipblasHalf *)B->mat, B->ld,
                      beta,
                      (hipblasHalf *)C->mat, C->ld,
                      handle );
}
#endif
