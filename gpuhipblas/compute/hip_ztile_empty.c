/**
 *
 * @file hip_ztile.c
 *
 * @copyright 2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Chameleon HIP kernel interface from CHAM_tile_t layout.
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @author Florent Pruvost
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
}
#endif
