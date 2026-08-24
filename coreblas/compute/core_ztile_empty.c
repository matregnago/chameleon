/**
 *
 * @file core_ztile_empty.c
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Chameleon CPU kernel interface from CHAM_tile_t layout to the real one.
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @author Abel Calluaud
 * @author Matteo Marcos
 * @author Brieuc Nicolas
 * @date 2025-12-19
 * @precisions normal z -> c d s
 *
 */
#include "coreblas.h"
#include "coreblas/coreblas_ztile.h"

#if defined( CHAMELEON_USE_HMATOSS )
#include "coreblas/hmat.h"
#endif

#if defined( PRECISION_z ) || defined( PRECISION_c )
void
TCORE_dlag2z( __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              const CHAM_tile_t *A,
              CHAM_tile_t       *B )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( B->format & CHAMELEON_TILE_FULLRANK );
}
#endif

void
TCORE_dzasum( __attribute__((unused)) cham_store_t       storev,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              __attribute__((unused)) double *           work )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

int
TCORE_zaxpy( __attribute__((unused)) int                   M,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
                                     const CHAM_tile_t *   A,
             __attribute__((unused)) int                   incA,
                                     CHAM_tile_t *         B,
             __attribute__((unused)) int                   incB )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zgeadd( __attribute__((unused)) cham_trans_t          trans,
              __attribute__((unused)) int                   M,
              __attribute__((unused)) int                   N,
              __attribute__((unused)) CHAMELEON_Complex64_t alpha,
              __attribute__((unused)) const CHAM_tile_t *   A,
              __attribute__((unused)) CHAMELEON_Complex64_t beta,
              __attribute__((unused)) CHAM_tile_t *         B )
{
    coreblas_kernel_trace( A, B );
    return 0;
}

int
TCORE_zgelqt( __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    IB,
              __attribute__((unused)) CHAM_tile_t *          A,
              __attribute__((unused)) CHAM_tile_t *          T,
              __attribute__((unused)) CHAMELEON_Complex64_t *TAU,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK )
{
    coreblas_kernel_trace( A, T );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

void
TCORE_zgemv( __attribute__((unused)) cham_trans_t trans,
             __attribute__((unused)) int M,
             __attribute__((unused)) int N,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *A,
             const CHAM_tile_t *x,
             __attribute__((unused)) int incX,
             __attribute__((unused)) CHAMELEON_Complex64_t beta,
             CHAM_tile_t *y,
             __attribute__((unused)) int incY )
{
    coreblas_kernel_trace( A, x, y );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
    assert( x->format & CHAMELEON_TILE_FULLRANK );
    assert( y->format & CHAMELEON_TILE_FULLRANK );
}

void
TCORE_zgemm( __attribute__((unused)) cham_trans_t          transA,
             __attribute__((unused)) cham_trans_t          transB,
             __attribute__((unused)) int                   M,
             __attribute__((unused)) int                   N,
             __attribute__((unused)) int                   K,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *   A,
             const CHAM_tile_t *   B,
             __attribute__((unused)) CHAMELEON_Complex64_t beta,
             CHAM_tile_t *         C )
{
    coreblas_kernel_trace( A, B, C );
}

int
TCORE_zgeqrt( __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    IB,
              CHAM_tile_t *          A,
              CHAM_tile_t *          T,
              __attribute__((unused)) CHAMELEON_Complex64_t *TAU,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK )
{
    coreblas_kernel_trace( A, T );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zgessm( __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              __attribute__((unused)) int K,
              __attribute__((unused)) int IB,
              __attribute__((unused)) const int *IPIV,
              const CHAM_tile_t *L,
              CHAM_tile_t *A )
{
    coreblas_kernel_trace( L, A );
    assert( L->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zgessq( __attribute__((unused)) cham_store_t storev,
              __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              const CHAM_tile_t *A,
              CHAM_tile_t *sclssq )
{
    coreblas_kernel_trace( A, sclssq );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( sclssq->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zgetrf( __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              CHAM_tile_t *A,
              __attribute__((unused)) int *IPIV,
              __attribute__((unused)) int *INFO )
{
    coreblas_kernel_trace( A );
    return 0;
}

int
TCORE_zgetrf_incpiv( __attribute__((unused)) int M,
                     __attribute__((unused)) int N,
                     __attribute__((unused)) int IB,
                     CHAM_tile_t *A,
                     __attribute__((unused)) int *IPIV,
                     __attribute__((unused)) int *INFO )
{
    coreblas_kernel_trace( A );
    return 0;
}

int
TCORE_zgetrf_nopiv( __attribute__((unused)) int M,
                    __attribute__((unused)) int N,
                    __attribute__((unused)) int IB,
                    CHAM_tile_t *A,
                    __attribute__((unused)) int *INFO )
{
    coreblas_kernel_trace( A );
    return 0;
}

void
TCORE_zhe2ge( __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              const CHAM_tile_t *A,
              CHAM_tile_t *B )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

#if defined( PRECISION_z ) || defined( PRECISION_c )
void
TCORE_zhemm( __attribute__((unused)) cham_side_t           side,
             __attribute__((unused)) cham_uplo_t           uplo,
             __attribute__((unused)) int                   M,
             __attribute__((unused)) int                   N,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *   A,
             const CHAM_tile_t *   B,
             __attribute__((unused)) CHAMELEON_Complex64_t beta,
             CHAM_tile_t *         C )
{
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zherk( __attribute__((unused)) cham_uplo_t        uplo,
             __attribute__((unused)) cham_trans_t       trans,
             __attribute__((unused)) int                N,
             __attribute__((unused)) int                K,
             __attribute__((unused)) double             alpha,
             const CHAM_tile_t *A,
             __attribute__((unused)) double             beta,
             CHAM_tile_t *      C )
{
    coreblas_kernel_trace( A, C );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zher2k( __attribute__((unused)) cham_uplo_t           uplo,
              __attribute__((unused)) cham_trans_t          trans,
              __attribute__((unused)) int                   N,
              __attribute__((unused)) int                   K,
              __attribute__((unused)) CHAMELEON_Complex64_t alpha,
              const CHAM_tile_t *   A,
              const CHAM_tile_t *   B,
              __attribute__((unused)) double                beta,
              CHAM_tile_t *         C )
{
    coreblas_kernel_trace( A, B, C );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}
#endif

int
TCORE_zherfb( __attribute__((unused)) cham_uplo_t            uplo,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    K,
              __attribute__((unused)) int                    IB,
              __attribute__((unused)) int                    NB,
              const CHAM_tile_t *    A,
              const CHAM_tile_t *    T,
              CHAM_tile_t *          C,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK,
              __attribute__((unused)) int                    ldwork )
{
    coreblas_kernel_trace( A, T, C );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

#if defined( PRECISION_z ) || defined( PRECISION_c )
int
TCORE_zhessq( __attribute__((unused)) cham_store_t       storev,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              CHAM_tile_t *      sclssq )
{
    coreblas_kernel_trace( A, sclssq );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( sclssq->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}
#endif

void
TCORE_zlacpy( __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              __attribute__((unused)) const CHAM_tile_t *A,
              __attribute__((unused)) CHAM_tile_t *B )
{
    return;
}

void
TCORE_zlacpyx( __attribute__((unused)) cham_uplo_t uplo,
               __attribute__((unused)) int M,
               __attribute__((unused)) int N,
               __attribute__((unused)) int displA,
               __attribute__((unused)) const CHAM_tile_t *A,
               __attribute__((unused)) int LDA,
               __attribute__((unused)) int displB,
               __attribute__((unused)) CHAM_tile_t *B,
               __attribute__((unused)) int LDB )
{
    return;
}

void
TCORE_zlange( __attribute__((unused)) cham_normtype_t    norm,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              __attribute__((unused)) double *           work,
              __attribute__((unused)) double *           normA )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

#if defined( PRECISION_z ) || defined( PRECISION_c )
void
TCORE_zlanhe( __attribute__((unused)) cham_normtype_t    norm,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              __attribute__((unused)) double *           work,
              __attribute__((unused)) double *           normA )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}
#endif

void
TCORE_zlansy( __attribute__((unused)) cham_normtype_t    norm,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              __attribute__((unused)) double *           work,
              __attribute__((unused)) double *           normA )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zlantr( __attribute__((unused)) cham_normtype_t    norm,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) cham_diag_t        diag,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              __attribute__((unused)) double *           work,
              __attribute__((unused)) double *           normA )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

int
TCORE_zlascal( __attribute__((unused)) cham_uplo_t uplo,
               __attribute__((unused)) int m,
               __attribute__((unused)) int n,
               __attribute__((unused)) CHAMELEON_Complex64_t alpha,
               CHAM_tile_t *A )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

void
TCORE_zlaset( __attribute__((unused)) cham_uplo_t           uplo,
              __attribute__((unused)) int                   n1,
              __attribute__((unused)) int                   n2,
              __attribute__((unused)) CHAMELEON_Complex64_t alpha,
              __attribute__((unused)) CHAMELEON_Complex64_t beta,
              CHAM_tile_t *         A )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zlaset2( __attribute__((unused)) cham_uplo_t uplo,
               __attribute__((unused)) int n1,
               __attribute__((unused)) int n2,
               __attribute__((unused)) CHAMELEON_Complex64_t alpha,
               CHAM_tile_t *A )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

int
TCORE_zlaswp_get( __attribute__((unused)) cham_side_t side,
                  __attribute__((unused)) int m0,
                  __attribute__((unused)) int m,
                  __attribute__((unused)) int n,
                  __attribute__((unused)) int k,
                  CHAM_tile_t *A,
                  CHAM_tile_t *B,
                  __attribute__((unused)) const int *perm )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zlaswp_get_idx( __attribute__((unused)) cham_side_t side,
                      __attribute__((unused)) int         m0,
                      __attribute__((unused)) int         m,
                      __attribute__((unused)) int         n,
                      __attribute__((unused)) int         k,
                      const CHAM_tile_t *A,
                      CHAM_tile_t       *B,
                      __attribute__((unused)) int        perm_m,
                      __attribute__((unused)) int        perm_mt,
                      __attribute__((unused)) const int *perm_idx )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zlaswp_set( __attribute__((unused)) cham_side_t side,
                  __attribute__((unused)) int m0,
                  __attribute__((unused)) int m,
                  __attribute__((unused)) int n,
                  __attribute__((unused)) int k,
                  CHAM_tile_t *A,
                  CHAM_tile_t *B,
                  __attribute__((unused)) const int *invp )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zlaswp_set_idx( __attribute__((unused)) cham_side_t side,
                      __attribute__((unused)) int         m0,
                      __attribute__((unused)) int         m,
                      __attribute__((unused)) int         n,
                      __attribute__((unused)) int         k,
                      const CHAM_tile_t *A,
                      CHAM_tile_t       *B,
                      __attribute__((unused)) int        invp_m,
                      __attribute__((unused)) int        invp_mt,
                      __attribute__((unused)) const int *invp_idx )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zlatro( __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) cham_trans_t       trans,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              CHAM_tile_t *      B )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

void
TCORE_zlauum( __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) int N,
              CHAM_tile_t *A )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

#if defined( PRECISION_z ) || defined( PRECISION_c )
void
TCORE_zplghe( __attribute__((unused)) double                 bump,
              __attribute__((unused)) int                    m,
              __attribute__((unused)) int                    n,
              CHAM_tile_t *          A,
              __attribute__((unused)) int                    bigM,
              __attribute__((unused)) int                    m0,
              __attribute__((unused)) int                    n0,
              __attribute__((unused)) unsigned long long int seed )
{
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}
#endif

void
TCORE_zplgsy( __attribute__((unused)) CHAMELEON_Complex64_t  bump,
              __attribute__((unused)) int                    m,
              __attribute__((unused)) int                    n,
              CHAM_tile_t *          A,
              __attribute__((unused)) int                    bigM,
              __attribute__((unused)) int                    m0,
              __attribute__((unused)) int                    n0,
              __attribute__((unused)) unsigned long long int seed )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zplrnt( __attribute__((unused)) int                    m,
              __attribute__((unused)) int                    n,
              CHAM_tile_t *          A,
              __attribute__((unused)) int                    bigM,
              __attribute__((unused)) int                    m0,
              __attribute__((unused)) int                    n0,
              __attribute__((unused)) unsigned long long int seed )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zpotrf( __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) int n,
              CHAM_tile_t *A,
              __attribute__((unused)) int *INFO )
{
    coreblas_kernel_trace( A );
    return;
}

int
TCORE_zssssm( __attribute__((unused)) int                M1,
              __attribute__((unused)) int                N1,
              __attribute__((unused)) int                M2,
              __attribute__((unused)) int                N2,
              __attribute__((unused)) int                K,
              __attribute__((unused)) int                IB,
              CHAM_tile_t *      A1,
              CHAM_tile_t *      A2,
              const CHAM_tile_t *L1,
              const CHAM_tile_t *L2,
              __attribute__((unused)) const int *        IPIV )
{
    coreblas_kernel_trace( A1, A2, L1, L2 );
    assert( A1->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A2->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( L1->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( L2->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

void
TCORE_zsymm( __attribute__((unused)) cham_side_t           side,
             __attribute__((unused)) cham_uplo_t           uplo,
             __attribute__((unused)) int                   M,
             __attribute__((unused)) int                   N,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *   A,
             const CHAM_tile_t *   B,
             __attribute__((unused)) CHAMELEON_Complex64_t beta,
             CHAM_tile_t *         C )
{
    coreblas_kernel_trace( A, B, C );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zsyrk( __attribute__((unused)) cham_uplo_t           uplo,
             __attribute__((unused)) cham_trans_t          trans,
             __attribute__((unused)) int                   N,
             __attribute__((unused)) int                   K,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *   A,
             __attribute__((unused)) CHAMELEON_Complex64_t beta,
             CHAM_tile_t *         C )
{
    coreblas_kernel_trace( A, C );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_zsyr2k( __attribute__((unused)) cham_uplo_t           uplo,
              __attribute__((unused)) cham_trans_t          trans,
              __attribute__((unused)) int                   N,
              __attribute__((unused)) int                   K,
              __attribute__((unused)) CHAMELEON_Complex64_t alpha,
              const CHAM_tile_t *   A,
              const CHAM_tile_t *   B,
              __attribute__((unused)) CHAMELEON_Complex64_t beta,
              CHAM_tile_t *         C )
{
    coreblas_kernel_trace( A, B, C );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

int
TCORE_zsyssq( __attribute__((unused)) cham_store_t       storev,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              CHAM_tile_t *      sclssq )
{
    coreblas_kernel_trace( A, sclssq );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( sclssq->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

#if defined( PRECISION_z ) || defined( PRECISION_c )
int
TCORE_zsytf2_nopiv( __attribute__((unused)) cham_uplo_t uplo,
                     __attribute__((unused)) int n,
                     CHAM_tile_t *A )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}
#endif

int
TCORE_ztplqt( __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    L,
              __attribute__((unused)) int                    IB,
              CHAM_tile_t *          A,
              CHAM_tile_t *          B,
              CHAM_tile_t *          T,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK )
{
    coreblas_kernel_trace( A, B, T );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_ztpmlqt( __attribute__((unused)) cham_side_t            side,
               __attribute__((unused)) cham_trans_t           trans,
               __attribute__((unused)) int                    M,
               __attribute__((unused)) int                    N,
               __attribute__((unused)) int                    K,
               __attribute__((unused)) int                    L,
               __attribute__((unused)) int                    IB,
               const CHAM_tile_t *    V,
               const CHAM_tile_t *    T,
               CHAM_tile_t *          A,
               CHAM_tile_t *          B,
               __attribute__((unused)) CHAMELEON_Complex64_t *WORK )
{
    coreblas_kernel_trace( V, T, A, B );
    assert( V->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_ztpmqrt( __attribute__((unused)) cham_side_t            side,
               __attribute__((unused)) cham_trans_t           trans,
               __attribute__((unused)) int                    M,
               __attribute__((unused)) int                    N,
               __attribute__((unused)) int                    K,
               __attribute__((unused)) int                    L,
               __attribute__((unused)) int                    IB,
               const CHAM_tile_t *    V,
               const CHAM_tile_t *    T,
               CHAM_tile_t *          A,
               CHAM_tile_t *          B,
               __attribute__((unused)) CHAMELEON_Complex64_t *WORK )
{
    coreblas_kernel_trace( V, T, A, B );
    assert( V->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_ztpqrt( __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    L,
              __attribute__((unused)) int                    IB,
              CHAM_tile_t *          A,
              CHAM_tile_t *          B,
              CHAM_tile_t *          T,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK )
{
    coreblas_kernel_trace( A, B, T );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_ztradd( __attribute__((unused)) cham_uplo_t           uplo,
              __attribute__((unused)) cham_trans_t          trans,
              __attribute__((unused)) int                   M,
              __attribute__((unused)) int                   N,
              __attribute__((unused)) CHAMELEON_Complex64_t alpha,
              const CHAM_tile_t *   A,
              __attribute__((unused)) CHAMELEON_Complex64_t beta,
              CHAM_tile_t *         B )
{
    coreblas_kernel_trace( A, B );
    return 0;
}

void
TCORE_ztrasm( __attribute__((unused)) cham_store_t       storev,
              __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) cham_diag_t        diag,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              __attribute__((unused)) double *           work )
{
    coreblas_kernel_trace( A );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_ztrmm( __attribute__((unused)) cham_side_t           side,
             __attribute__((unused)) cham_uplo_t           uplo,
             __attribute__((unused)) cham_trans_t          transA,
             __attribute__((unused)) cham_diag_t           diag,
             __attribute__((unused)) int                   M,
             __attribute__((unused)) int                   N,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *   A,
             CHAM_tile_t *         B )
{
    coreblas_kernel_trace( A, B );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( B->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

void
TCORE_ztrsm( __attribute__((unused)) cham_side_t           side,
             __attribute__((unused)) cham_uplo_t           uplo,
             __attribute__((unused)) cham_trans_t          transA,
             __attribute__((unused)) cham_diag_t           diag,
             __attribute__((unused)) int                   M,
             __attribute__((unused)) int                   N,
             __attribute__((unused)) CHAMELEON_Complex64_t alpha,
             const CHAM_tile_t *   A,
             CHAM_tile_t *         B )
{
    coreblas_kernel_trace( A, B );
}

int
TCORE_ztrssq( __attribute__((unused)) cham_uplo_t        uplo,
              __attribute__((unused)) cham_diag_t        diag,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              const CHAM_tile_t *A,
              CHAM_tile_t *      sclssq )
{
    coreblas_kernel_trace( A, sclssq );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( sclssq->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

void
TCORE_ztrtri( __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) cham_diag_t diag,
              __attribute__((unused)) int N,
              CHAM_tile_t *A,
              __attribute__((unused)) int *info )
{
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
}

int
TCORE_ztsmlq_hetra1( __attribute__((unused)) cham_side_t            side,
                     __attribute__((unused)) cham_trans_t           trans,
                     __attribute__((unused)) int                    m1,
                     __attribute__((unused)) int                    n1,
                     __attribute__((unused)) int                    m2,
                     __attribute__((unused)) int                    n2,
                     __attribute__((unused)) int                    k,
                     __attribute__((unused)) int                    ib,
                     CHAM_tile_t *          A1,
                     CHAM_tile_t *          A2,
                     const CHAM_tile_t *    V,
                     const CHAM_tile_t *    T,
                     __attribute__((unused)) CHAMELEON_Complex64_t *WORK,
                     __attribute__((unused)) int                    ldwork )
{
    coreblas_kernel_trace( A1, A2, V, T );
    assert( A1->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A2->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( V->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_ztsmqr_hetra1( __attribute__((unused)) cham_side_t            side,
                     __attribute__((unused)) cham_trans_t           trans,
                     __attribute__((unused)) int                    m1,
                     __attribute__((unused)) int                    n1,
                     __attribute__((unused)) int                    m2,
                     __attribute__((unused)) int                    n2,
                     __attribute__((unused)) int                    k,
                     __attribute__((unused)) int                    ib,
                     CHAM_tile_t *          A1,
                     CHAM_tile_t *          A2,
                     const CHAM_tile_t *    V,
                     const CHAM_tile_t *    T,
                     __attribute__((unused)) CHAMELEON_Complex64_t *WORK,
                     __attribute__((unused)) int                    ldwork )
{
    coreblas_kernel_trace( A1, A2, V, T );
    assert( A1->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A2->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( V->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_ztstrf( __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    IB,
              __attribute__((unused)) int                    NB,
              CHAM_tile_t *          U,
              CHAM_tile_t *          A,
              CHAM_tile_t *          L,
              __attribute__((unused)) int *                  IPIV,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK,
              __attribute__((unused)) int                    LDWORK,
              __attribute__((unused)) int *                  INFO )
{
    coreblas_kernel_trace( U, A, L );
    assert( U->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( L->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zunmlq( __attribute__((unused)) cham_side_t            side,
              __attribute__((unused)) cham_trans_t           trans,
              __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    K,
              __attribute__((unused)) int                    IB,
              const CHAM_tile_t *    V,
              const CHAM_tile_t *    T,
              CHAM_tile_t *          C,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK,
              __attribute__((unused)) int                    LDWORK )
{
    coreblas_kernel_trace( V, T, C );
    assert( V->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zunmqr( __attribute__((unused)) cham_side_t            side,
              __attribute__((unused)) cham_trans_t           trans,
              __attribute__((unused)) int                    M,
              __attribute__((unused)) int                    N,
              __attribute__((unused)) int                    K,
              __attribute__((unused)) int                    IB,
              const CHAM_tile_t *    V,
              const CHAM_tile_t *    T,
              CHAM_tile_t *          C,
              __attribute__((unused)) CHAMELEON_Complex64_t *WORK,
              __attribute__((unused)) int                    LDWORK )
{
    coreblas_kernel_trace( V, T, C );
    assert( V->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( T->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( C->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zgesum( __attribute__((unused)) cham_store_t storev,
              __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              const CHAM_tile_t *A,
              CHAM_tile_t *sum )
{
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( sum->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zcesca( __attribute__((unused)) int center,
              __attribute__((unused)) int scale,
              __attribute__((unused)) cham_store_t axis,
              __attribute__((unused)) int                M,
              __attribute__((unused)) int                N,
              __attribute__((unused)) int                Mt,
              __attribute__((unused)) int                Nt,
              const CHAM_tile_t *Gi,
              const CHAM_tile_t *Gj,
              const CHAM_tile_t *G,
              const CHAM_tile_t *Di,
              const CHAM_tile_t *Dj,
              CHAM_tile_t *      A )
{
    assert( Gi->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( Gj->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( G->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( Di->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( Dj->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

int
TCORE_zgram( __attribute__((unused)) cham_uplo_t        uplo,
             __attribute__((unused)) int                M,
             __attribute__((unused)) int                N,
             __attribute__((unused)) int                Mt,
             __attribute__((unused)) int                Nt,
             const CHAM_tile_t *Di,
             const CHAM_tile_t *Dj,
             const CHAM_tile_t *D,
             CHAM_tile_t *      A )
{
    coreblas_kernel_trace( Di, Dj, D, A );
    assert( Di->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( Dj->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( D->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    assert( A->format & (CHAMELEON_TILE_FULLRANK | CHAMELEON_TILE_DESC) );
    return 0;
}

void
TCORE_zprint( __attribute__((unused)) FILE *file,
              __attribute__((unused)) const char *header,
              __attribute__((unused)) cham_uplo_t uplo,
              __attribute__((unused)) int M,
              __attribute__((unused)) int N,
              __attribute__((unused)) int Am,
              __attribute__((unused)) int An,
              const CHAM_tile_t *A )
{
    coreblas_kernel_trace( A );
    assert( A->format & CHAMELEON_TILE_FULLRANK );
}
