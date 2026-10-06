/**
 *
 * @file parsec/codelet_zlaswp.c
 *
 * @copyright 2023-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC codelets to apply zlaswp on a panel
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Matteo Marcos
 * @author Alycia Lisito
 * @date 2026-10-05
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"
#include <coreblas/cblas_wrapper.h>

/*
 * The permutation vectors are stored as perm/invp or as their index variants
 * (perm_mt >= 0) depending on ipiv->withidx.
 */
static inline void
chameleon_parsec_laswp_perm_args( const CHAM_ipiv_t *ipiv, int ipivk, cham_side_t side,
                                  int Am, int An, int *perm_m, int *perm_mt )
{
    if ( ipiv->withidx ) {
        *perm_m  = ( side == ChamLeft ) ? Am - ipivk : An - ipivk;
        *perm_mt = ipiv->max_mt - ipivk;
    }
    else {
        *perm_m  = -1;
        *perm_mt = -1;
    }
}

static inline int
CORE_zlaswp_get_parsec( parsec_execution_stream_t *context,
                        parsec_task_t             *this_task )
{
    cham_side_t            side;
    int                    m0, m, n, k, lda, ldb, perm_m, perm_mt, *perm;
    CHAMELEON_Complex64_t *A, *B;

    parsec_dtd_unpack_args( this_task, &side, &m0, &m, &n, &k, &perm_m, &perm_mt,
                            &perm, &A, &lda, &B, &ldb );

    if ( perm_mt < 0 ) {
        CORE_zlaswp_get( side, m0, m, n, k, A, lda, B, ldb, perm );
    }
    else {
        CORE_zlaswp_get_idx( side, m0, m, n, k, A, lda, B, ldb, perm_m, perm_mt, perm );
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Gather the rows of A(Am, An) selected by the permutation into WAP. Several
 * tiles contribute to the same WAP, which is executed on its owner.
 */
void INSERT_TASK_zlaswp_get( const RUNTIME_option_t *options,
                             cham_side_t side, cham_dir_t dir,
                             int m0, int m, int n, int k,
                             const CHAM_ipiv_t *ipiv, int ipivk,
                             const CHAM_desc_t *A,   int Am,   int An,
                             const CHAM_desc_t *WAP, int WAPm, int WAPn )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileA   = A->get_blktile( A, Am, An );
    CHAM_tile_t       *tileW   = WAP->get_blktile( WAP, WAPm, WAPn );
    chameleon_parsec_ipiv_family_t family = ( dir == ChamDirForward ) ? ChamParsecPerm : ChamParsecInvp;
    int perm_m, perm_mt;

    chameleon_parsec_laswp_perm_args( ipiv, ipivk, side, Am, An, &perm_m, &perm_mt );

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zlaswp_get_parsec, options->priority, PARSEC_DEV_CPU, "laswp_get",
        sizeof(cham_side_t), &side,    PARSEC_VALUE,
        sizeof(int),         &m0,      PARSEC_VALUE,
        sizeof(int),         &m,       PARSEC_VALUE,
        sizeof(int),         &n,       PARSEC_VALUE,
        sizeof(int),         &k,       PARSEC_VALUE,
        sizeof(int),         &perm_m,  PARSEC_VALUE,
        sizeof(int),         &perm_mt, PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipiv, family, ipivk ),
                       chameleon_parsec_ipiv_arena( ipiv, family, ipivk ) | PARSEC_INPUT,
        PASSED_BY_REF, RTBLKADDR( A, ChamComplexDouble, Am, An ),
                       chameleon_parsec_get_arena_index( A, Am, An ) | PARSEC_INPUT,
        sizeof(int),         &(tileA->ld), PARSEC_VALUE,
        PASSED_BY_REF, RTBLKADDR( WAP, ChamComplexDouble, WAPm, WAPn ),
                       chameleon_parsec_get_arena_index( WAP, WAPm, WAPn ) | PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),         &(tileW->ld), PARSEC_VALUE,
        PARSEC_DTD_ARG_END );
}

static inline int
CORE_zlaswp_set_parsec( parsec_execution_stream_t *context,
                        parsec_task_t             *this_task )
{
    cham_side_t            side;
    int                    m0, m, n, k, lda, ldb, perm_m, perm_mt, *invp;
    CHAMELEON_Complex64_t *A, *B;

    parsec_dtd_unpack_args( this_task, &side, &m0, &m, &n, &k, &perm_m, &perm_mt,
                            &invp, &A, &lda, &B, &ldb );

    if ( perm_mt < 0 ) {
        CORE_zlaswp_set( side, m0, m, n, k, A, lda, B, ldb, invp );
    }
    else {
        CORE_zlaswp_set_idx( side, m0, m, n, k, A, lda, B, ldb, perm_m, perm_mt, invp );
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Scatter the rows of WA into A(Am, An) following the inverse permutation.
 */
void INSERT_TASK_zlaswp_set( const RUNTIME_option_t *options,
                             cham_side_t             side,
                             cham_dir_t              dir,
                             int m0, int m, int n, int k,
                             const CHAM_ipiv_t *ipiv, int ipivk,
                             const CHAM_desc_t *WA, int WAm, int WAn,
                             const CHAM_desc_t *A,  int Am,  int An )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileWA = WA->get_blktile( WA, WAm, WAn );
    CHAM_tile_t       *tileA  = A->get_blktile(  A,  Am,  An  );
    chameleon_parsec_ipiv_family_t family = ( dir == ChamDirForward ) ? ChamParsecInvp : ChamParsecPerm;
    int perm_m, perm_mt;

    chameleon_parsec_laswp_perm_args( ipiv, ipivk, side, Am, An, &perm_m, &perm_mt );

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zlaswp_set_parsec, options->priority, PARSEC_DEV_CPU, "laswp_set",
        sizeof(cham_side_t), &side,    PARSEC_VALUE,
        sizeof(int),         &m0,      PARSEC_VALUE,
        sizeof(int),         &m,       PARSEC_VALUE,
        sizeof(int),         &n,       PARSEC_VALUE,
        sizeof(int),         &k,       PARSEC_VALUE,
        sizeof(int),         &perm_m,  PARSEC_VALUE,
        sizeof(int),         &perm_mt, PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipiv, family, ipivk ),
                       chameleon_parsec_ipiv_arena( ipiv, family, ipivk ) | PARSEC_INPUT,
        PASSED_BY_REF, RTBLKADDR( WA, ChamComplexDouble, WAm, WAn ),
                       chameleon_parsec_get_arena_index( WA, WAm, WAn ) | PARSEC_INPUT,
        sizeof(int),         &(tileWA->ld), PARSEC_VALUE,
        PASSED_BY_REF, RTBLKADDR( A, ChamComplexDouble, Am, An ),
                       chameleon_parsec_get_arena_index( A, Am, An ) | PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),         &(tileA->ld), PARSEC_VALUE,
        PARSEC_DTD_ARG_END );
}

#if defined(CHAMELEON_USE_MPI)

static inline int
CORE_zlaswp_ret_parsec( parsec_execution_stream_t *context,
                        parsec_task_t             *this_task )
{
    CHAMELEON_Complex64_t *A, *rows;
    void                  *wsbuf;
    CHAM_laswpws_t         lws;
    chameleon_parsec_perm_hdr_t *hdr;
    int i, lda, ldb, A_inc;

    parsec_dtd_unpack_args( this_task, &A, &lda, &wsbuf );

    hdr = (chameleon_parsec_perm_hdr_t *)wsbuf;
    chameleon_parsec_perm_load( wsbuf, &lws );
    rows  = (CHAMELEON_Complex64_t *)(lws.rows);
    ldb   = hdr->n;
    A_inc = ( hdr->side == ChamLeft ) ? 1   : lda;
    lda   = ( hdr->side == ChamLeft ) ? lda : 1;

    for ( i = 0; i < hdr->m; i++ ) {
        if ( lws.index[i] != -1 ) {
            cblas_zcopy( hdr->n, rows + lws.index[i] * ldb, 1,
                                 A    + i * A_inc,          lda );
        }
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Copy back the rows gathered in the permutation workspace into A.
 */
void INSERT_TASK_zlaswp_ret( const RUNTIME_option_t *options,
                             CHAM_perm_t       *ws, int Wm, int Wn,
                             const CHAM_desc_t *A,  int Am, int An )
{
    parsec_taskpool_t *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    CHAM_tile_t       *tileA = A->get_blktile( A, Am, An );

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zlaswp_ret_parsec, options->priority, PARSEC_DEV_CPU, "laswp_ret",
        PASSED_BY_REF, RTBLKADDR( A, ChamComplexDouble, Am, An ),
                       chameleon_parsec_get_arena_index( A, Am, An ) | PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),   &(tileA->ld), PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_perm_tile( options, ws, Wm, Wn ),
                       chameleon_parsec_perm_arena( ws ) | PARSEC_INPUT,
        PARSEC_DTD_ARG_END );
}

#endif /* if defined(CHAMELEON_USE_MPI) */
