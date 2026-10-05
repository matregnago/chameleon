/**
 *
 * @file parsec/runtime_perm.c
 *
 * @copyright 2025-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC panel permutation update routines. These routines are used by
 * laswp/lapmt operations.
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2026-10-05
 *
 */
#include "chameleon_parsec.h"

/*
 * Same keys as StarPU: (m + n * NP) on the left side, (n + m * NP) on the
 * right side, and the rank index (m on the left, n on the right) is the owner.
 */
static int
chameleon_parsec_perm_owner( const chameleon_parsec_vdc_t *vdc, int key )
{
    const CHAM_perm_t *ws = (const CHAM_perm_t *)(vdc->args);
    return key % ws->NP;
}

static void
chameleon_parsec_perm_init( const chameleon_parsec_vdc_t *vdc, int key, void *buf )
{
    const CHAM_perm_t           *ws  = (const CHAM_perm_t *)(vdc->args);
    chameleon_parsec_perm_hdr_t *hdr = (chameleon_parsec_perm_hdr_t *)buf;
    int idx = key / ws->NP;
    int i;

    hdr->nindex = 0;
    hdr->side   = ws->side;
    if ( ws->side == ChamLeft ) {
        hdr->m = ws->mb;
        hdr->n = ( idx == ws->nt - 1 ) ? ws->n - idx * ws->nb : ws->nb;
    }
    else {
        hdr->m = ws->nb;
        hdr->n = ( idx == ws->mt - 1 ) ? ws->m - idx * ws->mb : ws->mb;
    }
    for ( i = 0; i < hdr->m; i++ ) {
        ((int *)((char *)buf + CHAMELEON_PARSEC_PERM_HDR))[i] = -1;
    }
}

void
RUNTIME_perm_create( CHAM_perm_t *ws )
{
    chameleon_parsec_vdc_t *vdc = calloc( 1, sizeof(chameleon_parsec_vdc_t) );
    int    nkeys = ( ws->side == ChamLeft ) ? ws->nt * ws->NP : ws->mt * ws->NP;
    int    mrows = ( ws->side == ChamLeft ) ? ws->mb : ws->nb;
    int    ncols = ( ws->side == ChamLeft ) ? ws->nb : ws->mb;
    size_t size  = chameleon_parsec_perm_rowsoff( mrows )
        + (size_t)mrows * ncols * CHAMELEON_Element_Size( ws->dtyp );

    chameleon_parsec_vdc_init( vdc, nkeys, size, size,
                               chameleon_parsec_perm_owner, NULL,
                               chameleon_parsec_perm_init, ws );
    ws->ws = vdc;
}

void *
RUNTIME_perm_getaddr( const CHAM_perm_t *ws,
                      int                m,
                      int                n )
{
    return parsec_dtd_tile_of( (parsec_data_collection_t *)(ws->ws),
                               chameleon_parsec_perm_key( ws, m, n ) );
}

void
RUNTIME_perm_destroy( CHAM_perm_t *ws )
{
    chameleon_parsec_vdc_t *vdc = (chameleon_parsec_vdc_t *)(ws->ws);

    if ( vdc == NULL ) {
        return;
    }
    chameleon_parsec_vdc_fini( vdc );
    free( vdc );
    ws->ws = NULL;
}

/* The workspaces are flushed at the wait of the sequence */
void
RUNTIME_perm_flush( RUNTIME_sequence_t *sequence,
                    int                 rank,
                    const CHAM_perm_t  *ws,
                    int                 m,
                    int                 n )
{
    chameleon_parsec_flush_defer( (parsec_taskpool_t *)(sequence->schedopt),
                                  (parsec_data_collection_t *)(ws->ws) );
    (void)rank;
    (void)m;
    (void)n;
}
