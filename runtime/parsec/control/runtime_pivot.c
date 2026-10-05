/**
 *
 * @file parsec/runtime_pivot.c
 *
 * @copyright 2022-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC descriptor routines
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Matthieu Kuhn
 * @author Alycia Lisito
 * @author Florent Pruvost
 * @author Matteo Marcos
 * @date 2026-10-05
 *
 */
#include "chameleon_parsec.h"

/*
 * Key 2 * rank + (h & 1) is owned by rank.
 */
static int
chameleon_parsec_pivot_owner( const chameleon_parsec_vdc_t *vdc, int key )
{
    (void)vdc;
    return key >> 1;
}

static void
chameleon_parsec_pivot_init( const chameleon_parsec_vdc_t *vdc, int key, void *buf )
{
    const CHAM_desc_pivot_t *pivot = (const CHAM_desc_pivot_t *)(vdc->args);
    chameleon_parsec_pivot_reset( buf, pivot->nb, pivot->dtyp );
    (void)key;
}

/**
 *  Create the pivot runtime structures
 */
void RUNTIME_pivot_create( CHAM_desc_pivot_t *pivot )
{
    chameleon_parsec_pivot_t *ppivot;
    size_t size;

    assert( pivot );
    ppivot = calloc( 1, sizeof(chameleon_parsec_pivot_t) );
    ppivot->NP         = RUNTIME_comm_size( chameleon_context_self() );
    ppivot->root       = -1;
    ppivot->stamp      = 0;
    ppivot->init_stamp = calloc( ppivot->NP, sizeof(int) );

    size = chameleon_parsec_pivot_size( pivot->nb, pivot->dtyp );
    chameleon_parsec_vdc_init( &(ppivot->vdc), 2 * ppivot->NP, size, size,
                               chameleon_parsec_pivot_owner, NULL,
                               chameleon_parsec_pivot_init, pivot );

    pivot->nextpiv = ppivot;
    pivot->prevpiv = ppivot;
}

/*
 * The buffers may still be used by the tasks of the sequence: they are
 * released by the synchronous destroy, after the wait.
 */
void RUNTIME_pivot_destroy_submit( RUNTIME_sequence_t *sequence,
                                   CHAM_desc_pivot_t  *pivot )
{
    RUNTIME_pivot_flushall( sequence, pivot );
}

void RUNTIME_pivot_destroy( CHAM_desc_pivot_t *pivot )
{
    chameleon_parsec_pivot_t *ppivot = chameleon_parsec_pivot( pivot );

    if ( ppivot == NULL ) {
        return;
    }
    chameleon_parsec_vdc_fini( &(ppivot->vdc) );
    free( ppivot->init_stamp );
    free( ppivot );
    pivot->nextpiv = NULL;
    pivot->prevpiv = NULL;
}

void *RUNTIME_pivot_getaddr( const CHAM_desc_pivot_t *pivot,
                             int rank, int h )
{
    return parsec_dtd_tile_of( (parsec_data_collection_t *)&(chameleon_parsec_pivot( pivot )->vdc),
                               chameleon_parsec_pivot_key( rank, h ) );
}

void RUNTIME_pivot_flushone( RUNTIME_sequence_t      *sequence,
                             const CHAM_desc_pivot_t *pivot, int rank )
{
    RUNTIME_pivot_flushall( sequence, pivot );
    (void)rank;
}

void RUNTIME_pivot_flushall( RUNTIME_sequence_t      *sequence,
                             const CHAM_desc_pivot_t *pivot )
{
    chameleon_parsec_pivot_t *ppivot = chameleon_parsec_pivot( pivot );

    if ( ppivot == NULL ) {
        return;
    }
    chameleon_parsec_flush_defer( (parsec_taskpool_t *)(sequence->schedopt),
                                  (parsec_data_collection_t *)&(ppivot->vdc) );
}

/* The buffers are explicitly reset by the first task of each step */
void RUNTIME_pivot_invalidate( const CHAM_desc_pivot_t *pivot,
                               int                      rank,
                               int                      h )
{
    (void)pivot;
    (void)h;
    (void)rank;
}
