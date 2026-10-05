/**
 *
 * @file parsec/runtime_descriptor.c
 *
 * @copyright 2012-2017 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC descriptor routines
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @author Guillaume Sylvand
 * @author Samuel Thibault
 * @author Florent Pruvost
 * @author Brieuc Nicolas
 * @date 2025-12-19
 *
 */
#include "chameleon_parsec.h"
#include <parsec/data.h>
#include <parsec/datatype.h>
#include <parsec/arena.h>

void *RUNTIME_malloc( size_t size )
{
    return malloc(size);
}

void RUNTIME_free( void *ptr, size_t size )
{
    (void)size;
    free(ptr);
    return;
}

static inline void
chameleon_parsec_key_to_coordinates(parsec_data_collection_t *data_collection, parsec_data_key_t key,
                                int *m, int *n)
{
    chameleon_parsec_desc_t *pdesc = (chameleon_parsec_desc_t*)data_collection;
    CHAM_desc_t *mdesc = pdesc->desc;
    int _m, _n;

    _m = key % mdesc->lmt;
    _n = key / mdesc->lmt;
    *m = _m - mdesc->i / mdesc->mb;
    *n = _n - mdesc->j / mdesc->nb;
}

static inline parsec_data_key_t
chameleon_parsec_data_key(parsec_data_collection_t *data_collection, ...)
{
    chameleon_parsec_desc_t *pdesc = (chameleon_parsec_desc_t*)data_collection;
    CHAM_desc_t *mdesc = pdesc->desc;
    va_list ap;
    int m, n;

    /* Get coordinates */
    va_start(ap, data_collection);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += mdesc->i / mdesc->mb;
    n += mdesc->j / mdesc->nb;

    return ((n * mdesc->lmt) + m);
}

/*
 * The key is the index of the tile in the global tile array of the
 * descriptor, so the rank is directly given by the tile structure. This also
 * supports any custom data distribution.
 */
static inline uint32_t
chameleon_parsec_rank_of_key(parsec_data_collection_t *data_collection, parsec_data_key_t key)
{
    chameleon_parsec_desc_t *pdesc = (chameleon_parsec_desc_t*)data_collection;
    CHAM_desc_t *mdesc = pdesc->desc;

    assert( key < (parsec_data_key_t)(mdesc->lmt * mdesc->lnt) );
    return mdesc->tiles[key].rank;
}

static inline uint32_t
chameleon_parsec_rank_of(parsec_data_collection_t *data_collection, ...)
{
    va_list ap;
    int m, n;

    /* Get coordinates */
    va_start(ap, data_collection);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    return chameleon_parsec_rank_of_key( data_collection,
                                         chameleon_parsec_data_key( data_collection, m, n ) );
}

static inline int32_t
chameleon_parsec_vpid_of(parsec_data_collection_t *data_collection, ... )
{
    (void)data_collection;
    return 0;
}

static inline int32_t
chameleon_parsec_vpid_of_key(parsec_data_collection_t *data_collection, parsec_data_key_t key)
{
    (void)data_collection;
    (void)key;
    return 0;
}

/*
 * Only called for the local tiles. Tiles without memory (allocation tile per
 * tile) are allocated on first use and released with the descriptor.
 */
static inline parsec_data_t*
chameleon_parsec_data_of_key(parsec_data_collection_t *data_collection, parsec_data_key_t key)
{
    chameleon_parsec_desc_t *pdesc = (chameleon_parsec_desc_t*)data_collection;
    CHAM_desc_t *mdesc = pdesc->desc;
    CHAM_tile_t *tile;
    size_t       eltsize, size;

    assert( key < (parsec_data_key_t)(mdesc->lmt * mdesc->lnt) );
    if ( pdesc->data_map[key] != NULL ) {
        return pdesc->data_map[key];
    }

    tile    = mdesc->tiles + key;
    eltsize = CHAMELEON_Element_Size( mdesc->dtyp );
    size    = ( (tile->m > 0) && (tile->n > 0) ) ? ((size_t)(tile->ld) * (tile->n - 1) + tile->m) * eltsize : 0;
    assert( tile->rank == (int)data_collection->myrank );

    if ( tile->mat == NULL ) {
        tile->mat = parsec_data_allocate( (size > 0) ? size : eltsize );
        pdesc->allocated[key] = 1;
    }

    return parsec_data_create( pdesc->data_map + key, data_collection, key,
                               CHAM_tile_get_ptr( tile ), size,
                               PARSEC_DATA_FLAG_PARSEC_MANAGED );
}

static inline parsec_data_t*
chameleon_parsec_data_of(parsec_data_collection_t *data_collection, ...)
{
    va_list ap;
    int m, n;

    /* Get coordinates */
    va_start(ap, data_collection);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    return chameleon_parsec_data_of_key( data_collection,
                                         chameleon_parsec_data_key( data_collection, m, n ) );
}

#if defined(PARSEC_PROF_TRACE)
static inline int
chameleon_parsec_key_to_string(parsec_data_collection_t *data_collection, parsec_data_key_t key, char * buffer, uint32_t buffer_size)
{
    int m, n, res;
    chameleon_parsec_key_to_coordinates( data_collection, key, &m, &n );
    res = snprintf( buffer, buffer_size, "(%d, %d)", m, n );
    if ( res < 0 )
    {
        printf( "error in key_to_string for tile (%u, %u) key: %u\n",
                (unsigned int)m, (unsigned int)n, key );
    }
    return res;
}
#endif

/**
 *  Create data descriptor
 *
 *  Collective call: the arenas and the data collection identifier must be
 *  created in the same order on all the ranks.
 */
void RUNTIME_desc_create( CHAM_desc_t *mdesc )
{
    CHAM_context_t           *chamctxt = chameleon_context_self();
    parsec_data_collection_t *data_collection;
    chameleon_parsec_desc_t  *pdesc;
    int                       lastm, lastn, i, j;

    pdesc = calloc( 1, sizeof(chameleon_parsec_desc_t) );
    data_collection = (parsec_data_collection_t*)pdesc;

    /* Super setup */
    parsec_data_collection_init( data_collection, RUNTIME_comm_size( chamctxt ), mdesc->myrank );

    data_collection->data_key    = chameleon_parsec_data_key;
    data_collection->rank_of     = chameleon_parsec_rank_of;
    data_collection->rank_of_key = chameleon_parsec_rank_of_key;
    data_collection->data_of     = chameleon_parsec_data_of;
    data_collection->data_of_key = chameleon_parsec_data_of_key;
    data_collection->vpid_of     = chameleon_parsec_vpid_of;
    data_collection->vpid_of_key = chameleon_parsec_vpid_of_key;
#if defined(PARSEC_PROF_TRACE)
    {
        data_collection->key_to_string = chameleon_parsec_key_to_string;
        data_collection->key           = NULL;
        chameleon_asprintf(&(data_collection->key_dim), "(%d, %d)", mdesc->lmt, mdesc->lnt);
    }
#endif

    pdesc->data_map  = calloc( mdesc->lmt * mdesc->lnt, sizeof(parsec_data_t*) );
    pdesc->allocated = calloc( mdesc->lmt * mdesc->lnt, sizeof(int8_t) );

    /* Double linking */
    pdesc->desc     = mdesc;
    mdesc->schedopt = pdesc;

    parsec_dtd_data_collection_init(data_collection);

    /*
     * Arenas: one per shape of tile. Only the last tile row and the last tile
     * column may differ from the regular mb-by-nb tiles.
     */
    lastm = chameleon_max( mdesc->lmt - 1, 0 );
    lastn = chameleon_max( mdesc->lnt - 1, 0 );
    for ( i=0; i<2; i++ ) {
        for ( j=0; j<2; j++ ) {
            int ii = i ? lastm : 0;
            int jj = j ? lastn : 0;
            int tm = ( ii == mdesc->lmt-1 ) ? mdesc->lm - ii * mdesc->mb : mdesc->mb;
            int tn = ( jj == mdesc->lnt-1 ) ? mdesc->ln - jj * mdesc->nb : mdesc->nb;
            int ld = mdesc->get_blkldd( mdesc, ii - mdesc->i / mdesc->mb );

            pdesc->arena_ids[i][j] = chameleon_parsec_arena_typed( mdesc->dtyp, tm, tn, ld );
        }
    }
    return;
}

int RUNTIME_desc_create_flatview( CHAM_desc_t *desc, const CHAM_desc_t *recdesc )
{
    (void)desc;
    (void)recdesc;
    return CHAMELEON_ERR_NOT_SUPPORTED;
}

void RUNTIME_desc_register_recursive( CHAM_desc_t *desc, int dist_level )
{
    (void)desc;
    (void)dist_level;
}

void RUNTIME_desc_destroy_submit( CHAM_desc_t *desc, const RUNTIME_sequence_t *sequence )
{
    (void)desc;
    (void)sequence;
    return;
}

/**
 *  Destroy data descriptor
 */
void RUNTIME_desc_destroy( CHAM_desc_t *mdesc )
{
    chameleon_parsec_desc_t *pdesc = (chameleon_parsec_desc_t*)(mdesc->schedopt);
    if ( pdesc == NULL ) {
        return;
    }

    /* Submatrices share the runtime descriptor of their parent */
    if ( pdesc->desc != mdesc ) {
        return;
    }

    if ( pdesc->pending_tp != NULL ) {
        chameleon_parsec_flush_forget( (parsec_data_collection_t *)pdesc );
    }

    if ( pdesc->data_map != NULL ) {
        parsec_data_t **data = pdesc->data_map;
        int nb_local_tiles = mdesc->lmt * mdesc->lnt;
        int i;

        for(i=0; i<nb_local_tiles; i++, data++)
        {
            if (*data) {
                parsec_data_destroy( *data );
            }
            if ( pdesc->allocated[i] ) {
                parsec_data_free( mdesc->tiles[i].mat );
                mdesc->tiles[i].mat = NULL;
            }
        }

        free( pdesc->data_map );
        free( pdesc->allocated );
        pdesc->data_map  = NULL;
        pdesc->allocated = NULL;
    }

    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)pdesc );
    parsec_data_collection_destroy( (parsec_data_collection_t *)pdesc );

    free(pdesc);
    mdesc->schedopt = NULL;
    return;
}

/**
 *  Acquire data
 */
int RUNTIME_desc_acquire( const CHAM_desc_t *desc )
{
    (void)desc;
    return CHAMELEON_SUCCESS;
}

/**
 *  Release data
 */
int RUNTIME_desc_release( const CHAM_desc_t *desc )
{
    (void)desc;
    return CHAMELEON_SUCCESS;
}

/**
 *  Flush cached data
 */
void RUNTIME_flush( CHAM_context_t *chamctxt )
{
    (void)chamctxt;
    return;
}

/*
 * A flushed tile cannot be reused before the wait of the taskpool, and all the
 * tiles must be flushed before this wait. The flush is thus deferred to
 * RUNTIME_sequence_wait() which flushes every collection used by the sequence.
 */
void RUNTIME_desc_flush( CHAM_desc_t              *desc,
                         const RUNTIME_sequence_t *sequence )
{
    chameleon_parsec_flush_defer( (parsec_taskpool_t *)(sequence->schedopt),
                                  (parsec_data_collection_t*)(desc->schedopt) );
}

void RUNTIME_data_flush( const RUNTIME_sequence_t *sequence,
                         const CHAM_desc_t *A, int Am, int An )
{
    /*
     * For now, we do nothing in this function as in PaRSEC, once the data is
     * flushed it cannot be reused in the same sequence, when this issue will be
     * fixed, we will uncomment this function
     */
    /* parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(sequence->schedopt); */
    /* parsec_dtd_data_flush( PARSEC_dtd_taskpool, RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ) ); */

    (void)sequence; (void)A; (void)Am; (void)An;
    return;
}

void RUNTIME_data_unregister( const RUNTIME_sequence_t *sequence,
                              const CHAM_desc_t *A, int Am, int An )
{
    (void)sequence;
    (void)A;
    (void)Am;
    (void)An;
    return;
}

#if defined(CHAMELEON_USE_MIGRATE)
void RUNTIME_data_migrate( const RUNTIME_sequence_t *sequence,
                           const CHAM_desc_t *A, int Am, int An, int new_rank )
{
    (void)sequence; (void)A; (void)Am; (void)An; (void)new_rank;
}
#endif

/**
 *  Get data addr
 */
void *RUNTIME_desc_getaddr( const CHAM_desc_t *desc, int m, int n )
{
    assert(0); /* This should not be called because we also need the handle to match the address we need. */
    return desc->get_blkaddr( desc, m, n );
}
