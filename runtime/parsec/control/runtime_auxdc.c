/**
 *
 * @file parsec/runtime_auxdc.c
 *
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC auxiliary data management: arena datatypes cache,
 * deferred flush of the data collections, and generic vector data collection.
 *
 * @version 1.4.0
 * @date 2026-10-05
 *
 */
#include "chameleon_parsec.h"
#include <parsec/data.h>
#include <parsec/datatype.h>
#include <parsec/arena.h>
#include <parsec/data_dist/matrix/matrix.h>
#include <pthread.h>

/*
 * Arena datatypes cache
 * ---------------------
 *
 * The DTD arena identifiers are given by a per-context counter, so they must be
 * created in the same order on all the ranks: arenas are created eagerly in
 * collective calls (descriptor/ipiv creation) and cached to be shared among
 * the data collections with the same shape. The cache lives as long as the
 * PaRSEC context (one Chameleon context at a time).
 */
typedef struct chameleon_parsec_arena_s {
    int    kind; /**< 0: typed rectangle, 1: bytes */
    int    dtyp;
    int    m, n, ld;
    size_t nbytes;
    int    id;
} chameleon_parsec_arena_t;

static chameleon_parsec_arena_t *chameleon_parsec_arenas      = NULL;
static int                       chameleon_parsec_arenas_nb   = 0;
static int                       chameleon_parsec_arenas_size = 0;
static pthread_mutex_t           chameleon_parsec_arenas_lock = PTHREAD_MUTEX_INITIALIZER;

static parsec_context_t *
chameleon_parsec_context( void )
{
    CHAM_context_t *chamctxt = chameleon_context_self();
    assert( chamctxt != NULL );
    return (parsec_context_t *)(chamctxt->schedopt);
}

static int
chameleon_parsec_arena_lookup_or_create( const chameleon_parsec_arena_t *key,
                                         parsec_datatype_t oldtype,
                                         int m, int n, int ld )
{
    chameleon_parsec_arena_t *arena;
    parsec_arena_datatype_t  *adt;
    int i, id = -1;

    pthread_mutex_lock( &chameleon_parsec_arenas_lock );
    for ( i=0, arena=chameleon_parsec_arenas; i<chameleon_parsec_arenas_nb; i++, arena++ ) {
        if ( (arena->kind == key->kind) && (arena->dtyp == key->dtyp) &&
             (arena->m    == key->m   ) && (arena->n    == key->n   ) &&
             (arena->ld   == key->ld  ) && (arena->nbytes == key->nbytes) )
        {
            id = arena->id;
            break;
        }
    }

    if ( id == -1 ) {
        adt = parsec_matrix_adt_new_rect( oldtype, m, n, ld );
        if ( (adt == NULL) ||
             (parsec_dtd_attach_arena_datatype( chameleon_parsec_context(), adt, &id ) != PARSEC_SUCCESS) )
        {
            chameleon_fatal_error( "chameleon_parsec_arena", "Failed to create a PaRSEC arena datatype" );
        }

        if ( chameleon_parsec_arenas_nb == chameleon_parsec_arenas_size ) {
            chameleon_parsec_arenas_size = (chameleon_parsec_arenas_size == 0) ? 16 : 2 * chameleon_parsec_arenas_size;
            chameleon_parsec_arenas = realloc( chameleon_parsec_arenas,
                                               chameleon_parsec_arenas_size * sizeof(chameleon_parsec_arena_t) );
        }
        arena     = chameleon_parsec_arenas + chameleon_parsec_arenas_nb;
        *arena    = *key;
        arena->id = id;
        chameleon_parsec_arenas_nb++;
    }
    pthread_mutex_unlock( &chameleon_parsec_arenas_lock );

    assert( (id & PARSEC_GET_REGION_INFO) == id );
    return id;
}

/**
 * @brief Return the arena id of a m-by-n tile with leading dimension ld of the
 * given arithmetic.
 */
int
chameleon_parsec_arena_typed( cham_flttype_t dtyp, int m, int n, int ld )
{
    chameleon_parsec_arena_t key = { 0, (int)dtyp, m, n, ld, 0, -1 };
    parsec_datatype_t        oldtype;
    int                      eltsize = 1;

    switch( dtyp ) {
    case ChamInteger:       oldtype = parsec_datatype_int32_t;          break;
    case ChamInteger64:     oldtype = parsec_datatype_int64_t;          break;
    case ChamRealFloat:     oldtype = parsec_datatype_float_t;          break;
    case ChamRealDouble:    oldtype = parsec_datatype_double_t;         break;
    case ChamComplexFloat:  oldtype = parsec_datatype_complex_t;        break;
    case ChamComplexDouble: oldtype = parsec_datatype_double_complex_t; break;
    default:
        /* Other arithmetics (half, bytes) are exchanged as raw bytes */
        oldtype = parsec_datatype_uint8_t;
        eltsize = CHAMELEON_Element_Size( dtyp );
    }

    /* Empty tiles still need a valid datatype */
    m  = chameleon_max( m,  1 );
    n  = chameleon_max( n,  1 );
    ld = chameleon_max( ld, m );

    return chameleon_parsec_arena_lookup_or_create( &key, oldtype,
                                                    m * eltsize, n, ld * eltsize );
}

/**
 * @brief Return the arena id of a contiguous buffer of nbytes bytes.
 */
int
chameleon_parsec_arena_bytes( size_t nbytes )
{
    chameleon_parsec_arena_t key = { 1, 0, 0, 0, 0, nbytes, -1 };

    nbytes = ( nbytes > 0 ) ? nbytes : 1;
    return chameleon_parsec_arena_lookup_or_create( &key, parsec_datatype_uint8_t,
                                                    nbytes, 1, nbytes );
}

/**
 * @brief Release all the cached arenas. Must be called before parsec_fini().
 */
void
chameleon_parsec_arena_fini( parsec_context_t *parsec )
{
    int i;

    pthread_mutex_lock( &chameleon_parsec_arenas_lock );
    for ( i=0; i<chameleon_parsec_arenas_nb; i++ ) {
        parsec_dtd_free_arena_datatype( parsec, chameleon_parsec_arenas[i].id );
    }
    free( chameleon_parsec_arenas );
    chameleon_parsec_arenas      = NULL;
    chameleon_parsec_arenas_nb   = 0;
    chameleon_parsec_arenas_size = 0;
    pthread_mutex_unlock( &chameleon_parsec_arenas_lock );
}

/*
 * Deferred flush
 * --------------
 *
 * A DTD tile that is not flushed survives in the data collection hash table
 * with last users pointing to tasks of a taskpool that may already be freed,
 * and a flushed tile can only be reused after a wait on the taskpool. Thus,
 * every collection touched by a taskpool is recorded here, in order of first
 * touch (which is the same on all ranks as they insert the same tasks), and
 * all of them are flushed right before waiting for the taskpool.
 */
typedef struct chameleon_parsec_flush_s {
    struct chameleon_parsec_flush_s *next;
    parsec_taskpool_t               *tp;
    parsec_data_collection_t       **dcs;
    int                              nb;
    int                              size;
} chameleon_parsec_flush_t;

static chameleon_parsec_flush_t *chameleon_parsec_flush_list = NULL;
static pthread_mutex_t           chameleon_parsec_flush_lock = PTHREAD_MUTEX_INITIALIZER;

void
chameleon_parsec_flush_defer( parsec_taskpool_t *tp, parsec_data_collection_t *dc )
{
    chameleon_parsec_dc_t    *cdc = (chameleon_parsec_dc_t *)dc;
    chameleon_parsec_flush_t *entry;
    int i;

    /* Fast path: already registered in this taskpool */
    if ( cdc->pending_tp == tp ) {
        return;
    }

    pthread_mutex_lock( &chameleon_parsec_flush_lock );
    for ( entry = chameleon_parsec_flush_list; entry != NULL; entry = entry->next ) {
        if ( entry->tp == tp ) {
            break;
        }
    }
    if ( entry == NULL ) {
        entry = calloc( 1, sizeof(chameleon_parsec_flush_t) );
        entry->tp   = tp;
        entry->next = chameleon_parsec_flush_list;
        chameleon_parsec_flush_list = entry;
    }

    /* The collection may have been registered and then used in another taskpool */
    for ( i=0; i<entry->nb; i++ ) {
        if ( entry->dcs[i] == dc ) {
            break;
        }
    }
    if ( i == entry->nb ) {
        if ( entry->nb == entry->size ) {
            entry->size = (entry->size == 0) ? 16 : 2 * entry->size;
            entry->dcs  = realloc( entry->dcs, entry->size * sizeof(parsec_data_collection_t *) );
        }
        entry->dcs[entry->nb] = dc;
        entry->nb++;
    }
    cdc->pending_tp = tp;
    pthread_mutex_unlock( &chameleon_parsec_flush_lock );
}

static chameleon_parsec_flush_t *
chameleon_parsec_flush_pop( const parsec_taskpool_t *tp )
{
    chameleon_parsec_flush_t **prev, *entry;

    for ( prev = &chameleon_parsec_flush_list; *prev != NULL; prev = &((*prev)->next) ) {
        entry = *prev;
        if ( entry->tp == tp ) {
            *prev = entry->next;
            return entry;
        }
    }
    return NULL;
}

/**
 * @brief Insert the flush tasks of all the collections touched by tp, in order
 * of first touch. The caller must wait on tp before reusing the data.
 */
void
chameleon_parsec_flush_pending( parsec_taskpool_t *tp )
{
    chameleon_parsec_flush_t *entry;
    int i;

    pthread_mutex_lock( &chameleon_parsec_flush_lock );
    entry = chameleon_parsec_flush_pop( tp );
    if ( entry != NULL ) {
        for ( i=0; i<entry->nb; i++ ) {
            chameleon_parsec_dc_t *cdc = (chameleon_parsec_dc_t *)(entry->dcs[i]);
            if ( cdc->pending_tp == tp ) {
                cdc->pending_tp = NULL;
            }
        }
    }
    pthread_mutex_unlock( &chameleon_parsec_flush_lock );

    if ( entry == NULL ) {
        return;
    }

    for ( i=0; i<entry->nb; i++ ) {
        parsec_dtd_data_flush_all( tp, entry->dcs[i] );
    }
    free( entry->dcs );
    free( entry );
}

int
chameleon_parsec_flush_has_pending( const parsec_taskpool_t *tp )
{
    chameleon_parsec_flush_t *entry;

    pthread_mutex_lock( &chameleon_parsec_flush_lock );
    for ( entry = chameleon_parsec_flush_list; entry != NULL; entry = entry->next ) {
        if ( entry->tp == tp ) {
            break;
        }
    }
    pthread_mutex_unlock( &chameleon_parsec_flush_lock );
    return entry != NULL;
}

/**
 * @brief Remove a collection that is about to be destroyed from all the
 * pending flush lists.
 */
void
chameleon_parsec_flush_forget( parsec_data_collection_t *dc )
{
    chameleon_parsec_dc_t    *cdc = (chameleon_parsec_dc_t *)dc;
    chameleon_parsec_flush_t *entry;
    int i, found = 0;

    pthread_mutex_lock( &chameleon_parsec_flush_lock );
    for ( entry = chameleon_parsec_flush_list; entry != NULL; entry = entry->next ) {
        for ( i=0; i<entry->nb; i++ ) {
            if ( entry->dcs[i] == dc ) {
                memmove( entry->dcs + i, entry->dcs + i + 1,
                         (entry->nb - i - 1) * sizeof(parsec_data_collection_t *) );
                entry->nb--;
                found = 1;
                break;
            }
        }
    }
    cdc->pending_tp = NULL;
    pthread_mutex_unlock( &chameleon_parsec_flush_lock );

    if ( found ) {
        chameleon_warning( "chameleon_parsec_flush_forget",
                           "A data collection is destroyed before the wait of the sequence using it" );
    }
}

/*
 * Generic vector data collection
 * ------------------------------
 */
static parsec_data_key_t
chameleon_parsec_vdc_data_key( parsec_data_collection_t *dc, ... )
{
    va_list ap;
    int key;

    va_start( ap, dc );
    key = va_arg( ap, int );
    va_end( ap );

    (void)dc;
    return key;
}

static uint32_t
chameleon_parsec_vdc_rank_of_key( parsec_data_collection_t *dc, parsec_data_key_t key )
{
    chameleon_parsec_vdc_t *vdc = (chameleon_parsec_vdc_t *)dc;
    assert( (int)key < vdc->nkeys );
    return vdc->owner( vdc, key );
}

static uint32_t
chameleon_parsec_vdc_rank_of( parsec_data_collection_t *dc, ... )
{
    va_list ap;
    int key;

    va_start( ap, dc );
    key = va_arg( ap, int );
    va_end( ap );

    return chameleon_parsec_vdc_rank_of_key( dc, key );
}

static int32_t
chameleon_parsec_vdc_vpid_of_key( parsec_data_collection_t *dc, parsec_data_key_t key )
{
    (void)dc;
    (void)key;
    return 0;
}

static int32_t
chameleon_parsec_vdc_vpid_of( parsec_data_collection_t *dc, ... )
{
    (void)dc;
    return 0;
}

static parsec_data_t *
chameleon_parsec_vdc_data_of_key( parsec_data_collection_t *dc, parsec_data_key_t key )
{
    chameleon_parsec_vdc_t *vdc = (chameleon_parsec_vdc_t *)dc;
    size_t size;
    void  *buf;

    assert( (int)key < vdc->nkeys );
    if ( vdc->data_map[key] != NULL ) {
        return vdc->data_map[key];
    }

    size = ( (int)key == vdc->nkeys-1 ) ? vdc->size_last : vdc->size;
    buf  = ( vdc->userptr != NULL ) ? vdc->userptr( vdc, key ) : NULL;
    if ( buf == NULL ) {
        buf = parsec_data_allocate( ( size > 0 ) ? size : 1 );
        vdc->ptrs[key] = buf;
        if ( vdc->init != NULL ) {
            vdc->init( vdc, key, buf );
        }
    }

    return parsec_data_create( vdc->data_map + key, dc, key, buf, size,
                               PARSEC_DATA_FLAG_PARSEC_MANAGED );
}

static parsec_data_t *
chameleon_parsec_vdc_data_of( parsec_data_collection_t *dc, ... )
{
    va_list ap;
    int key;

    va_start( ap, dc );
    key = va_arg( ap, int );
    va_end( ap );

    return chameleon_parsec_vdc_data_of_key( dc, key );
}

/**
 * @brief Initialize a vector data collection. Collective call: all the ranks
 * must create the same collections in the same order.
 *
 * @param[in] size
 *          Size in bytes of the buffers 0 to nkeys-2.
 *
 * @param[in] size_last
 *          Size in bytes of the buffer nkeys-1.
 *
 * @param[in] owner
 *          Returns the rank owning a key.
 *
 * @param[in] userptr
 *          If not NULL, returns the user buffer associated to a key, or NULL
 *          if the buffer must be allocated internally.
 *
 * @param[in] init
 *          If not NULL, initializes the internally allocated buffers.
 */
void
chameleon_parsec_vdc_init( chameleon_parsec_vdc_t *vdc, int nkeys,
                           size_t size, size_t size_last,
                           chameleon_parsec_vdc_owner_fct_t   owner,
                           chameleon_parsec_vdc_userptr_fct_t userptr,
                           chameleon_parsec_vdc_init_fct_t    init,
                           void *args )
{
    CHAM_context_t           *chamctxt = chameleon_context_self();
    parsec_data_collection_t *dc       = (parsec_data_collection_t *)vdc;

    memset( vdc, 0, sizeof(chameleon_parsec_vdc_t) );
    parsec_data_collection_init( dc, RUNTIME_comm_size( chamctxt ), RUNTIME_comm_rank( chamctxt ) );

    dc->data_key    = chameleon_parsec_vdc_data_key;
    dc->rank_of     = chameleon_parsec_vdc_rank_of;
    dc->rank_of_key = chameleon_parsec_vdc_rank_of_key;
    dc->data_of     = chameleon_parsec_vdc_data_of;
    dc->data_of_key = chameleon_parsec_vdc_data_of_key;
    dc->vpid_of     = chameleon_parsec_vdc_vpid_of;
    dc->vpid_of_key = chameleon_parsec_vdc_vpid_of_key;

    vdc->nkeys     = nkeys;
    vdc->size      = size;
    vdc->size_last = size_last;
    vdc->owner     = owner;
    vdc->userptr   = userptr;
    vdc->init      = init;
    vdc->args      = args;
    vdc->ptrs      = calloc( chameleon_max( nkeys, 1 ), sizeof(void *) );
    vdc->data_map  = calloc( chameleon_max( nkeys, 1 ), sizeof(parsec_data_t *) );

    vdc->arena_id      = chameleon_parsec_arena_bytes( size );
    vdc->arena_id_last = chameleon_parsec_arena_bytes( size_last );

    parsec_dtd_data_collection_init( dc );
}

void
chameleon_parsec_vdc_fini( chameleon_parsec_vdc_t *vdc )
{
    parsec_data_collection_t *dc = (parsec_data_collection_t *)vdc;
    int key;

    if ( vdc->pending_tp != NULL ) {
        chameleon_parsec_flush_forget( dc );
    }

    parsec_dtd_data_collection_fini( dc );

    for ( key=0; key<vdc->nkeys; key++ ) {
        if ( vdc->data_map[key] != NULL ) {
            parsec_data_destroy( vdc->data_map[key] );
        }
        if ( vdc->ptrs[key] != NULL ) {
            parsec_data_free( vdc->ptrs[key] );
        }
    }
    free( vdc->data_map );
    free( vdc->ptrs );
    vdc->data_map = NULL;
    vdc->ptrs     = NULL;

    parsec_data_collection_destroy( dc );
}
