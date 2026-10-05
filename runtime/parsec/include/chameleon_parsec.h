/**
 *
 * @file parsec/chameleon_parsec.h
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC runtime header
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Reazul Hoque
 * @author Florent Pruvost
 * @author Samuel Thibault
 * @date 2024-02-18
 *
 */
#ifndef _chameleon_parsec_h_
#define _chameleon_parsec_h_

#include "control/common.h"

#include <parsec.h>
#include <parsec/interfaces/dtd/insert_function.h>
#include <parsec/mca/device/device.h>

/**
 * @brief Common header of every Chameleon data collection.
 *
 * pending_tp is the taskpool in which the collection is registered for the
 * deferred flush (see chameleon_parsec_flush_defer()), or NULL.
 */
typedef struct chameleon_parsec_dc_s {
    parsec_data_collection_t super;
    parsec_taskpool_t       *pending_tp;
} chameleon_parsec_dc_t;

struct chameleon_parsec_desc_s {
    parsec_data_collection_t super;
    parsec_taskpool_t       *pending_tp;
    int                      arena_ids[2][2]; /**< Arenas of the tiles: [last tile row?][last tile col?] */
    CHAM_desc_t             *desc;
    parsec_data_t          **data_map;
};

typedef struct chameleon_parsec_desc_s chameleon_parsec_desc_t;

/**
 * @brief Generic data collection of 1D buffers (ipiv, pivot, permutation
 * workspaces), one buffer per key.
 */
typedef struct chameleon_parsec_vdc_s chameleon_parsec_vdc_t;

typedef int   (*chameleon_parsec_vdc_owner_fct_t)  ( const chameleon_parsec_vdc_t *vdc, int key );
typedef void *(*chameleon_parsec_vdc_userptr_fct_t)( const chameleon_parsec_vdc_t *vdc, int key );
typedef void  (*chameleon_parsec_vdc_init_fct_t)   ( const chameleon_parsec_vdc_t *vdc, int key, void *buf );

struct chameleon_parsec_vdc_s {
    parsec_data_collection_t           super;
    parsec_taskpool_t                 *pending_tp;
    int                                nkeys;     /**< Number of buffers                        */
    size_t                             size;      /**< Size in bytes of each buffer             */
    size_t                             size_last; /**< Size in bytes of the last buffer         */
    int                                arena_id;      /**< Arena of the regular buffers         */
    int                                arena_id_last; /**< Arena of the last buffer             */
    chameleon_parsec_vdc_owner_fct_t   owner;     /**< Rank owning a key                        */
    chameleon_parsec_vdc_userptr_fct_t userptr;   /**< User buffer of a key, or NULL (internal) */
    chameleon_parsec_vdc_init_fct_t    init;      /**< Initialization of internal buffers       */
    void                              *args;      /**< Opaque argument of the callbacks         */
    void                             **ptrs;      /**< Internally allocated buffers             */
    parsec_data_t                    **data_map;
};

/*
 * Arena datatypes cache (runtime_auxdc.c)
 */
int  chameleon_parsec_arena_typed( cham_flttype_t dtyp, int m, int n, int ld );
int  chameleon_parsec_arena_bytes( size_t nbytes );
void chameleon_parsec_arena_fini( parsec_context_t *parsec );

/*
 * Deferred flush (runtime_auxdc.c)
 */
void chameleon_parsec_flush_defer( parsec_taskpool_t *tp, parsec_data_collection_t *dc );
void chameleon_parsec_flush_pending( parsec_taskpool_t *tp );
int  chameleon_parsec_flush_has_pending( const parsec_taskpool_t *tp );
void chameleon_parsec_flush_forget( parsec_data_collection_t *dc );

/*
 * Generic vector data collection (runtime_auxdc.c)
 */
void chameleon_parsec_vdc_init( chameleon_parsec_vdc_t *vdc, int nkeys,
                                size_t size, size_t size_last,
                                chameleon_parsec_vdc_owner_fct_t   owner,
                                chameleon_parsec_vdc_userptr_fct_t userptr,
                                chameleon_parsec_vdc_init_fct_t    init,
                                void *args );
void chameleon_parsec_vdc_fini( chameleon_parsec_vdc_t *vdc );

static inline int
chameleon_parsec_vdc_arena( const chameleon_parsec_vdc_t *vdc, int key ) {
    return ( key == vdc->nkeys-1 ) ? vdc->arena_id_last : vdc->arena_id;
}

static inline parsec_dtd_tile_t *
chameleon_parsec_vdc_tile_of( const RUNTIME_option_t *options, const chameleon_parsec_vdc_t *vdc, int key ) {
    parsec_data_collection_t *dc = (parsec_data_collection_t *)vdc;
    assert( (key >= 0) && (key < vdc->nkeys) );
    chameleon_parsec_flush_defer( (parsec_taskpool_t *)(options->sequence->schedopt), dc );
    return parsec_dtd_tile_of( dc, key );
}

/**
 * @brief Return the arena of the tile (m, n) of desc.
 *
 * Border tiles may be smaller and, in tile storage, have a smaller leading
 * dimension, so the datatype depends on the position of the tile in the
 * global matrix.
 */
static inline int
chameleon_parsec_get_arena_index( const CHAM_desc_t *desc, int m, int n ) {
    const chameleon_parsec_desc_t *pdesc = (const chameleon_parsec_desc_t *)(desc->schedopt);
    const CHAM_desc_t             *mdesc = pdesc->desc;
    int mm = m + desc->i / desc->mb;
    int nn = n + desc->j / desc->nb;
    return pdesc->arena_ids[ mm == (mdesc->lmt-1) ][ nn == (mdesc->lnt-1) ];
}

/**
 * @brief Return the DTD tile of the tile (m, n) of desc, and register the
 * collection for the deferred flush of the sequence.
 *
 * The key is computed from desc itself and not from the runtime descriptor,
 * which is shared with the parent matrix of a submatrix and thus does not
 * know the (i, j) offset.
 */
static inline parsec_dtd_tile_t *
chameleon_parsec_tile_of( const RUNTIME_option_t *options, const CHAM_desc_t *desc, int m, int n ) {
    parsec_data_collection_t *dc    = (parsec_data_collection_t *)(desc->schedopt);
    const CHAM_desc_t        *mdesc = ((chameleon_parsec_desc_t *)dc)->desc;
    parsec_data_key_t         key;

    key = (parsec_data_key_t)(n + desc->j / desc->nb) * mdesc->lmt + (m + desc->i / desc->mb);
    chameleon_parsec_flush_defer( (parsec_taskpool_t *)(options->sequence->schedopt), dc );
    return parsec_dtd_tile_of( dc, key );
}

static inline int
chameleon_parsec_get_arena_index_ipiv( const CHAM_ipiv_t *ipiv ) {
    assert(0);
    return -1;
}

static inline int
chameleon_parsec_get_arena_index_perm( const CHAM_ipiv_t *ipiv ) {
    assert(0);
    return -1;
}

static inline int
chameleon_parsec_get_arena_index_invp( const CHAM_ipiv_t *ipiv ) {
    assert(0);
    return -1;
}

static inline int cham_to_parsec_access( cham_access_t accessA ) {
    if ( accessA == ChamR ) {
        return PARSEC_INPUT;
    }
    if ( accessA == ChamW ) {
        return PARSEC_OUTPUT;
    }
    return PARSEC_INOUT;
}

/*
 * Access to block pointer and leading dimension
 */
#define RTBLKADDR( desc, type, m, n ) chameleon_parsec_tile_of( options, (desc), (m), (n) )

#define RUNTIME_BEGIN_ACCESS_DECLARATION

#define RUNTIME_ACCESS_R(A, Am, An)

#define RUNTIME_ACCESS_W(A, Am, An)

#define RUNTIME_ACCESS_RW(A, Am, An)

#define RUNTIME_RANK_CHANGED(rank)

#define RUNTIME_END_ACCESS_DECLARATION

#endif /* _chameleon_parsec_h_ */
