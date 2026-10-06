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
#include <parsec/interfaces/dtd/insert_function_internal.h>
#include <parsec/mca/device/device.h>

/*
 * The CUDA incarnations of the codelets require both Chameleon and PaRSEC to be
 * built with CUDA.
 */
#if defined(CHAMELEON_USE_CUDA) && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) && !defined(CHAMELEON_SIMULATION)
#define CHAMELEON_PARSEC_CUDA
#include <parsec/parsec_internal.h>
#include <parsec/mca/device/device_gpu.h>
#include <parsec/mca/device/cuda/device_cuda.h>
#endif

#if defined(CHAMELEON_PARSEC_CUDA)
/**
 * @brief Library handles of a PaRSEC CUDA execution stream, bound to it.
 */
typedef struct chameleon_parsec_cuda_handles_s {
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolverDn;
} chameleon_parsec_cuda_handles_t;

extern int              chameleon_parsec_ncudas;
extern parsec_info_id_t chameleon_parsec_cuda_handles_id;

static inline chameleon_parsec_cuda_handles_t *
chameleon_parsec_cuda_handles( parsec_gpu_exec_stream_t *gpu_stream ) {
    return (chameleon_parsec_cuda_handles_t *)parsec_info_get( &(gpu_stream->infos),
                                                               chameleon_parsec_cuda_handles_id );
}

static inline cudaStream_t
chameleon_parsec_cuda_stream( parsec_gpu_exec_stream_t *gpu_stream ) {
    return ((parsec_cuda_exec_stream_t *)gpu_stream)->cuda_stream;
}
#endif

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
    int8_t                  *allocated; /**< 1 if the tile memory is allocated by the runtime */
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
 * Deferred release of the taskpools (runtime_async.c)
 */
void chameleon_parsec_taskpool_release_all( void );

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

/*
 * IPIV descriptors (runtime_ipiv.c)
 * ---------------------------------
 * One vector collection per family (ipiv, perm, invp), indexed by the tile
 * index of the full vector and owned by the owner of the diagonal tile.
 */
typedef enum chameleon_parsec_ipiv_family_e {
    ChamParsecIpiv = 0,
    ChamParsecPerm = 1,
    ChamParsecInvp = 2,
} chameleon_parsec_ipiv_family_t;

typedef struct chameleon_parsec_ipiv_s {
    chameleon_parsec_vdc_t vdc[3];
    const CHAM_ipiv_t     *ipiv;
} chameleon_parsec_ipiv_t;

static inline int
chameleon_parsec_ipiv_key( const CHAM_ipiv_t *ipiv, int m ) {
    return m + ipiv->i / ipiv->mb;
}

static inline chameleon_parsec_vdc_t *
chameleon_parsec_ipiv_vdc( const CHAM_ipiv_t *ipiv, chameleon_parsec_ipiv_family_t family ) {
    return ((chameleon_parsec_ipiv_t *)(ipiv->ipiv))->vdc + family;
}

static inline parsec_dtd_tile_t *
chameleon_parsec_ipiv_tile( const RUNTIME_option_t *options, const CHAM_ipiv_t *ipiv,
                            chameleon_parsec_ipiv_family_t family, int m ) {
    return chameleon_parsec_vdc_tile_of( options, chameleon_parsec_ipiv_vdc( ipiv, family ),
                                         chameleon_parsec_ipiv_key( ipiv, m ) );
}

static inline int
chameleon_parsec_ipiv_arena( const CHAM_ipiv_t *ipiv, chameleon_parsec_ipiv_family_t family, int m ) {
    return chameleon_parsec_vdc_arena( chameleon_parsec_ipiv_vdc( ipiv, family ),
                                       chameleon_parsec_ipiv_key( ipiv, m ) );
}

/*
 * Pivot structures of the panel factorization (runtime_pivot.c)
 * -------------------------------------------------------------
 * One buffer per rank and parity of the column index h: key = 2 * rank + (h & 1).
 * The buffer holds a header followed by the pivot row and the diagonal row.
 *
 * DTD has no reduction: each rank accumulates the candidates of its tiles in
 * its own buffer (the first task of a step on a rank resets it), and the
 * buffers are then merged in the buffer of the root (owner of the diagonal
 * tile) which is read by the next step.
 */
#define CHAMELEON_PARSEC_PIVOT_HDR 64

typedef struct chameleon_parsec_pivot_hdr_s {
    int has_diag; /**< 1 if the diagonal row is stored, -1 otherwise */
    int h;        /**< Column index of the step                      */
    int blkm0;
    int blkidx;
} chameleon_parsec_pivot_hdr_t;

typedef struct chameleon_parsec_pivot_s {
    chameleon_parsec_vdc_t vdc;
    int                    NP;
    int                    root;       /**< Owner of the diagonal tile of the current panel */
    int                    stamp;      /**< Current step, incremented by each diagonal task */
    int                   *init_stamp; /**< Last step reset by each rank                    */
} chameleon_parsec_pivot_t;

static inline chameleon_parsec_pivot_t *
chameleon_parsec_pivot( const CHAM_desc_pivot_t *pivot ) {
    return (chameleon_parsec_pivot_t *)(pivot->nextpiv);
}

static inline int
chameleon_parsec_pivot_key( int rank, int h ) {
    return 2 * rank + (h & 1);
}

static inline parsec_dtd_tile_t *
chameleon_parsec_pivot_tile( const RUNTIME_option_t *options, const CHAM_desc_pivot_t *pivot,
                             int rank, int h ) {
    return chameleon_parsec_vdc_tile_of( options, &(chameleon_parsec_pivot( pivot )->vdc),
                                         chameleon_parsec_pivot_key( rank, h ) );
}

static inline int
chameleon_parsec_pivot_arena( const CHAM_desc_pivot_t *pivot ) {
    return chameleon_parsec_pivot( pivot )->vdc.arena_id;
}

static inline size_t
chameleon_parsec_pivot_size( int nb, cham_flttype_t dtyp ) {
    return CHAMELEON_PARSEC_PIVOT_HDR + 2 * (size_t)nb * CHAMELEON_Element_Size( dtyp );
}

/**
 * @brief Build the CHAM_pivot_t view of a pivot buffer.
 */
static inline void
chameleon_parsec_pivot_load( void *buf, int nb, cham_flttype_t dtyp, CHAM_pivot_t *piv ) {
    chameleon_parsec_pivot_hdr_t *hdr = (chameleon_parsec_pivot_hdr_t *)buf;
    char *rows = (char *)buf + CHAMELEON_PARSEC_PIVOT_HDR;

    piv->blkm0   = hdr->blkm0;
    piv->blkidx  = hdr->blkidx;
    piv->pivrow  = rows;
    piv->diagrow = rows + (size_t)nb * CHAMELEON_Element_Size( dtyp );
}

/**
 * @brief Save the scalar fields of the CHAM_pivot_t view in the buffer.
 */
static inline void
chameleon_parsec_pivot_store( void *buf, const CHAM_pivot_t *piv ) {
    chameleon_parsec_pivot_hdr_t *hdr = (chameleon_parsec_pivot_hdr_t *)buf;
    hdr->blkm0  = piv->blkm0;
    hdr->blkidx = piv->blkidx;
}

/**
 * @brief Reset a pivot buffer as the StarPU reduction initialization does.
 */
static inline void
chameleon_parsec_pivot_reset( void *buf, int nb, cham_flttype_t dtyp ) {
    chameleon_parsec_pivot_hdr_t *hdr = (chameleon_parsec_pivot_hdr_t *)buf;
    memset( buf, 0, chameleon_parsec_pivot_size( nb, dtyp ) );
    hdr->has_diag = -1;
    hdr->h        = -1;
}

/**
 * @brief Register a contribution of rank to the current step of the pivot
 * search, and return 1 if the buffer of rank must be reset by the task.
 *
 * The diagonal task is always the first one of a step: it sets the root (its
 * rank) and starts a new step. The stamps are updated at submission, which is
 * identical on all the ranks.
 */
static inline int
chameleon_parsec_pivot_contrib( const CHAM_desc_pivot_t *pivot, int rank, int isdiag ) {
    chameleon_parsec_pivot_t *ppivot = chameleon_parsec_pivot( pivot );
    int reset;

    if ( isdiag ) {
        ppivot->root = rank;
        ppivot->stamp++;
    }
    assert( ppivot->root != -1 );
    reset = ( ppivot->init_stamp[rank] != ppivot->stamp );
    ppivot->init_stamp[rank] = ppivot->stamp;
    return reset;
}

/*
 * Permutation workspaces (runtime_perm.c)
 * ---------------------------------------
 * Same keys and owners as StarPU. The buffer holds a header followed by the
 * index array and the rows.
 */
#define CHAMELEON_PARSEC_PERM_HDR 64

typedef struct chameleon_parsec_perm_hdr_s {
    int nindex; /**< Number of rows stored                     */
    int m;      /**< Maximal number of rows                    */
    int n;      /**< Number of columns of each row (row stride) */
    int side;
} chameleon_parsec_perm_hdr_t;

static inline int
chameleon_parsec_perm_key( const CHAM_perm_t *ws, int m, int n ) {
    return ( ws->side == ChamLeft ) ? m + n * ws->NP : n + m * ws->NP;
}

static inline size_t
chameleon_parsec_perm_rowsoff( int mrows ) {
    size_t off = CHAMELEON_PARSEC_PERM_HDR + sizeof(int) * (size_t)mrows;
    return ( off + 63 ) & ~((size_t)63);
}

static inline parsec_dtd_tile_t *
chameleon_parsec_perm_tile( const RUNTIME_option_t *options, const CHAM_perm_t *ws, int m, int n ) {
    return chameleon_parsec_vdc_tile_of( options, (chameleon_parsec_vdc_t *)(ws->ws),
                                         chameleon_parsec_perm_key( ws, m, n ) );
}

static inline int
chameleon_parsec_perm_arena( const CHAM_perm_t *ws ) {
    return ((chameleon_parsec_vdc_t *)(ws->ws))->arena_id;
}

/**
 * @brief Build the CHAM_laswpws_t view of a permutation workspace buffer.
 */
static inline void
chameleon_parsec_perm_load( void *buf, CHAM_laswpws_t *lws ) {
    chameleon_parsec_perm_hdr_t *hdr = (chameleon_parsec_perm_hdr_t *)buf;
    lws->index  = (int *)((char *)buf + CHAMELEON_PARSEC_PERM_HDR);
    lws->offset = chameleon_parsec_perm_rowsoff( hdr->m );
    lws->rows   = (char *)buf + lws->offset;
    lws->nindex = hdr->nindex;
}

static inline void
chameleon_parsec_perm_store( void *buf, const CHAM_laswpws_t *lws ) {
    ((chameleon_parsec_perm_hdr_t *)buf)->nindex = lws->nindex;
}

/**
 * PaRSEC packs the PARSEC_SCRATCH buffers right after the values in the task
 * without any alignment, while vectorized kernels may require aligned
 * workspaces. Scratch buffers are thus over-allocated with
 * CHAMELEON_PARSEC_SCRATCH_SIZE() and aligned in the task body with
 * chameleon_parsec_scratch_align().
 */
#define CHAMELEON_PARSEC_SCRATCH_ALIGN 64
#define CHAMELEON_PARSEC_SCRATCH_SIZE( _size_ ) ( (_size_) + CHAMELEON_PARSEC_SCRATCH_ALIGN )

static inline void *
chameleon_parsec_scratch_align( void *ptr ) {
    uintptr_t addr = (uintptr_t)ptr;
    return (void *)( (addr + CHAMELEON_PARSEC_SCRATCH_ALIGN - 1) & ~((uintptr_t)CHAMELEON_PARSEC_SCRATCH_ALIGN - 1) );
}

/**
 * @brief Task classes of the codelets with several incarnations (CPU and GPU).
 *
 * A direct parsec_dtd_insert_task() creates a task class with a single
 * incarnation, so the codelets with a GPU version create their task class with
 * parsec_dtd_create_task_class() and add a chore per device type.
 *
 * A DTD task class belongs to a taskpool, i.e. to a sequence: it is created at
 * the first insertion of the codelet in the sequence, and destroyed with the
 * taskpool. All the ranks insert the same tasks in the same order, so the task
 * classes get the same identifiers everywhere. They are cached with the key of
 * the direct insertions (CPU body and number of flows), so that the DTD removes
 * them from its cache when the taskpool destroys them.
 */
typedef parsec_task_class_t *(*chameleon_parsec_tc_create_fct_t)( parsec_taskpool_t *tp );

static inline parsec_task_class_t *
chameleon_parsec_task_class( const RUNTIME_option_t        *options,
                             void                          *cpu_body,
                             int                            nb_flows,
                             chameleon_parsec_tc_create_fct_t create )
{
    parsec_taskpool_t       *tp  = (parsec_taskpool_t *)(options->sequence->schedopt);
    uint64_t                 key = (uint64_t)(uintptr_t)cpu_body + (uint64_t)nb_flows;
    parsec_dtd_task_class_t *dtc = parsec_dtd_find_task_class( (parsec_dtd_taskpool_t *)tp, key );
    parsec_task_class_t     *tc;

    if ( dtc != NULL ) {
        return &(dtc->super);
    }

    tc = create( tp );
    assert( tc->nb_flows == nb_flows );
    parsec_dtd_register_task_class( tp, key, tc );
    return tc;
}

#if defined(CHAMELEON_PARSEC_CUDA)
/**
 * @brief Add a CUDA chore to a task class, if PaRSEC drives CUDA devices.
 */
static inline void
chameleon_parsec_add_cuda_chore( parsec_taskpool_t *tp, parsec_task_class_t *tc,
                                 parsec_advance_task_function_t body )
{
    if ( chameleon_parsec_ncudas > 0 ) {
        parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CUDA, (void *)body );
    }
}

/**
 * The DTD copies the data written by a GPU task back to the host only if the
 * flow is marked with PARSEC_PUSHOUT: neither a CPU task nor the final flush
 * fetches it from the GPU. The written flows of the tasks that may run on a GPU
 * are thus all pushed out, so that the host copy is always up to date.
 */
#define CHAMELEON_PARSEC_GPU_OUT PARSEC_PUSHOUT
#else
#define CHAMELEON_PARSEC_GPU_OUT 0
#endif

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
