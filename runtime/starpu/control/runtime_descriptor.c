/**
 *
 * @file starpu/runtime_descriptor.c
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon StarPU descriptor routines
 *
 * @version 1.4.0
 * @author Cedric Augonnet
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @author Guillaume Sylvand
 * @author Raphael Boucherie
 * @author Samuel Thibault
 * @author Loris Lucido
 * @author Brieuc Nicolas
 * @date 2025-12-19
 *
 */
#include "chameleon_starpu_internal.h"

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
static void
runtime_data_clean_desc_tile( starpu_data_handle_t handle,
                              CHAM_tile_t         *tile );
#endif

/**
 *  Malloc/Free of the data
 */
#ifdef STARPU_MALLOC_SIMULATION_FOLDED
#define FOLDED STARPU_MALLOC_SIMULATION_FOLDED
#else
#define FOLDED 0
#endif

void *RUNTIME_malloc( size_t size )
{
#if defined(CHAMELEON_SIMULATION) && !defined(STARPU_MALLOC_SIMULATION_FOLDED) && !defined(CHAMELEON_USE_MPI)
    return (void*) 1;
#else
    void *ptr;

    if (starpu_malloc_flags(&ptr, size, STARPU_MALLOC_PINNED|FOLDED|STARPU_MALLOC_COUNT) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void RUNTIME_free( void  *ptr,
                   size_t size )
{
#if defined(CHAMELEON_SIMULATION) && !defined(STARPU_MALLOC_SIMULATION_FOLDED) && !defined(CHAMELEON_USE_MPI)
    (void)ptr; (void)size;
    return;
#else
    starpu_free_flags(ptr, size, STARPU_MALLOC_PINNED|FOLDED|STARPU_MALLOC_COUNT);
#endif
}

#if defined(CHAMELEON_USE_CUDA)

#define gpuError_t              cudaError_t
#define gpuHostRegister         cudaHostRegister
#define gpuHostUnregister       cudaHostUnregister
#define gpuHostRegisterPortable cudaHostRegisterPortable
#define gpuSuccess              cudaSuccess
#define gpuGetErrorString       cudaGetErrorString

#elif defined(CHAMELEON_USE_HIP)

#define gpuError_t              hipError_t
#define gpuHostRegister         hipHostRegister
#define gpuHostUnregister       hipHostUnregister
#define gpuHostRegisterPortable hipHostRegisterPortable
#define gpuSuccess              hipSuccess
#define gpuGetErrorString       hipGetErrorString

#endif

/**
 *  Create data descriptor
 */
void RUNTIME_desc_create( CHAM_desc_t *desc )
{
    int64_t lmt = desc->lmt;
    int64_t lnt = desc->lnt;
    size_t  nbtiles = lmt * lnt;

    desc->occurences = 1;

    /*
     * Allocate starpu_handle_t array (handlers are initialized on the fly when
     * discovered by any algorithm to save space)
     */
    if ( cham_is_mixed( desc->dtyp ) ) {
        nbtiles *= 3;
    }

    desc->schedopt = (void*)calloc( nbtiles, sizeof(starpu_data_handle_t) );
    assert( desc->schedopt );

#if !defined(CHAMELEON_SIMULATION)
#if defined(CHAMELEON_USE_CUDA) || defined(CHAMELEON_USE_HIP)
    /*
     * Register allocated memory as GPU pinned memory
     */
    if ( (desc->use_mat == 1) && (desc->register_mat == 1) )
    {
        int64_t eltsze = CHAMELEON_Element_Size(desc->dtyp);
        size_t size = (size_t)(desc->llm) * (size_t)(desc->lln) * eltsze;
        gpuError_t rc;

        /* Register the matrix as pinned memory */
        rc = gpuHostRegister( desc->mat, size, gpuHostRegisterPortable );
        if ( rc != gpuSuccess )
        {
            /* Disable the unregister as register failed */
            desc->register_mat = 0;
            chameleon_warning("RUNTIME_desc_create(StarPU): gpuHostRegister - ", gpuGetErrorString( rc ));
        }
    }
#endif
#endif

    if (desc->ooc) {
        char *backend = chameleon_getenv( "STARPU_DISK_SWAP_BACKEND" );

        if (backend && strcmp(backend, "unistd_o_direct") == 0) {
            int     lastmm   = desc->lm - (desc->lmt-1) * desc->mb;
            int     lastnn   = desc->ln - (desc->lnt-1) * desc->nb;
            int64_t eltsze   = CHAMELEON_Element_Size(desc->dtyp);
            int     pagesize = getpagesize();

            if ( ((desc->mb * desc->nb * eltsze) % pagesize != 0) ||
                 ((lastmm   * desc->nb * eltsze) % pagesize != 0) ||
                 ((desc->mb * lastnn   * eltsze) % pagesize != 0) ||
                 ((lastmm   * lastnn   * eltsze) % pagesize != 0) )
            {
                chameleon_error("RUNTIME_desc_create",
                                "Matrix and tile size not suitable for out-of-core with the unistd_o_direct backend: all tiles have to be multiples of the system page size.\n"
                                      "Tip : choose 'n' and 'nb' as both multiples of 32, or use STARPU_DISK_SWAP_BACKEND=unistd to drop the o_direct part." );
                chameleon_cleanenv( backend );
                return;
            }
        }
        chameleon_cleanenv( backend );
    }

#if defined(CHAMELEON_USE_MPI)
    /*
     * Book the number of tags required to describe this matrix
     */
    {
        chameleon_starpu_tag_init( );
        desc->mpitag = chameleon_starpu_tag_book( nbtiles );

        if ( desc->mpitag == -1 ) {
            chameleon_fatal_error("RUNTIME_desc_create", "Can't pursue computation since no more tags are available");
            return;
        }
    }
#endif
}

/**
 *  Submit unregistration of the handles of a descriptor
 */
void RUNTIME_desc_destroy_submit( CHAM_desc_t              *desc,
                                  const RUNTIME_sequence_t *sequence )
{
    /*
     * If this is the last descriptor using the matrix, we release the handle
     */
    starpu_data_handle_t *handle = (starpu_data_handle_t*)(desc->schedopt);
    int64_t lmt     = desc->lmt;
    int64_t lnt     = desc->lnt;
    int64_t nbtiles = lmt * lnt;
    int64_t m;

    if ( cham_is_mixed( desc->dtyp ) ) {
        nbtiles *= 3;
    }

    for (m = 0; m < nbtiles; m++, handle++)
    {
        if ( *handle != NULL ) {
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
            if ( m < ( lmt * lnt ) ) {
                CHAM_tile_t *tile = desc->get_blktile( desc, m % lmt, m / lmt );

                if ( tile->format & CHAMELEON_TILE_DESC ) {
                    runtime_data_clean_desc_tile( *handle, tile );
                }
            }
#endif
            starpu_data_unregister_submit(*handle);
            /* StarPU has marked the handle for unregistering,
             * We don't need it anymore, put to NULL for later destroy
             */
            *handle = NULL;
        }
    }


    /*
     * WARNING: tags are not released in submit as they may always been used by
     * the runtime it can only be done by the synchronous destroy
     */
    (void)sequence;
}

/**
 *  Destroy data descriptor
 */
void RUNTIME_desc_destroy( CHAM_desc_t *desc )
{
    /*
     * If this is the last descriptor using the matrix, we release the handle
     * and unregister the GPU data
     */
    if ( desc->occurences > 0 ) {
        return;
    }

    starpu_data_handle_t *handle = (starpu_data_handle_t*)(desc->schedopt);
    int64_t lmt = desc->lmt;
    int64_t lnt = desc->lnt;
    int64_t nbtiles = lmt * lnt;
    int64_t m;

    if ( cham_is_mixed( desc->dtyp ) ) {
        nbtiles *= 3;
    }

    for (m = 0; m < nbtiles; m++, handle++) {
        if ( *handle != NULL ) {
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
            if ( m < ( lmt * lnt ) ) {
                CHAM_tile_t *tile = desc->get_blktile( desc, m % lmt, m / lmt );

                if ( tile->format & CHAMELEON_TILE_DESC ) {
                    runtime_data_clean_desc_tile( *handle, tile );
                }
            }
#endif
            starpu_data_unregister(*handle);
            *handle = NULL;
        }
    }

    /*
     * WARNING: tags and global memory are released here and not in the submit
     * as they may been used after the unregistration submission. It can only be
     * done by the synchronous destroy.
     */
#if !defined(CHAMELEON_SIMULATION)
#if defined(CHAMELEON_USE_CUDA) || defined(CHAMELEON_USE_HIP)
    if ( (desc->use_mat == 1) && (desc->register_mat == 1) )
    {
        /* Unmap the pinned memory associated to the matrix */
        if (gpuHostUnregister(desc->mat) != gpuSuccess)
        {
            chameleon_warning("RUNTIME_desc_destroy(StarPU)",
                              "gpuHostUnregister failed to unregister the "
                              "pinned memory associated to the matrix");
        }
    }
#endif
#endif
    chameleon_starpu_tag_release( desc->mpitag );

    free( desc->schedopt );
}

/**
 *  Acquire data
 */
int RUNTIME_desc_acquire( const CHAM_desc_t *desc )
{
    starpu_data_handle_t *handle = (starpu_data_handle_t*)(desc->schedopt);
    int lmt = desc->lmt;
    int lnt = desc->lnt;
    int m, n;

    for (n = 0; n < lnt; n++) {
        for (m = 0; m < lmt; m++) {
            if ( (*handle == NULL) ||
                 !chameleon_desc_islocal( desc, m, n ) )
            {
                handle++;
                continue;
            }
            starpu_data_acquire( *handle, STARPU_RW );
            handle++;
        }
    }
    return CHAMELEON_SUCCESS;
}

/**
 *  Release data
 */
int RUNTIME_desc_release( const CHAM_desc_t *desc )
{
    starpu_data_handle_t *handle = (starpu_data_handle_t*)(desc->schedopt);
    int lmt = desc->lmt;
    int lnt = desc->lnt;
    int m, n;

    for (n = 0; n < lnt; n++) {
        for (m = 0; m < lmt; m++) {
            if ( (*handle == NULL) ||
                 !chameleon_desc_islocal( desc, m, n ) )
            {
                handle++;
                continue;
            }
            starpu_data_release(*handle);
            handle++;
        }
    }
    return CHAMELEON_SUCCESS;
}

/**
 *  Flush cached data
 */
void RUNTIME_flush( CHAM_context_t *chamctxt )
{
#if defined(CHAMELEON_USE_MPI)
    starpu_mpi_cache_flush_all_data( chamctxt->comm );
#else
    (void)chamctxt;
#endif
}

static void cl_flush_cpu_func( void *descr[], void *cl_arg )
{
    (void)descr;
    (void)cl_arg;
    return;
}

static struct starpu_codelet cl_flush =
{
    .where     = STARPU_CPU,
    .nbuffers  = 1,
    .cpu_funcs = { cl_flush_cpu_func },
    .modes     = { STARPU_RW },
    .model     = NULL
};

static inline int
__insert_task_flush( starpu_data_handle_t handle )
{
    struct starpu_task *task = starpu_task_create();
    STARPU_ASSERT(task);
    task->name = "chameleon_flush";

    task->cl = &cl_flush;

    STARPU_TASK_SET_HANDLE(task, handle, 0);

    int ret = starpu_task_submit(task);
    STARPU_ASSERT_MSG(ret != -ENODEV, "Failed to submit the data flush task\n");
    STARPU_ASSERT_MSG(!ret, "Task data flush failed with code: %d\n", ret);

    return 0;
}

static inline void
runtime_data_flush_one( const RUNTIME_sequence_t *sequence,
                        const CHAM_tile_t        *tile,
                        starpu_data_handle_t      handle,
                        int                       sync )
{
    int home_node;

    if ( handle == NULL ) {
        return;
    }

#if defined(CHAMELEON_USE_MPI)
    starpu_mpi_cache_flush( sequence->comm, handle );
#endif

    if ( sequence->myrank != tile->rank )
    {
        return;
    }

    home_node = starpu_data_get_home_node( handle );
    if ( sync && (home_node >= 0) ) {
        assert( home_node == 0 );
        __insert_task_flush( handle );
        /* starpu_data_acquire_on_node_cb( handle, home_node, STARPU_RW, */
        /*                                 (callback_fct_t)starpu_data_release, handle ); */
    }
    else {
        chameleon_starpu_data_wont_use( handle );
    }
}

/**
 *  Flush or retrieve data to its home location for later use outside the runtime
 */
void RUNTIME_desc_flush( CHAM_desc_t              *desc,
                         const RUNTIME_sequence_t *sequence )
{
    CHAM_tile_t          *tile;
    starpu_data_handle_t *handle = desc->schedopt;
    int imax = 1;
    int mt   = desc->mt;
    int nt   = desc->nt;
    int i, m, n;
    int sync = desc->sync;

    /* Fallback if the matrix is allocated by the runtime */
    if ( !desc->use_mat ) {
        sync = 0;
    }

    if ( cham_is_mixed( desc->dtyp ) ) {
        imax = 3;
    }

    for( i=0; i<imax; i++ )
    {
        tile = desc->tiles;
        for (n = 0; n < nt; n++)
        {
            for (m = 0; m < mt; m++, handle++, tile++)
            {
                runtime_data_flush_one( sequence, tile, *handle, sync );
            }
        }
         /* Only the main precision is synchronized */
        sync = 0;
    }
    desc->sync = 0;
}

void RUNTIME_data_flush( const RUNTIME_sequence_t *sequence,
                         const CHAM_desc_t *A, int m, int n )
{
    int     i, imax = 1;
    int64_t mm      = m + (A->i / A->mb);
    int64_t nn      = n + (A->j / A->nb);
    int64_t shift   = ((int64_t)(A->lmt)) * nn + mm;
    int64_t nbtiles = ((int64_t)(A->lmt)) * ((int64_t)(A->lnt));
    CHAM_tile_t          *tile   = A->get_blktile( A, m, n );
    starpu_data_handle_t *handle = A->schedopt;
    handle += shift;

    if ( cham_is_mixed( A->dtyp ) ) {
        imax = 3;
    }

    for( i=0; i<imax; i++ ) {
        starpu_data_handle_t *handlebis;
        handlebis = handle + i * nbtiles;

        runtime_data_flush_one( sequence, tile, *handlebis, 0 );
    }
    (void)sequence;
}

void RUNTIME_data_unregister( const RUNTIME_sequence_t *sequence,
                              const CHAM_desc_t *A, int Am, int An )
{
    int     i, imax = 1;
    int64_t mm      = Am + (A->i / A->mb);
    int64_t nn      = An + (A->j / A->nb);
    int64_t shift   = ((int64_t)(A->lmt)) * nn + mm;
    int64_t nbtiles = ((int64_t)(A->lmt)) * ((int64_t)(A->lnt));
    starpu_data_handle_t *handle = A->schedopt;
    handle += shift;

    if ( cham_is_mixed( A->dtyp ) ) {
        imax = 3;
    }

    for( i=0; i<imax; i++ ) {
        starpu_data_handle_t *handlebis;

        handlebis = handle + i * nbtiles;

        if ( *handlebis == NULL ) {
            continue;
        }

        starpu_data_unregister_submit( *handlebis );
        *handlebis = NULL;
    }
    (void)sequence;
}

#if defined(CHAMELEON_USE_MIGRATE)
void RUNTIME_data_migrate( const RUNTIME_sequence_t *sequence,
                           const CHAM_desc_t *A, int Am, int An, int new_rank )
{
#if defined(HAVE_STARPU_MPI_DATA_MIGRATE)
    int old_rank;
    starpu_data_handle_t *handle = (starpu_data_handle_t*)(A->schedopt);
    starpu_data_handle_t lhandle;
    handle += ((int64_t)(A->lmt) * (int64_t)An + (int64_t)Am);

    lhandle = *handle;
    if ( lhandle == NULL ) {
        /* Register the data */
        lhandle = RUNTIME_data_getaddr( A, Am, An );
    }
    old_rank = starpu_mpi_data_get_rank( lhandle );

    if ( old_rank != new_rank ) {
        starpu_mpi_data_migrate( sequence->comm, lhandle, new_rank );
    }

    (void)sequence;
#else
    (void)sequence; (void)A; (void)Am; (void)An; (void)new_rank;
#endif
}
#endif

/**
 *  Get data addr
 */
/* For older revision of StarPU, STARPU_MAIN_RAM is not defined */
#ifndef STARPU_MAIN_RAM
#define STARPU_MAIN_RAM 0
#endif

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
static void
runtime_data_clean_desc_tile( starpu_data_handle_t handle,
                              CHAM_tile_t         *tile )
{
    CHAM_desc_t *child_desc = (CHAM_desc_t *)tile->mat;
    unsigned     child_count;

    assert( child_desc != NULL );

    child_count = child_desc->lmt * child_desc->lnt;

#if defined(CHAMELEON_USE_MPI)
    starpu_mpi_data_partition_clean_node( handle, child_count,
                                          (starpu_data_handle_t *)(child_desc->schedopt),
                                          STARPU_MAIN_RAM, MPI_COMM_WORLD );
#else
    starpu_data_partition_clean_node( handle, child_count,
                                      (starpu_data_handle_t *)(child_desc->schedopt),
                                      STARPU_MAIN_RAM );
#endif

    memset( child_desc->schedopt, 0, child_count * sizeof(starpu_data_handle_t) );
}

static void
runtime_data_register_desc_tile( const CHAM_desc_t    *A,
                                 int                   m,
                                 int                   n,
                                 starpu_data_handle_t *ptrtile,
                                 CHAM_tile_t          *tile,
                                 int64_t               tag,
                                 cham_flttype_t        flttype )
{
    struct starpu_data_filter filter_tile = {
        .filter_func = chameleon_recursive_tile_filter,
    };
    CHAM_desc_t          *child_desc = (CHAM_desc_t *)tile->mat;
    starpu_data_handle_t *child_handle;
    int64_t               child_ind;
    int                   child_count;
    int                   owner;
#if defined(CHAMELEON_USE_MPI)
    int                   parent_owner = STARPU_MPI_MULTIPLE_NODE_WITHOUT_ME;
#endif

    assert( child_desc != NULL );

    filter_tile.nchildren = child_desc->lmt * child_desc->lnt;
    child_count  = filter_tile.nchildren;
    child_handle = (starpu_data_handle_t *)(child_desc->schedopt);

    starpu_cham_tile_register( ptrtile, STARPU_MAIN_RAM, tile, flttype );
    starpu_data_partition_plan( *ptrtile, &filter_tile, child_handle );

    for ( child_ind = 0; child_ind < child_count; child_ind++, child_handle++ ) {
        int child_m = child_ind % child_desc->lmt;
        int child_n = child_ind / child_desc->lmt;
        CHAM_tile_t *child_tile = child_desc->get_blktile( child_desc, child_m, child_n );

        owner = child_desc->get_rankof( child_desc, child_m, child_n );
        starpu_cham_tile_child_set( child_handle, STARPU_MAIN_RAM,
                                    child_tile, flttype );

#if defined(CHAMELEON_KERNELS_TRACE)
        starpu_data_set_name( *child_handle, child_tile->name );
#endif

#if defined(CHAMELEON_USE_MPI)
        if ( owner == A->myrank ) {
            parent_owner = STARPU_MPI_MULTIPLE_NODE_WITH_ME;
            starpu_subdata_ptr_register( *child_handle, STARPU_MAIN_RAM );
        }
        starpu_mpi_data_register( *child_handle, child_desc->mpitag + child_ind, owner );
#endif
    }

#if defined(CHAMELEON_USE_MPI)
    starpu_mpi_register_hierarchy( *ptrtile, child_count,
                                   (starpu_data_handle_t *)(child_desc->schedopt) );
    starpu_mpi_data_register( *ptrtile, tag, parent_owner );
#else
    (void)tag;
#endif

#if defined(HAVE_STARPU_DATA_SET_OOC_FLAG)
    if ( A->ooc == 0 ) {
        starpu_data_set_ooc_flag( *ptrtile, 0 );
    }
#endif

#if defined(HAVE_STARPU_DATA_SET_COORDINATES)
    starpu_data_set_coordinates( *ptrtile, 2, m, n );
#else
    (void)m;
    (void)n;
#endif
}
#endif

void *RUNTIME_data_getaddr( const CHAM_desc_t *A, int m, int n )
{
    int64_t mm = m + (A->i / A->mb);
    int64_t nn = n + (A->j / A->nb);

    starpu_data_handle_t *ptrtile = A->schedopt;
    ptrtile += ((int64_t)A->lmt) * nn + mm;

    if ( *ptrtile != NULL ) {
        return (void*)(*ptrtile);
    }

    int home_node = -1;
    int myrank = A->myrank;
    int owner  = A->get_rankof( A, m, n );
    CHAM_tile_t *tile = A->get_blktile( A, m, n );

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
    if ( tile->format & CHAMELEON_TILE_DESC ) {
        int64_t block_ind = A->lmt * nn + mm;

        runtime_data_register_desc_tile( A, m, n, ptrtile, tile,
                                         A->mpitag + block_ind,
                                         cham_get_flttype( A->dtyp ) );
        assert( *ptrtile );
        return (void*)(*ptrtile);
    }
#endif

    if ( myrank == owner ) {
        if ( (tile->format & CHAMELEON_TILE_HMAT) ||
             (tile->mat != NULL) )
        {
            home_node = STARPU_MAIN_RAM;
        }
    }

    starpu_cham_tile_register( ptrtile, home_node, tile, cham_get_flttype( A->dtyp ) );

#if defined(HAVE_STARPU_DATA_SET_OOC_FLAG)
    if ( A->ooc == 0 ) {
        starpu_data_set_ooc_flag( *ptrtile, 0 );
    }
#endif

#if defined(HAVE_STARPU_DATA_SET_COORDINATES)
    starpu_data_set_coordinates( *ptrtile, 2, m, n );
#endif

#if defined(CHAMELEON_USE_MPI)
    {
        int64_t block_ind = A->lmt * nn + mm;
        starpu_mpi_data_register( *ptrtile, A->mpitag + block_ind, owner );
    }
#endif /* defined(CHAMELEON_USE_MPI) */

    CHAMELEON_DEBUG( "starpu", "%s - %p registered with tag %ld\n",
                     tile->name, (void*)(*ptrtile), A->mpitag + A->lmt * nn + mm );

    assert( *ptrtile );
    return (void*)(*ptrtile);
}

void *RUNTIME_data_getaddr_withconversion( const RUNTIME_option_t *options,
                                           cham_access_t access, cham_flttype_t flttype,
                                           const CHAM_desc_t *A, int m, int n )
{
    int64_t mm = m + (A->i / A->mb);
    int64_t nn = n + (A->j / A->nb);

    CHAM_tile_t *tile = A->get_blktile( A, m, n );
    starpu_data_handle_t *ptrtile = A->schedopt;

    int     fltshift = (cham_get_arith( tile->flttype ) - cham_get_arith( flttype ) + 3 ) % 3;
    int64_t shift = (int64_t)fltshift * ((int64_t)A->lmt * (int64_t)A->lnt);
    shift = shift + ((int64_t)A->lmt) * nn + mm;

    /* Get the correct starpu_handle */
    ptrtile += shift;

    /* Invalidate copies on write access */
    if ( access & ChamW ) {
        starpu_data_handle_t *copy = ptrtile;
        assert( fltshift == 0 );

        /* Remove first copy */
        copy += ((int64_t)A->lmt * (int64_t)A->lnt);
        if ( *copy ) {
            starpu_data_unregister_no_coherency( *copy );
            *copy = NULL;
        }

        /* Remove second copy */
        copy += ((int64_t)A->lmt * (int64_t)A->lnt);
        if ( *copy ) {
            starpu_data_unregister_no_coherency( *copy );
            *copy = NULL;
        }
    }

    if ( *ptrtile != NULL ) {
        return (void*)(*ptrtile);
    }

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
    if ( ( tile->format & CHAMELEON_TILE_DESC ) && ( fltshift == 0 ) ) {
        runtime_data_register_desc_tile( A, m, n, ptrtile, tile,
                                         A->mpitag + shift, flttype );
        assert( *ptrtile );
        return (void*)(*ptrtile);
    }
#endif

    int home_node = -1;
    int myrank = A->myrank;
    int owner  = A->get_rankof( A, m, n );

    if ( (myrank == owner) && (shift == 0) ) {
        if ( (tile->format & CHAMELEON_TILE_HMAT) ||
             (tile->mat != NULL) )
        {
            home_node = STARPU_MAIN_RAM;
        }
    }

    starpu_cham_tile_register( ptrtile, home_node, tile, flttype );

#if defined(HAVE_STARPU_DATA_SET_OOC_FLAG)
    if ( A->ooc == 0 ) {
        starpu_data_set_ooc_flag( *ptrtile, 0 );
    }
#endif

#if defined(HAVE_STARPU_DATA_SET_COORDINATES)
    starpu_data_set_coordinates( *ptrtile, 3, m, n, cham_get_arith( flttype ) );
#endif

#if defined(CHAMELEON_USE_MPI)
    starpu_mpi_data_register( *ptrtile, A->mpitag + shift, owner );
#endif /* defined(CHAMELEON_USE_MPI) */

#if defined(CHAMELEON_KERNELS_TRACE)
    fprintf( stderr, "%s - %p registered with tag %ld\n",
             tile->name, (void*)(*ptrtile), A->mpitag + shift );
#endif
    assert( *ptrtile );

    /* Submit the data conversion */
    if (( fltshift != 0 ) && (access & ChamR) && (owner == myrank) ) {
        starpu_data_handle_t *fromtile = A->schedopt;
        starpu_data_handle_t *totile = ptrtile;

        fromtile += ((int64_t)A->lmt) * nn + mm;
        assert( fromtile != totile );
        assert( tile->flttype != flttype );
        if ( *fromtile != NULL ) {
            insert_task_convert( options, tile->m, tile->n, tile->flttype, *fromtile, flttype, *totile );
        }
    }
    return (void*)(*ptrtile);
}
