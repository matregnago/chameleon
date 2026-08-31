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
 * @brief Allocate the StarPU-specific state and its handle slots.
 */
static starpu_cham_schedopt_t *
runtime_desc_schedopt_create( size_t nhandles )
{
    starpu_cham_schedopt_t *schedopt;
    size_t                  size;

    size = sizeof(*schedopt) + nhandles * sizeof(schedopt->handles[0]);
    schedopt = calloc( 1, size );
    if ( schedopt != NULL ) {
        schedopt->nhandles = nhandles;
    }
    return schedopt;
}

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
    starpu_cham_schedopt_t *schedopt;
    int64_t                 lmt = desc->lmt;
    int64_t                 lnt = desc->lnt;
    size_t                  nbtiles = (size_t)lmt * (size_t)lnt;

    desc->occurences = 1;

    /*
     * Allocate all handle slots up front. Individual StarPU handles are
     * created lazily when an algorithm first requests their tiles.
     */
    if ( cham_is_mixed( desc->dtyp ) ) {
        nbtiles *= 3;
    }

    /*
     * Keep descriptor-owned StarPU state and its flexible handle array in one
     * typed allocation.
     */
    schedopt = runtime_desc_schedopt_create( nbtiles );
    assert( schedopt != NULL );
    desc->schedopt = schedopt;

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
    /* Reserve the MPI-tag range once, on the descriptor that owns it. */
    {
        chameleon_starpu_tag_init( );
        if ( desc->mpitag < 0 ) {
            size_t tag_count = ( desc->mpitag_size > 0 )
                             ? (size_t)desc->mpitag_size
                             : nbtiles;

            desc->mpitag       = chameleon_starpu_tag_book( tag_count );
            desc->mpitag_size  = tag_count;
            desc->mpitag_owner = 1;
        }

        if ( desc->mpitag == -1 ) {
            chameleon_fatal_error("RUNTIME_desc_create", "Can't pursue computation since no more tags are available");
            return;
        }
    }
#endif
}

/**
 *  Unregister the handles of a descriptor
 *
 *  Keep this operation synchronous with StarPU: MPI tag ranges may be
 *  released by the subsequent descriptor destruction and must not be reused
 *  while an old handle is still registered.
 */
void RUNTIME_desc_destroy_submit( CHAM_desc_t              *desc,
                                  const RUNTIME_sequence_t *sequence )
{
    starpu_cham_schedopt_t *schedopt;
    starpu_data_handle_t   *handle;
    int64_t                 lmt = desc->lmt;
    int64_t                 lnt = desc->lnt;
    int64_t                 tile_count = lmt * lnt;
    int64_t                 nbtiles;
    int64_t                 m;

    /*
     * If this is the last descriptor using the matrix, we release the handle
     */
    schedopt = chameleon_starpu_desc_get_schedopt( desc );
    handle   = schedopt->handles;
    nbtiles  = (int64_t)schedopt->nhandles;

    for (m = 0; m < nbtiles; m++, handle++)
    {
        if ( *handle != NULL ) {
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
            if ( m < tile_count ) {
                CHAM_tile_t *tile = desc->get_blktile( desc, m % lmt, m / lmt );

                if ( tile->format & CHAMELEON_TILE_DESC ) {
                    runtime_data_clean_desc_tile( *handle, tile );
                }
            }
#endif
            starpu_data_unregister(*handle); /* _submit */
            *handle = NULL;
        }
    }


    /*
     * WARNING: tags are not released in submit as they may always been used by
     * the runtime it can only be done by the synchronous destroy
     */
    (void)sequence;
    (void)tile_count;
}

/**
 *  Destroy data descriptor
 */
void RUNTIME_desc_destroy( CHAM_desc_t *desc )
{
    starpu_cham_schedopt_t *schedopt;
    starpu_data_handle_t   *handle;
    int64_t                 lmt = desc->lmt;
    int64_t                 lnt = desc->lnt;
    int64_t                 tile_count = lmt * lnt;
    int64_t                 nbtiles;
    int64_t                 m;

    /*
     * If this is the last descriptor using the matrix, we release the handle
     * and unregister the GPU data
     */
    if ( desc->occurences > 0 ) {
        return;
    }

    schedopt = chameleon_starpu_desc_get_schedopt( desc );
    handle   = schedopt->handles;
    nbtiles  = (int64_t)schedopt->nhandles;

    for (m = 0; m < nbtiles; m++, handle++) {
        if ( *handle != NULL ) {
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
            if ( m < tile_count ) {
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
    if ( desc->mpitag_owner ) {
        chameleon_starpu_tag_release( desc->mpitag );
    }

    free( desc->schedopt );
}

/**
 *  Acquire data
 */
int RUNTIME_desc_acquire( const CHAM_desc_t *desc )
{
    starpu_data_handle_t *handle = chameleon_starpu_desc_get_handles( desc );
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
    starpu_data_handle_t *handle = chameleon_starpu_desc_get_handles( desc );
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
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
    if ( tile->format & CHAMELEON_TILE_DESC ) {
        int owner = starpu_mpi_data_get_rank( handle );

        if ( ( owner == STARPU_MPI_MULTIPLE_NODE ) ||
             ( owner == STARPU_MPI_MULTIPLE_NODE_WITH_ME ) ||
             ( owner == STARPU_MPI_MULTIPLE_NODE_WITHOUT_ME ) )
        {
            /* Flush the parent handle and every child registered below it. */
            starpu_mpi_cache_flush_recursive( /* sequence->comm, */ handle );
        }
        else {
            starpu_mpi_cache_flush( sequence->comm, handle );
        }
    }
    else
#endif
    {
        starpu_mpi_cache_flush( sequence->comm, handle );
    }
#endif

    if ( ( tile->rank != CHAMELEON_MPI_WITH_ME ) &&
         ( tile->rank != sequence->myrank ) )
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
    starpu_data_handle_t *handles = chameleon_starpu_desc_get_handles( desc );
    int64_t               nbtiles = (int64_t)desc->lmt * desc->lnt;
    int                   imax    = 1;
    int                   sync    = desc->sync;
    int                   i, m, n;

    /* Fallback if the matrix is allocated by the runtime */
    if ( !desc->use_mat ) {
        sync = 0;
    }

    if ( cham_is_mixed( desc->dtyp ) ) {
        imax = 3;
    }

    for ( i = 0; i < imax; i++ )
    {
        for ( n = 0; n < desc->nt; n++ )
        {
            int64_t nn = n + ( desc->j / desc->nb );

            for ( m = 0; m < desc->mt; m++ )
            {
                int64_t              mm     = m + ( desc->i / desc->mb );
                CHAM_tile_t         *tile   = desc->get_blktile( desc, m, n );
                starpu_data_handle_t handle =
                    handles[i * nbtiles + nn * desc->lmt + mm];

                runtime_data_flush_one( sequence, tile, handle, sync );
            }
        }
        /* Only the main precision is synchronized. */
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
    starpu_data_handle_t *handle = chameleon_starpu_desc_get_handles( A );
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
    starpu_data_handle_t *handle = chameleon_starpu_desc_get_handles( A );
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

        starpu_data_unregister( *handlebis );
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
    starpu_data_handle_t *handle = chameleon_starpu_desc_get_handles( A );
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

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
/**
 * @brief Clean a recursive StarPU handle hierarchy from leaves to root.
 *
 * A StarPU partition plan owns its immediate child handles. Descendant plans
 * must therefore be cleaned before the plan that created their parent handle.
 */
static void
runtime_data_clean_desc_tile( starpu_data_handle_t handle,
                              CHAM_tile_t         *tile )
{
    CHAM_desc_t          *child_desc = (CHAM_desc_t *)tile->mat;
    starpu_data_handle_t *child_handles;
    unsigned              child_count;
    unsigned              child_ind;
    int                   plan_count;

    assert( child_desc != NULL );

    /* Recursively clean children first. */
    child_count   = child_desc->mt * child_desc->nt;
    child_handles = malloc( (size_t)child_count * sizeof(starpu_data_handle_t) );
    assert( child_handles != NULL );
    plan_count    = starpu_data_partition_get_nplans( handle );

    for ( child_ind = 0; child_ind < child_count; child_ind++ ) {
        starpu_data_handle_t *child_handle;
        CHAM_tile_t          *child_tile;
        int                   child_m = child_ind % child_desc->mt;
        int                   child_n = child_ind / child_desc->mt;

        child_handle = chameleon_starpu_desc_get_handles( child_desc ) +
                       (size_t)child_n * child_desc->lmt + child_m;
        child_handles[child_ind] = *child_handle;
        child_tile   = child_desc->get_blktile( child_desc, child_m, child_n );

        if ( ( *child_handle != NULL ) &&
             ( child_tile->format & CHAMELEON_TILE_DESC ) )
        {
            runtime_data_clean_desc_tile( *child_handle, child_tile );
        }
    }

    /*
     * Recursive execution may already have cleaned the partition plan while
     * descendant handles remain registered in the persistent descriptor
     * arrays. Use StarPU's partition cleanup only for a live plan; otherwise,
     * unregister those remaining handles explicitly.
     */
    if ( plan_count > 0 ) {
#if defined(CHAMELEON_USE_MPI)
        starpu_mpi_data_partition_clean_node( handle, child_count,
                                              child_handles,
                                              STARPU_MAIN_RAM, MPI_COMM_WORLD );
#else
        starpu_data_partition_clean_node( handle, child_count,
                                          child_handles,
                                          STARPU_MAIN_RAM );
#endif
    }
    else {
        for ( child_ind = 0; child_ind < child_count; child_ind++ ) {
            if ( child_handles[child_ind] != NULL ) {
                starpu_data_unregister( child_handles[child_ind] );
            }
        }
    }

    for ( child_ind = 0; child_ind < child_count; child_ind++ ) {
        int child_m = child_ind % child_desc->mt;
        int child_n = child_ind / child_desc->mt;
        starpu_data_handle_t *child_handle =
            chameleon_starpu_desc_get_handles( child_desc ) +
            (size_t)child_n * child_desc->lmt + child_m;

        *child_handle = NULL;
    }
    free( child_handles );
}

/**
 * @brief Build the full-rank tile interface exposed to StarPU.
 *
 * CHAMELEON_TILE_DESC describes the Chameleon hierarchy layered over a
 * full-rank storage tile. StarPU must see the backing tile instead, so only
 * the hierarchy bit is removed. The data pointer is selected separately once
 * ownership is known.
 */
static CHAM_tile_t
runtime_data_get_starpu_tile( const CHAM_tile_t *tile )
{
    CHAM_tile_t runtime_tile = *tile;

    runtime_tile.mat = NULL;
    if ( runtime_tile.format & CHAMELEON_TILE_DESC ) {
        assert( runtime_tile.format & CHAMELEON_TILE_FULLRANK );
        runtime_tile.format &= ~CHAMELEON_TILE_DESC;
    }

    return runtime_tile;
}

/**
 * @brief Return the storage represented by a regular or recursive tile.
 */
static void *
runtime_data_get_tile_storage( const CHAM_tile_t *tile )
{
    if ( tile->format & CHAMELEON_TILE_DESC ) {
        const CHAM_desc_t *child_desc = (const CHAM_desc_t *)tile->mat;

        assert( child_desc != NULL );
        return child_desc->mat;
    }

    return tile->mat;
}

/**
 * @brief Register and partition one recursive descriptor tile.
 *
 * The parent handle may be new or may already exist as a child of an ancestor
 * partition plan. StarPU returns the new child handles in a compact @c mt by
 * @c nt array; this routine copies them into the descriptor array, whose row
 * stride is @c lmt, before registering their MPI ownership and hierarchy.
 */
static void
runtime_data_register_desc_tile( const CHAM_desc_t    *A,
                                 int                   m,
                                 int                   n,
                                 starpu_data_handle_t *ptrtile,
                                 CHAM_tile_t          *tile,
                                 int64_t               tag,
                                 cham_flttype_t        flttype )
{
    struct starpu_data_filter *filter_tile;
    CHAM_tile_t                runtime_tile;
    CHAM_desc_t               *child_desc = (CHAM_desc_t *)tile->mat;
    starpu_data_handle_t      *child_handles;
    void                      *tile_mat;
    int64_t                    child_ind;
    int                        child_count;
    int                        owner;
    int                        home_node = -1;
#if defined(CHAMELEON_USE_MPI)
    int                        parent_owner;
    int                        parent_is_multiple;
    int                        parent_registered = ( *ptrtile != NULL );
#endif

    assert( child_desc != NULL );

    owner = A->get_rankof( A, m, n );
#if defined(CHAMELEON_USE_MPI)
    parent_is_multiple =
        ( child_desc->mat == NULL ) ||
        ( owner == CHAMELEON_MPI_WITH_ME ) ||
        ( owner == CHAMELEON_MPI_WITHOUT_ME );
    parent_owner = parent_is_multiple ? STARPU_MPI_MULTIPLE_NODE_WITHOUT_ME
                                      : owner;
#endif
    assert( tile->format & CHAMELEON_TILE_DESC );
    runtime_tile = runtime_data_get_starpu_tile( tile );
    tile_mat     = runtime_data_get_tile_storage( tile );
    if ( ( ( owner == A->myrank )
#if !defined(CHAMELEON_USE_MPI)
           || ( owner == CHAMELEON_MPI_WITH_ME )
#endif
         ) &&
         ( tile_mat != NULL ) )
    {
        home_node        = STARPU_MAIN_RAM;
        runtime_tile.mat = tile_mat;
    }

    /* StarPU partition plans always return a compact array of child handles. */
    child_count   = child_desc->mt * child_desc->nt;
    child_handles = malloc( (size_t)child_count * sizeof(starpu_data_handle_t) );
    assert( child_handles != NULL );
    filter_tile  = runtime_desc_get_partition_filter( child_desc );
    assert( ( filter_tile->filter_func == NULL ) ||
            ( filter_tile->filter_func == chameleon_recursive_tile_filter ) );
    filter_tile->filter_func    = chameleon_recursive_tile_filter;
    filter_tile->nchildren      = child_count;
    filter_tile->filter_arg_ptr = child_desc;

    if ( *ptrtile == NULL ) {
        starpu_cham_tile_register( ptrtile, home_node, &runtime_tile, flttype );
    }
    else if ( ( home_node == STARPU_MAIN_RAM ) &&
              ( starpu_data_get_home_node( *ptrtile ) < 0 ) )
    {
        starpu_cham_tile_child_set( ptrtile, STARPU_MAIN_RAM,
                                    &runtime_tile, flttype );
        starpu_subdata_ptr_register( *ptrtile, STARPU_MAIN_RAM );
    }
    starpu_data_partition_plan( *ptrtile, filter_tile, child_handles );

    for ( child_ind = 0; child_ind < child_count; child_ind++ ) {
        int child_m = child_ind % child_desc->mt;
        int child_n = child_ind / child_desc->mt;
        CHAM_tile_t          *child_tile;
        CHAM_tile_t           runtime_child_tile;
        starpu_data_handle_t *child_handle =
            child_handles + child_ind;
        starpu_data_handle_t *desc_child_handle =
            chameleon_starpu_data_gethandle( child_desc, child_m, child_n );
        void                 *child_mat;

        child_tile         = child_desc->get_blktile( child_desc, child_m, child_n );
        runtime_child_tile = runtime_data_get_starpu_tile( child_tile );

        assert( *desc_child_handle == NULL );
        *desc_child_handle = *child_handle;

        child_mat = runtime_data_get_tile_storage( child_tile );

        owner = child_desc->get_rankof( child_desc, child_m, child_n );
        if ( ( owner == A->myrank ) ||
             ( owner == CHAMELEON_MPI_WITH_ME ) )
        {
#if defined(CHAMELEON_USE_MPI)
            if ( parent_is_multiple ) {
                parent_owner = STARPU_MPI_MULTIPLE_NODE_WITH_ME;
            }
#endif
            if ( ( child_desc->mat != NULL ) &&
                 ( child_mat != NULL ) &&
                 ( !( child_tile->format & CHAMELEON_TILE_DESC ) ||
                   ( owner != CHAMELEON_MPI_WITH_ME ) ) )
            {
                runtime_child_tile.mat = child_mat;
            }
        }

        starpu_cham_tile_child_set( child_handle, STARPU_MAIN_RAM,
                                    &runtime_child_tile, flttype );
        if ( ( runtime_child_tile.mat != NULL ) &&
             ( starpu_data_get_home_node( *child_handle ) < 0 ) )
        {
            starpu_subdata_ptr_register( *child_handle, STARPU_MAIN_RAM );
        }

#if defined(CHAMELEON_KERNELS_TRACE)
        starpu_data_set_name( *child_handle, child_tile->name );
#endif

#if defined(CHAMELEON_USE_MPI)
        starpu_mpi_data_register(
            *child_handle,
            chameleon_desc_get_mpi_tag( child_desc, child_m, child_n, 0 ), owner );
#endif
    }

#if defined(CHAMELEON_USE_MPI)
    starpu_mpi_register_hierarchy( *ptrtile, child_count, child_handles );
    if ( parent_registered ) {
        starpu_mpi_data_set_rank( *ptrtile, parent_owner );
    }
    else {
        starpu_mpi_data_register( *ptrtile, tag, parent_owner );
    }
#else
    (void)tag;
#endif

    free( child_handles );

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

/**
 *  Get data addr
 */
void *RUNTIME_data_getaddr( const CHAM_desc_t *A, int m, int n )
{
    int64_t mm = m + (A->i / A->mb);
    int64_t nn = n + (A->j / A->nb);

    starpu_data_handle_t *ptrtile = chameleon_starpu_desc_get_handles( A );
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
        runtime_data_register_desc_tile( A, m, n, ptrtile, tile,
                                         chameleon_desc_get_mpi_tag( A, mm, nn, 0 ),
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
    starpu_mpi_data_register( *ptrtile,
                              chameleon_desc_get_mpi_tag( A, mm, nn, 0 ), owner );
#endif /* defined(CHAMELEON_USE_MPI) */

    CHAMELEON_DEBUG( "starpu", "%s - %p registered with tag %ld\n",
                     tile->name, (void*)(*ptrtile),
                     chameleon_desc_get_mpi_tag( A, mm, nn, 0 ) );

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
    starpu_data_handle_t *ptrtile = chameleon_starpu_desc_get_handles( A );

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
                                         chameleon_desc_get_mpi_tag( A, mm, nn, 0 ),
                                         flttype );
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
    starpu_mpi_data_register(
        *ptrtile, chameleon_desc_get_mpi_tag( A, mm, nn, fltshift ), owner );
#endif /* defined(CHAMELEON_USE_MPI) */

#if defined(CHAMELEON_KERNELS_TRACE)
    fprintf( stderr, "%s - %p registered with tag %ld\n",
             tile->name, (void*)(*ptrtile),
             chameleon_desc_get_mpi_tag( A, mm, nn, fltshift ) );
#endif
    assert( *ptrtile );

    /* Submit the data conversion */
    if (( fltshift != 0 ) && (access & ChamR) && (owner == myrank) ) {
        starpu_data_handle_t *fromtile = chameleon_starpu_desc_get_handles( A );
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
