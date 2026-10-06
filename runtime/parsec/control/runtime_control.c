/**
 *
 * @file parsec/runtime_control.c
 *
 * @copyright 2012-2017 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC control routines
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @author Samuel Thibault
 * @author Philippe Swartvagher
 * @author Matthieu Kuhn
 * @author Florent Pruvost
 * @date 2025-12-19
 *
 */
#include "chameleon_parsec.h"

#if defined(CHAMELEON_USE_MPI)
#include <mpi.h>
#endif
#include <parsec/utils/mca_param.h>

extern char **environ;

#if defined(CHAMELEON_PARSEC_CUDA)
int              chameleon_parsec_ncudas          = 0;
parsec_info_id_t chameleon_parsec_cuda_handles_id = PARSEC_INFO_ID_UNDEFINED;

/* Handles of the main thread, for the queries of the algorithms (workspace sizes) */
static cublasHandle_t     chameleon_parsec_cublas_handle     = NULL;
static cusolverDnHandle_t chameleon_parsec_cusolverDn_handle = NULL;

/**
 * @brief Create the cuBLAS and cuSOLVER handles of a CUDA execution stream.
 *
 * Called by PaRSEC the first time a task body gets them on this stream, so the
 * device of the stream is already the current one.
 */
static void *
chameleon_parsec_cuda_handles_create( void *obj, void *user )
{
    parsec_cuda_exec_stream_t       *stream  = (parsec_cuda_exec_stream_t *)obj;
    chameleon_parsec_cuda_handles_t *handles = malloc( sizeof(chameleon_parsec_cuda_handles_t) );

    if ( cublasCreate( &(handles->cublas) ) != CUBLAS_STATUS_SUCCESS ) {
        chameleon_fatal_error( "chameleon_parsec_cuda_handles_create", "cublasCreate() failed" );
        free( handles );
        return NULL;
    }
    cublasSetStream( handles->cublas, stream->cuda_stream );

    if ( cusolverDnCreate( &(handles->cusolverDn) ) != CUSOLVER_STATUS_SUCCESS ) {
        chameleon_fatal_error( "chameleon_parsec_cuda_handles_create", "cusolverDnCreate() failed" );
        cublasDestroy( handles->cublas );
        free( handles );
        return NULL;
    }
    cusolverDnSetStream( handles->cusolverDn, stream->cuda_stream );

    (void)user;
    return handles;
}

static void
chameleon_parsec_cuda_handles_destroy( void *obj, void *user )
{
    chameleon_parsec_cuda_handles_t *handles = (chameleon_parsec_cuda_handles_t *)obj;

    cublasDestroy( handles->cublas );
    cusolverDnDestroy( handles->cusolverDn );
    free( handles );
    (void)user;
}

static void
chameleon_parsec_cuda_init( void )
{
    uint32_t i;

    chameleon_parsec_ncudas = 0;
    for ( i = 0; i < parsec_nb_devices; i++ ) {
        parsec_device_module_t *device = parsec_mca_device_get( i );
        if ( (device != NULL) && (device->type == PARSEC_DEV_CUDA) ) {
            chameleon_parsec_ncudas++;
        }
    }
    if ( chameleon_parsec_ncudas == 0 ) {
        return;
    }

    chameleon_parsec_cuda_handles_id =
        parsec_info_register( &parsec_per_stream_infos, "CHAMELEON::CUDA::HANDLES",
                              chameleon_parsec_cuda_handles_destroy, NULL,
                              chameleon_parsec_cuda_handles_create, NULL, NULL );
    assert( chameleon_parsec_cuda_handles_id != PARSEC_INFO_ID_UNDEFINED );

    cublasCreate( &chameleon_parsec_cublas_handle );
    cusolverDnCreate( &chameleon_parsec_cusolverDn_handle );
}

static void
chameleon_parsec_cuda_fini( void )
{
    if ( chameleon_parsec_ncudas == 0 ) {
        return;
    }

    parsec_info_unregister( &parsec_per_stream_infos, chameleon_parsec_cuda_handles_id, NULL );
    chameleon_parsec_cuda_handles_id = PARSEC_INFO_ID_UNDEFINED;

    cublasDestroy( chameleon_parsec_cublas_handle );
    cusolverDnDestroy( chameleon_parsec_cusolverDn_handle );
    chameleon_parsec_cublas_handle     = NULL;
    chameleon_parsec_cusolverDn_handle = NULL;
    chameleon_parsec_ncudas            = 0;
}
#endif /* defined(CHAMELEON_PARSEC_CUDA) */

/**
 * @brief Give the number of CUDA devices to PaRSEC, unless the user already
 * did it through the environment. A negative number keeps the PaRSEC default
 * (all the devices).
 */
static void
chameleon_parsec_set_ncudas( int ncudas )
{
    static int set_by_chameleon = 0;
    char       value[16];

    if ( (ncudas < 0) ||
         (!set_by_chameleon && (getenv( "PARSEC_MCA_device_cuda_enabled" ) != NULL)) )
    {
        return;
    }
    snprintf( value, sizeof(value), "%d", ncudas );
    parsec_setenv_mca_param( "device_cuda_enabled", value, &environ );
    set_by_chameleon = 1;
}

/**
 * Initialize CHAMELEON
 */
int RUNTIME_init( CHAM_context_t *chamctxt,
                  int ncpus,
                  int ncudas,
                  int nthreads_per_worker )
{
    int hres = CHAMELEON_ERR_NOT_INITIALIZED;
    int default_ncores = -1;
    int *argc = (int *)malloc(sizeof(int));
    *argc = 0;

    /* Initializing parsec context */
    if( 0 < ncpus ) {
        default_ncores = ncpus;
    }
    chamctxt->parallel_enabled = CHAMELEON_TRUE;
    chameleon_parsec_set_ncudas( ncudas );
    chamctxt->schedopt = (void *)parsec_init(default_ncores, argc, NULL);

    if ( NULL != chamctxt->schedopt ) {
#if defined(CHAMELEON_USE_MPI)
        /* PaRSEC uses MPI_COMM_WORLD by default */
        int same;
        MPI_Comm_compare( chamctxt->comm, MPI_COMM_WORLD, &same );
        if ( same != MPI_IDENT ) {
            parsec_remote_dep_set_ctx( (parsec_context_t *)(chamctxt->schedopt),
                                       (intptr_t)(chamctxt->comm) );
        }
#endif
        chamctxt->nworkers = ncpus;
        chamctxt->nthreads_per_worker = nthreads_per_worker;
#if defined(CHAMELEON_PARSEC_CUDA)
        chameleon_parsec_cuda_init();
        chamctxt->ncudas = chameleon_parsec_ncudas;
#else
        chamctxt->ncudas = 0;
#endif
        hres = CHAMELEON_SUCCESS;
    }

    free(argc);

    return hres;
}

/**
 * Finalize CHAMELEON
 */
void RUNTIME_finalize( CHAM_context_t *chamctxt )
{
    parsec_context_t *parsec = (parsec_context_t*)chamctxt->schedopt;

    /* Wait for the workers, the context must be started to be waited */
    parsec_context_start( parsec );
    parsec_context_wait( parsec );

    chameleon_parsec_taskpool_release_all();
#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_cuda_fini();
#endif
    chameleon_parsec_arena_fini( parsec );
    parsec_fini(&parsec);
    return;
}

/**
 *  To suspend the processing of new tasks by workers
 */
void RUNTIME_pause( CHAM_context_t *chamctxt )
{
    (void)chamctxt;
    return;
}

/**
 *  This is the symmetrical call to RUNTIME_pause,
 *  used to resume the workers polling for new tasks.
 */
void RUNTIME_resume( CHAM_context_t *chamctxt )
{
    (void)chamctxt;
    return;
}

/**
 * Barrier CHAMELEON.
 */
void RUNTIME_barrier( CHAM_context_t *chamctxt )
{
    /*
     * The tasks of the sequences are completed by their wait, and the
     * parsec_context_wait() is done at finalize: entering the wait of all the
     * taskpools of the context while some of them are being released by the
     * termination detection is not safe.
     */
#if defined(CHAMELEON_USE_MPI)
    MPI_Barrier( chamctxt->comm );
#endif
    (void)chamctxt;
    return;
}

/**
 *  Display a progress information when executing the tasks
 */
void RUNTIME_progress( CHAM_context_t *chamctxt )
{
    (void)chamctxt;
    return;
}

/**
 * Thread rank.
 */
int RUNTIME_thread_rank( const CHAM_context_t *chamctxt )
{
    (void)chamctxt;
    return 0;
}

/**
 * Thread rank.
 */
int RUNTIME_thread_size( const CHAM_context_t *chamctxt )
{
    // TODO: fixme
    //return vpmap_get_nb_total_threads();
    (void)chamctxt;
    return 1;
}

/**
 *  This returns the rank of this process
 */
int RUNTIME_comm_rank( const CHAM_context_t *chamctxt )
{
    int rank = 0;
#if defined(CHAMELEON_USE_MPI)
    MPI_Comm_rank( chamctxt->comm, &rank );
#endif

    (void)chamctxt;
    return rank;
}

/**
 *  This returns the size of the distributed computation
 */
int RUNTIME_comm_size( const CHAM_context_t *chamctxt )
{
    int size = 1;
#if defined(CHAMELEON_USE_MPI)
    MPI_Comm_size( chamctxt->comm, &size );
#endif

    (void)chamctxt;
    return size;
}

void RUNTIME_set_minmax_submitted_tasks( int min, int max ) {
    (void)min;
    (void)max;
}

#if !defined(CHAMELEON_SIMULATION)
#if defined(CHAMELEON_USE_CUDA)
/*
 * The task bodies get the handles of their stream with
 * chameleon_parsec_cuda_handles(). These ones belong to the main thread and are
 * only meant for the queries of the algorithms, as the workspace sizes.
 */
cublasHandle_t
RUNTIME_get_cublas_handle()
{
#if defined(CHAMELEON_PARSEC_CUDA)
    return chameleon_parsec_cublas_handle;
#else
    return NULL;
#endif
}

cusolverDnHandle_t
RUNTIME_get_cusolverDn_handle()
{
#if defined(CHAMELEON_PARSEC_CUDA)
    return chameleon_parsec_cusolverDn_handle;
#else
    return NULL;
#endif
}
#elif defined(CHAMELEON_USE_HIP)
hipblasHandle_t
RUNTIME_get_hipblas_handle()
{
    assert(0);
    return NULL;
}

hipsolverDnHandle_t
RUNTIME_get_hipsolverDn_handle()
{
    assert(0);
    return NULL;
}
#endif
#endif
