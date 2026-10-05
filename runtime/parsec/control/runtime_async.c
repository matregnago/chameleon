/**
 *
 * @file parsec/runtime_async.c
 *
 * @copyright 2012-2017 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC asynchronous routines
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @author Florent Pruvost
 * @date 2025-12-19
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/getenv.h"
#include <parsec/interfaces/dtd/insert_function_internal.h>

#include <sched.h>
#include <pthread.h>

/**
 * @brief Wait for a taskpool, and for the release of the reference held by
 * the thread that detected its termination.
 *
 * The termination detector sets the taskpool as terminated before releasing
 * its own reference, so parsec_taskpool_wait() may return while another
 * thread still uses the taskpool. If the taskpool is freed at this time, it
 * stays alive in the context list and a following parsec_context_wait() (used
 * by RUNTIME_barrier()) enters it again, racing with the delayed release that
 * then destroys a taskpool still in use.
 */
static void
chameleon_parsec_taskpool_wait( parsec_taskpool_t *tp )
{
    parsec_object_t *obj = (parsec_object_t *)tp;

    parsec_taskpool_wait( tp );
    while ( parsec_atomic_fetch_add_int32( &(obj->obj_reference_count), 0 ) > 1 ) {
        sched_yield();
    }
}

/*
 * Deferred release of the taskpools
 * ---------------------------------
 *
 * In rare cases, a worker of PaRSEC still completes the release of the last
 * task of a DTD taskpool after the wait on this taskpool has returned. The
 * taskpools are thus kept alive (idle, in the context) for a few sequences
 * before being released, and the remaining ones are released at finalize.
 */
#define CHAMELEON_PARSEC_TP_DEFERRED 16

static parsec_taskpool_t *chameleon_parsec_tp_ring[CHAMELEON_PARSEC_TP_DEFERRED] = { NULL };
static int                chameleon_parsec_tp_next = 0;
static pthread_mutex_t    chameleon_parsec_tp_lock = PTHREAD_MUTEX_INITIALIZER;

static void
chameleon_parsec_taskpool_release( parsec_taskpool_t *tp )
{
    parsec_taskpool_t *old;

    pthread_mutex_lock( &chameleon_parsec_tp_lock );
    old = chameleon_parsec_tp_ring[chameleon_parsec_tp_next];
    chameleon_parsec_tp_ring[chameleon_parsec_tp_next] = tp;
    chameleon_parsec_tp_next = (chameleon_parsec_tp_next + 1) % CHAMELEON_PARSEC_TP_DEFERRED;
    pthread_mutex_unlock( &chameleon_parsec_tp_lock );

    if ( old != NULL ) {
        parsec_taskpool_free( old );
    }
}

void
chameleon_parsec_taskpool_release_all( void )
{
    int i;

    pthread_mutex_lock( &chameleon_parsec_tp_lock );
    for ( i = 0; i < CHAMELEON_PARSEC_TP_DEFERRED; i++ ) {
        if ( chameleon_parsec_tp_ring[i] != NULL ) {
            parsec_taskpool_free( chameleon_parsec_tp_ring[i] );
            chameleon_parsec_tp_ring[i] = NULL;
        }
    }
    chameleon_parsec_tp_next = 0;
    pthread_mutex_unlock( &chameleon_parsec_tp_lock );
}

#if defined(CHAMELEON_USE_MPI)
#include <mpi.h>

/**
 * @brief Check that all the ranks inserted the same number of tasks.
 *
 * In DTD, a task is identified by its insertion counter in the taskpool, so
 * every rank must insert the same sequence of tasks. A divergence leads to a
 * hang in the wait: when CHAMELEON_PARSEC_CHECK_STREAM is set, abort with a
 * message instead.
 */
static void
chameleon_parsec_check_stream( CHAM_context_t *chamctxt, parsec_taskpool_t *tp )
{
    static int check = -1;
    int        local[2], global[2];

    if ( check == -1 ) {
        check = chameleon_env_on_off( "CHAMELEON_PARSEC_CHECK_STREAM", CHAMELEON_FALSE );
    }
    if ( !check ) {
        return;
    }

    local[0] =  ((parsec_dtd_taskpool_t *)tp)->task_id;
    local[1] = -((parsec_dtd_taskpool_t *)tp)->task_id;
    MPI_Allreduce( local, global, 2, MPI_INT, MPI_MAX, chamctxt->comm );
    if ( global[0] != -global[1] ) {
        chameleon_fatal_error( "RUNTIME_sequence_wait",
                               "The ranks inserted different numbers of tasks in the same taskpool" );
        MPI_Abort( chamctxt->comm, 1 );
    }
}
#else
#define chameleon_parsec_check_stream( _ctx_, _tp_ ) do { (void)(_ctx_); (void)(_tp_); } while(0)
#endif

/**
 *  Create a sequence
 */
int RUNTIME_sequence_create( CHAM_context_t     *chamctxt,
                             RUNTIME_sequence_t *sequence )
{
    parsec_context_t  *parsec        = (parsec_context_t *)(chamctxt->schedopt);
    parsec_taskpool_t *parsec_dtd_tp = parsec_dtd_taskpool_new();

    parsec_context_add_taskpool( parsec, (parsec_taskpool_t *)parsec_dtd_tp );
    sequence->schedopt = parsec_dtd_tp;

    parsec_context_start(parsec);

    return CHAMELEON_SUCCESS;
}

/**
 *  Destroy a sequence
 */
int RUNTIME_sequence_destroy( CHAM_context_t     *chamctxt,
                              RUNTIME_sequence_t *sequence )
{
    parsec_taskpool_t *parsec_dtd_tp = (parsec_taskpool_t *)(sequence->schedopt);

    assert( parsec_dtd_tp );

    /* The data must be flushed before the taskpool is released */
    if ( chameleon_parsec_flush_has_pending( parsec_dtd_tp ) ) {
        chameleon_parsec_flush_pending( parsec_dtd_tp );
        chameleon_parsec_taskpool_wait( parsec_dtd_tp );
    }
    chameleon_parsec_taskpool_release( parsec_dtd_tp );

    sequence->schedopt = NULL;

    (void)chamctxt;
    return CHAMELEON_SUCCESS;
}

/**
 *  Wait for the completion of a sequence
 */
int RUNTIME_sequence_wait( CHAM_context_t  *chamctxt,
                           RUNTIME_sequence_t *sequence )
{
    parsec_taskpool_t *parsec_dtd_tp = (parsec_taskpool_t *) sequence->schedopt;

    assert( parsec_dtd_tp );

    /* Bring back all the data used by the sequence to their owner */
    chameleon_parsec_flush_pending( parsec_dtd_tp );
    chameleon_parsec_check_stream( chamctxt, parsec_dtd_tp );
    chameleon_parsec_taskpool_wait( parsec_dtd_tp );

    return CHAMELEON_SUCCESS;
}

/**
 *  Terminate a sequence
 */
void RUNTIME_sequence_flush( CHAM_context_t  *chamctxt,
                             RUNTIME_sequence_t *sequence,
                             RUNTIME_request_t  *request,
                             int status )
{
    assert( status == CHAMELEON_SUCCESS );
    sequence->request = request;
    sequence->status = status;
    request->status = status;
    (void)chamctxt;
    return;
}
