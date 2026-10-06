/**
 *
 * @file parsec/codelet_zpotrf.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zpotrf PaRSEC codelet
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Florent Pruvost
 * @author Mathieu Faverge
 * @date 2024-02-18
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

/**
 *
 * @ingroup INSERT_TASK_Complex64_t
 *
 */
static inline int
CORE_zpotrf_parsec( parsec_execution_stream_t *context,
                    parsec_task_t             *this_task )
{
    cham_uplo_t uplo;
    int tempkm, ldak, iinfo, info;
    RUNTIME_sequence_t *sequence;
    RUNTIME_request_t *request;
    CHAMELEON_Complex64_t *A;

    parsec_dtd_unpack_args(
        this_task, &uplo, &tempkm, &A, &ldak, &iinfo, &sequence, &request );

    CORE_zpotrf( uplo, tempkm, A, ldak, &info );

    if ( (sequence->status == CHAMELEON_SUCCESS) && (info != 0) ) {
        RUNTIME_sequence_flush( NULL, sequence, request, iinfo+info );
    }

    (void)context;
    (void)info;
    (void)iinfo;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
/**
 * The info of the kernel is read once the stream reached the end of the task.
 */
static int
CORE_zpotrf_parsec_cuda_complete( parsec_device_gpu_module_t *gpu_device,
                                  parsec_gpu_task_t         **gpu_task,
                                  parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_uplo_t uplo;
    int tempkm, ldak, iinfo;
    RUNTIME_sequence_t *sequence;
    RUNTIME_request_t *request;
    CHAMELEON_Complex64_t *A;
    int **hinfo;

    parsec_dtd_unpack_args(
        (*gpu_task)->ec, &uplo, &tempkm, &A, &ldak, &iinfo, &sequence, &request, &hinfo );

    if ( (sequence->status == CHAMELEON_SUCCESS) && (**hinfo != 0) ) {
        RUNTIME_sequence_flush( NULL, sequence, request, iinfo + **hinfo );
    }

    /* Otherwise it is called again at the end of the next stages of the task */
    (*gpu_task)->complete_stage = NULL;

    (void)gpu_device;
    (void)gpu_stream;
    return PARSEC_HOOK_RETURN_DONE;
}

static int
CORE_zpotrf_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                         parsec_gpu_task_t          *gpu_task,
                         parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_uplo_t uplo;
    int tempkm, ldak, iinfo, lwork;
    RUNTIME_sequence_t *sequence;
    RUNTIME_request_t *request;
    CHAMELEON_Complex64_t *A;
    int **hinfo;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );
    chameleon_parsec_cuda_ws_t      *ws      = chameleon_parsec_cuda_ws( gpu_stream );
    cuDoubleComplex                 *work;

    parsec_dtd_unpack_args(
        gpu_task->ec, &uplo, &tempkm, &A, &ldak, &iinfo, &sequence, &request, &hinfo );

    cusolverDnZpotrf_bufferSize( handles->cusolverDn, chameleon_cublas_const(uplo),
                                 tempkm, (cuDoubleComplex *)A, ldak, &lwork );
    work = chameleon_parsec_cuda_ws_work( ws, gpu_stream, sizeof(cuDoubleComplex) * (size_t)lwork );
    if ( work == NULL ) {
        /* Retry once the device memory has been released */
        return PARSEC_HOOK_RETURN_AGAIN;
    }

    CUDA_zpotrf( uplo, tempkm, (cuDoubleComplex *)A, ldak, work, lwork, ws->dinfo,
                 handles->cusolverDn );

    *hinfo = ws->hinfo + (ws->next++ % CHAMELEON_PARSEC_CUDA_NINFO);
    cudaMemcpyAsync( *hinfo, ws->dinfo, sizeof(int), cudaMemcpyDeviceToHost,
                     chameleon_parsec_cuda_stream( gpu_stream ) );
    gpu_task->complete_stage = CORE_zpotrf_parsec_cuda_complete;

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

#if defined(CHAMELEON_PARSEC_HIP)
/**
 * The info of the kernel is read once the stream reached the end of the task.
 */
static int
CORE_zpotrf_parsec_hip_complete( parsec_device_gpu_module_t *gpu_device,
                                 parsec_gpu_task_t         **gpu_task,
                                 parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_uplo_t uplo;
    int tempkm, ldak, iinfo;
    RUNTIME_sequence_t *sequence;
    RUNTIME_request_t *request;
    CHAMELEON_Complex64_t *A;
    int **hinfo;

    parsec_dtd_unpack_args(
        (*gpu_task)->ec, &uplo, &tempkm, &A, &ldak, &iinfo, &sequence, &request, &hinfo );

    if ( (sequence->status == CHAMELEON_SUCCESS) && (**hinfo != 0) ) {
        RUNTIME_sequence_flush( NULL, sequence, request, iinfo + **hinfo );
    }

    /* Otherwise it is called again at the end of the next stages of the task */
    (*gpu_task)->complete_stage = NULL;

    (void)gpu_device;
    (void)gpu_stream;
    return PARSEC_HOOK_RETURN_DONE;
}

static int
CORE_zpotrf_parsec_hip( parsec_device_gpu_module_t *gpu_device,
                        parsec_gpu_task_t          *gpu_task,
                        parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_uplo_t uplo;
    int tempkm, ldak, iinfo, lwork;
    RUNTIME_sequence_t *sequence;
    RUNTIME_request_t *request;
    CHAMELEON_Complex64_t *A;
    int **hinfo;
    chameleon_parsec_hip_handles_t *handles = chameleon_parsec_hip_handles( gpu_stream );
    chameleon_parsec_hip_ws_t      *ws      = chameleon_parsec_hip_ws( gpu_stream );
    hipDoubleComplex               *work;

    parsec_dtd_unpack_args(
        gpu_task->ec, &uplo, &tempkm, &A, &ldak, &iinfo, &sequence, &request, &hinfo );

    hipsolverDnZpotrf_bufferSize( handles->hipsolverDn, chameleon_hipblas_const(uplo),
                                  tempkm, (hipDoubleComplex *)A, ldak, &lwork );
    work = chameleon_parsec_hip_ws_work( ws, gpu_stream, sizeof(hipDoubleComplex) * (size_t)lwork );
    if ( work == NULL ) {
        /* Retry once the device memory has been released */
        return PARSEC_HOOK_RETURN_AGAIN;
    }

    HIP_zpotrf( uplo, tempkm, (hipDoubleComplex *)A, ldak, work, lwork, ws->dinfo,
                handles->hipsolverDn );

    *hinfo = ws->hinfo + (ws->next++ % CHAMELEON_PARSEC_HIP_NINFO);
    hipMemcpyAsync( *hinfo, ws->dinfo, sizeof(int), hipMemcpyDeviceToHost,
                    chameleon_parsec_hip_stream( gpu_stream ) );
    gpu_task->complete_stage = CORE_zpotrf_parsec_hip_complete;

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
zpotrf_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "potrf",
        sizeof(cham_uplo_t),         PARSEC_VALUE,
        sizeof(int),                 PARSEC_VALUE,
        PASSED_BY_REF,               PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),                 PARSEC_VALUE,
        sizeof(int),                 PARSEC_VALUE,
        sizeof(RUNTIME_sequence_t*), PARSEC_VALUE,
        sizeof(RUNTIME_request_t*),  PARSEC_VALUE,
        sizeof(int *),               PARSEC_SCRATCH, /* Host info slot of the CUDA kernel */
        PARSEC_DTD_ARG_END );

#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_zpotrf_parsec_cuda );
#endif
#if defined(CHAMELEON_PARSEC_HIP)
    chameleon_parsec_add_hip_chore( tp, tc, CORE_zpotrf_parsec_hip );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_zpotrf_parsec );
    return tc;
}

void INSERT_TASK_zpotrf(const RUNTIME_option_t *options,
                       cham_uplo_t uplo, int n, int nb,
                       const CHAM_desc_t *A, int Am, int An,
                       int iinfo)
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_zpotrf_parsec, 1, zpotrf_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, CHAMELEON_PARSEC_DEVICES_OF( tileA ),
        PARSEC_DTD_EMPTY_FLAG, &uplo,
        PARSEC_DTD_EMPTY_FLAG, &n,
        chameleon_parsec_get_arena_index( A, Am, An ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG, &(tileA->ld),
        PARSEC_DTD_EMPTY_FLAG, &iinfo,
        PARSEC_DTD_EMPTY_FLAG, &(options->sequence),
        PARSEC_DTD_EMPTY_FLAG, &(options->request),
        PARSEC_DTD_EMPTY_FLAG, NULL,
        PARSEC_DTD_ARG_END );

    (void)nb;
}
