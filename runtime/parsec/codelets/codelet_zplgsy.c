/**
 *
 * @file parsec/codelet_zplgsy.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zplgsy PaRSEC codelet
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @author Florent Pruvost
 * @date 2024-02-18
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

static inline int
CORE_zplgsy_parsec( parsec_execution_stream_t *context,
                    parsec_task_t             *this_task )
{
    CHAMELEON_Complex64_t bump;
    int m;
    int n;
    CHAMELEON_Complex64_t *A;
    int lda;
    int bigM;
    int m0;
    int n0;
    unsigned long long int seed;

    parsec_dtd_unpack_args(
        this_task, &bump, &m, &n, &A, &lda, &bigM, &m0, &n0, &seed );

    CORE_zplgsy( bump, m, n, A, lda, bigM, m0, n0, seed );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_zplgsy_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                         parsec_gpu_task_t          *gpu_task,
                         parsec_gpu_exec_stream_t   *gpu_stream )
{
    CHAMELEON_Complex64_t bump;
    int m;
    int n;
    CHAMELEON_Complex64_t *A;
    int lda;
    int bigM;
    int m0;
    int n0;
    unsigned long long int seed;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &bump, &m, &n, &A, &lda, &bigM, &m0, &n0, &seed );

    CUDA_zplgsy( (cuDoubleComplex *)&bump, m, n, (cuDoubleComplex *)A, lda, bigM, m0, n0, seed,
                 handles->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
zplgsy_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "zplgsy",
        sizeof(CHAMELEON_Complex64_t),  PARSEC_VALUE,
        sizeof(int),                    PARSEC_VALUE,
        sizeof(int),                    PARSEC_VALUE,
        PASSED_BY_REF,                  PARSEC_OUTPUT | PARSEC_AFFINITY,
        sizeof(int),                    PARSEC_VALUE,
        sizeof(int),                    PARSEC_VALUE,
        sizeof(int),                    PARSEC_VALUE,
        sizeof(int),                    PARSEC_VALUE,
        sizeof(unsigned long long int), PARSEC_VALUE,
        PARSEC_DTD_ARG_END );

#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_zplgsy_parsec_cuda );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_zplgsy_parsec );
    return tc;
}

void INSERT_TASK_zplgsy( const RUNTIME_option_t *options,
                        CHAMELEON_Complex64_t bump, cham_uplo_t uplo,
                        int m, int n, const CHAM_desc_t *A, int Am, int An,
                        int bigM, int m0, int n0, unsigned long long int seed )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_zplgsy_parsec, 1, zplgsy_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );
    int devices = CHAMELEON_PARSEC_DEVICES_OF( tileA );

    /* The output is only partially written: the GPU would push back a tile it did not read */
    if ( (m != tileA->m) || (n != tileA->n) ) {
        devices = PARSEC_DEV_CPU;
    }

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, devices,
        PARSEC_DTD_EMPTY_FLAG,                                                    &bump,
        PARSEC_DTD_EMPTY_FLAG,                                                    &m,
        PARSEC_DTD_EMPTY_FLAG,                                                    &n,
        chameleon_parsec_get_arena_index( A, Am, An ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileA->ld),
        PARSEC_DTD_EMPTY_FLAG,                                                    &bigM,
        PARSEC_DTD_EMPTY_FLAG,                                                    &m0,
        PARSEC_DTD_EMPTY_FLAG,                                                    &n0,
        PARSEC_DTD_EMPTY_FLAG,                                                    &seed,
        PARSEC_DTD_ARG_END );

    (void)uplo;
}
