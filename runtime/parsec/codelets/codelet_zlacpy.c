/**
 *
 * @file parsec/codelet_zlacpy.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zlacpy PaRSEC codelet
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @author Florent Pruvost
 * @date 2024-02-18
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

static inline int
CORE_zlacpy_parsec( parsec_execution_stream_t *context,
                    parsec_task_t             *this_task )
{
    cham_uplo_t uplo;
    int M;
    int N;
    const CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t *B;
    int LDB;

    parsec_dtd_unpack_args(
        this_task, &uplo, &M, &N, &A, &LDA, &B, &LDB );

    CORE_zlacpy( uplo, M, N, A, LDA, B, LDB );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_zlacpy_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                         parsec_gpu_task_t          *gpu_task,
                         parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_uplo_t uplo;
    int M;
    int N;
    const CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t *B;
    int LDB;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &uplo, &M, &N, &A, &LDA, &B, &LDB );

    CUDA_zlacpy( uplo, M, N,
                 (const cuDoubleComplex *)A, LDA,
                 (cuDoubleComplex *)B,       LDB,
                 handles->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
zlacpy_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "lacpy",
        sizeof(cham_uplo_t), PARSEC_VALUE,
        sizeof(int),         PARSEC_VALUE,
        sizeof(int),         PARSEC_VALUE,
        PASSED_BY_REF,       PARSEC_INPUT,
        sizeof(int),         PARSEC_VALUE,
        PASSED_BY_REF,       PARSEC_OUTPUT | PARSEC_AFFINITY,
        sizeof(int),         PARSEC_VALUE,
        PARSEC_DTD_ARG_END );

#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_zlacpy_parsec_cuda );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_zlacpy_parsec );
    return tc;
}

void INSERT_TASK_zlacpy( const RUNTIME_option_t *options,
                         cham_uplo_t uplo, int m, int n,
                         const CHAM_desc_t *A, int Am, int An,
                         const CHAM_desc_t *B, int Bm, int Bn )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_zlacpy_parsec, 2, zlacpy_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );
    CHAM_tile_t *tileB = B->get_blktile( B, Bm, Bn );
    int devices = CHAMELEON_PARSEC_DEVICES_OF( tileA, tileB );

    /* The output is only partially written: the GPU would push back a tile it did not read */
    if ( (uplo != ChamUpperLower) || (m != tileB->m) || (n != tileB->n) ) {
        devices = PARSEC_DEV_CPU;
    }

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, devices,
        PARSEC_DTD_EMPTY_FLAG,                                                    &uplo,
        PARSEC_DTD_EMPTY_FLAG,                                                    &m,
        PARSEC_DTD_EMPTY_FLAG,                                                    &n,
        chameleon_parsec_get_arena_index( A, Am, An ),                            RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileA->ld),
        chameleon_parsec_get_arena_index( B, Bm, Bn ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( B, CHAMELEON_Complex64_t, Bm, Bn ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileB->ld),
        PARSEC_DTD_ARG_END );
}

void INSERT_TASK_zlacpy_panel( const RUNTIME_option_t *options,
                               cham_uplo_t uplo, int k,
                               const CHAM_desc_t *A, int Am, int An,
                               const CHAM_desc_t *B, int Bm, int Bn )
{
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );

    (void)k;
    INSERT_TASK_zlacpy( options, uplo, tileA->m, tileA->n,
                        A, Am, An, B, Bm, Bn );
}

static inline int
CORE_zlacpyx_parsec( parsec_execution_stream_t *context,
                    parsec_task_t             *this_task )
{
    cham_uplo_t uplo;
    int M;
    int N;
    int displA;
    const CHAMELEON_Complex64_t *A;
    int LDA;
    int displB;
    CHAMELEON_Complex64_t *B;
    int LDB;

    parsec_dtd_unpack_args(
        this_task, &uplo, &M, &N, &displA, &A, &LDA, &displB, &B, &LDB );

    CORE_zlacpy( uplo, M, N, A + displA, LDA, B + displB, LDB );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

void INSERT_TASK_zlacpyx( const RUNTIME_option_t *options,
                          cham_uplo_t uplo, int m, int n,
                          int displA, const CHAM_desc_t *A, int Am, int An, int lda,
                          int displB, const CHAM_desc_t *B, int Bm, int Bn, int ldb )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_zlacpyx_parsec, options->priority, PARSEC_DEV_CPU, "lacpyx",
        sizeof(cham_uplo_t), &uplo,   PARSEC_VALUE,
        sizeof(int),         &m,      PARSEC_VALUE,
        sizeof(int),         &n,      PARSEC_VALUE,
        sizeof(int),         &displA, PARSEC_VALUE,
        PASSED_BY_REF,       RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ), chameleon_parsec_get_arena_index( A, Am, An ) | PARSEC_INPUT,
        sizeof(int),         &lda,    PARSEC_VALUE,
        sizeof(int),         &displB, PARSEC_VALUE,
        PASSED_BY_REF,       RTBLKADDR( B, CHAMELEON_Complex64_t, Bm, Bn ), chameleon_parsec_get_arena_index( B, Bm, Bn ) | PARSEC_OUTPUT | PARSEC_AFFINITY,
        sizeof(int),         &ldb,    PARSEC_VALUE,
        PARSEC_DTD_ARG_END );
}
