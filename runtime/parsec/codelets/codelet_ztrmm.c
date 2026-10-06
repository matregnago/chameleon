/**
 *
 * @file parsec/codelet_ztrmm.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon ztrmm PaRSEC codelet
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @date 2024-02-18
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

static inline int
CORE_ztrmm_parsec( parsec_execution_stream_t *context,
                   parsec_task_t             *this_task )
{
    cham_side_t side;
    cham_uplo_t uplo;
    cham_trans_t transA;
    cham_diag_t diag;
    int M;
    int N;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t *B;
    int LDB;

    parsec_dtd_unpack_args(
        this_task, &side, &uplo, &transA, &diag, &M, &N, &alpha, &A, &LDA, &B, &LDB );

    CORE_ztrmm( side, uplo,
        transA, diag,
        M, N,
        alpha, A, LDA,
        B, LDB);

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_ztrmm_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                        parsec_gpu_task_t          *gpu_task,
                        parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_side_t side;
    cham_uplo_t uplo;
    cham_trans_t transA;
    cham_diag_t diag;
    int M;
    int N;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t *B;
    int LDB;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &side, &uplo, &transA, &diag, &M, &N, &alpha, &A, &LDA, &B, &LDB );

    CUDA_ztrmm( side, uplo, transA, diag, M, N,
                (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A, LDA,
                                           (cuDoubleComplex *)B, LDB,
                handles->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
ztrmm_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "trmm",
        sizeof(cham_side_t),           PARSEC_VALUE,
        sizeof(cham_uplo_t),           PARSEC_VALUE,
        sizeof(cham_trans_t),          PARSEC_VALUE,
        sizeof(cham_diag_t),           PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(CHAMELEON_Complex64_t), PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INPUT,
        sizeof(int),                   PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),                   PARSEC_VALUE,
        PARSEC_DTD_ARG_END );

#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_ztrmm_parsec_cuda );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_ztrmm_parsec );
    return tc;
}

void INSERT_TASK_ztrmm(const RUNTIME_option_t *options,
                      cham_side_t side, cham_uplo_t uplo, cham_trans_t transA, cham_diag_t diag,
                      int m, int n, int nb,
                      CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                      const CHAM_desc_t *B, int Bm, int Bn)
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_ztrmm_parsec, 2, ztrmm_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );
    CHAM_tile_t *tileB = B->get_blktile( B, Bm, Bn );

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, CHAMELEON_PARSEC_DEVICES_OF( tileA, tileB ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &side,
        PARSEC_DTD_EMPTY_FLAG,                                                    &uplo,
        PARSEC_DTD_EMPTY_FLAG,                                                    &transA,
        PARSEC_DTD_EMPTY_FLAG,                                                    &diag,
        PARSEC_DTD_EMPTY_FLAG,                                                    &m,
        PARSEC_DTD_EMPTY_FLAG,                                                    &n,
        PARSEC_DTD_EMPTY_FLAG,                                                    &alpha,
        chameleon_parsec_get_arena_index( A, Am, An ),                            RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileA->ld),
        chameleon_parsec_get_arena_index( B, Bm, Bn ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( B, CHAMELEON_Complex64_t, Bm, Bn ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileB->ld),
        PARSEC_DTD_ARG_END );

    (void)nb;
}
