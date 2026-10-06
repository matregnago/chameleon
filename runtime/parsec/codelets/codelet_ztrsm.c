/**
 *
 * @file parsec/codelet_ztrsm.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon ztrsm PaRSEC codelet
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
CORE_ztrsm_parsec( parsec_execution_stream_t *context,
                    parsec_task_t             *this_task )
{
    cham_side_t side;
    cham_uplo_t uplo;
    cham_trans_t trans;
    cham_diag_t diag;
    int tempmm, nb, ldak, ldam;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *T;
    CHAMELEON_Complex64_t *C;

    parsec_dtd_unpack_args(
        this_task, &side, &uplo, &trans, &diag, &tempmm, &nb, &alpha, &T, &ldak, &C, &ldam );

    CORE_ztrsm( side, uplo, trans, diag,
                tempmm, nb, alpha,
                T, ldak, C, ldam );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_ztrsm_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                        parsec_gpu_task_t          *gpu_task,
                        parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_side_t side;
    cham_uplo_t uplo;
    cham_trans_t trans;
    cham_diag_t diag;
    int tempmm, nb, ldak, ldam;
    cuDoubleComplex alpha; /* 16 bytes aligned: read by cuBLAS with aligned loads */
    CHAMELEON_Complex64_t *T;
    CHAMELEON_Complex64_t *C;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &side, &uplo, &trans, &diag, &tempmm, &nb, &alpha, &T, &ldak, &C, &ldam );

    CUDA_ztrsm( side, uplo, trans, diag, tempmm, nb,
                (cuDoubleComplex *)&alpha, (cuDoubleComplex *)T, ldak,
                                           (cuDoubleComplex *)C, ldam,
                handles->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

#if defined(CHAMELEON_PARSEC_HIP)
static int
CORE_ztrsm_parsec_hip( parsec_device_gpu_module_t *gpu_device,
                       parsec_gpu_task_t          *gpu_task,
                       parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_side_t side;
    cham_uplo_t uplo;
    cham_trans_t trans;
    cham_diag_t diag;
    int tempmm, nb, ldak, ldam;
    hipDoubleComplex alpha; /* 16 bytes aligned: read by hipBLAS with aligned loads */
    CHAMELEON_Complex64_t *T;
    CHAMELEON_Complex64_t *C;
    chameleon_parsec_hip_handles_t *handles = chameleon_parsec_hip_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &side, &uplo, &trans, &diag, &tempmm, &nb, &alpha, &T, &ldak, &C, &ldam );

    HIP_ztrsm( side, uplo, trans, diag, tempmm, nb,
               (hipDoubleComplex *)&alpha, (hipDoubleComplex *)T, ldak,
                                           (hipDoubleComplex *)C, ldam,
               handles->hipblas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
ztrsm_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "Trsm",
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
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_ztrsm_parsec_cuda );
#endif
#if defined(CHAMELEON_PARSEC_HIP)
    chameleon_parsec_add_hip_chore( tp, tc, CORE_ztrsm_parsec_hip );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_ztrsm_parsec );
    return tc;
}

void INSERT_TASK_ztrsm(const RUNTIME_option_t *options,
                      cham_side_t side, cham_uplo_t uplo, cham_trans_t transA, cham_diag_t diag,
                      int m, int n, int nb,
                      CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                      const CHAM_desc_t *B, int Bm, int Bn)
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_ztrsm_parsec, 2, ztrsm_task_class );
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
