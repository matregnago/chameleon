/**
 *
 * @file parsec/codelet_zgeadd.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zgeadd PaRSEC codelet
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Reazul Hoque
 * @author Florent Pruvost
 * @date 2024-02-18
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

static inline int
CORE_zgeadd_parsec( parsec_execution_stream_t *context,
                    parsec_task_t             *this_task )
{
    cham_trans_t trans;
    int M;
    int N;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t beta;
    CHAMELEON_Complex64_t *B;
    int LDB;

    parsec_dtd_unpack_args(
        this_task, &trans, &M, &N, &alpha, &A, &LDA, &beta, &B, &LDB );

    CORE_zgeadd( trans, M, N, alpha, A, LDA, beta, B, LDB );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 ******************************************************************************
 *
 * @ingroup INSERT_TASK_Complex64_t
 *
 * @brief Adds two general matrices together as in PBLAS pzgeadd.
 *
 *       B <- alpha * op(A)  + beta * B,
 *
 * where op(X) = X, X', or conj(X')
 *
 *******************************************************************************
 *
 * @param[in] trans
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          = ChamNoTrans:   op(A) = A
 *          = ChamTrans:     op(A) = A'
 *          = ChamConjTrans: op(A) = conj(A')
 *
 * @param[in] M
 *          Number of rows of the matrices op(A) and B.
 *
 * @param[in] N
 *          Number of columns of the matrices op(A) and B.
 *
 * @param[in] alpha
 *          Scalar factor of A.
 *
 * @param[in] A
 *          Matrix of size LDA-by-N, if trans = ChamNoTrans, LDA-by-M
 *          otherwise.
 *
 * @param[in] LDA
 *          Leading dimension of the array A. LDA >= max(1,k), with k=M, if
 *          trans = ChamNoTrans, and k=N otherwise.
 *
 * @param[in] beta
 *          Scalar factor of B.
 *
 * @param[in,out] B
 *          Matrix of size LDB-by-N.
 *          On exit, B = alpha * op(A) + beta * B
 *
 * @param[in] LDB
 *          Leading dimension of the array B. LDB >= max(1,M)
 *
 *******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 * @retval <0 if -i, the i-th argument had an illegal value
 *
 */
#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_zgeadd_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                         parsec_gpu_task_t          *gpu_task,
                         parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_trans_t trans;
    int M;
    int N;
    cuDoubleComplex alpha; /* 16 bytes aligned: read by cuBLAS with aligned loads */
    CHAMELEON_Complex64_t *A;
    int LDA;
    cuDoubleComplex beta;
    CHAMELEON_Complex64_t *B;
    int LDB;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &trans, &M, &N, &alpha, &A, &LDA, &beta, &B, &LDB );

    CUDA_zgeadd( trans, M, N,
                 (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A, LDA,
                 (cuDoubleComplex *)&beta,  (cuDoubleComplex *)B, LDB,
                 handles->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

#if defined(CHAMELEON_PARSEC_HIP)
static int
CORE_zgeadd_parsec_hip( parsec_device_gpu_module_t *gpu_device,
                        parsec_gpu_task_t          *gpu_task,
                        parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_trans_t trans;
    int M;
    int N;
    hipDoubleComplex alpha; /* 16 bytes aligned: read by hipBLAS with aligned loads */
    CHAMELEON_Complex64_t *A;
    int LDA;
    hipDoubleComplex beta;
    CHAMELEON_Complex64_t *B;
    int LDB;
    chameleon_parsec_hip_handles_t *handles = chameleon_parsec_hip_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &trans, &M, &N, &alpha, &A, &LDA, &beta, &B, &LDB );

    HIP_zgeadd( trans, M, N,
                (hipDoubleComplex *)&alpha, (hipDoubleComplex *)A, LDA,
                (hipDoubleComplex *)&beta,  (hipDoubleComplex *)B, LDB,
                handles->hipblas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
zgeadd_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "geadd",
        sizeof(cham_trans_t),          PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(CHAMELEON_Complex64_t), PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INPUT,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(CHAMELEON_Complex64_t), PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),                   PARSEC_VALUE,
        PARSEC_DTD_ARG_END );

#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_zgeadd_parsec_cuda );
#endif
#if defined(CHAMELEON_PARSEC_HIP)
    chameleon_parsec_add_hip_chore( tp, tc, CORE_zgeadd_parsec_hip );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_zgeadd_parsec );
    return tc;
}

void INSERT_TASK_zgeadd( const RUNTIME_option_t *options,
                         cham_trans_t trans, int m, int n, int nb,
                         CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                         CHAMELEON_Complex64_t beta,  const CHAM_desc_t *B, int Bm, int Bn )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_zgeadd_parsec, 2, zgeadd_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );
    CHAM_tile_t *tileB = B->get_blktile( B, Bm, Bn );

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, CHAMELEON_PARSEC_DEVICES_OF( tileA, tileB ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &trans,
        PARSEC_DTD_EMPTY_FLAG,                                                    &m,
        PARSEC_DTD_EMPTY_FLAG,                                                    &n,
        PARSEC_DTD_EMPTY_FLAG,                                                    &alpha,
        chameleon_parsec_get_arena_index( A, Am, An ),                            RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileA->ld),
        PARSEC_DTD_EMPTY_FLAG,                                                    &beta,
        chameleon_parsec_get_arena_index( B, Bm, Bn ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( B, CHAMELEON_Complex64_t, Bm, Bn ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileB->ld),
        PARSEC_DTD_ARG_END );

    (void)nb;
}
