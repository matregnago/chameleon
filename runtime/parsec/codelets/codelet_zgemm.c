/**
 *
 * @file parsec/codelet_zgemm.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zgemm PaRSEC codelet
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

static inline int
CORE_zgemm_parsec( parsec_execution_stream_t *context,
                   parsec_task_t             *this_task )
{
    cham_trans_t transA;
    cham_trans_t transB;
    int m;
    int n;
    int k;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *A;
    int lda;
    CHAMELEON_Complex64_t *B;
    int ldb;
    CHAMELEON_Complex64_t beta;
    CHAMELEON_Complex64_t *C;
    int ldc;

    parsec_dtd_unpack_args(
        this_task, &transA, &transB, &m, &n, &k, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc );

    CORE_zgemm( transA, transB, m, n, k,
                alpha, A, lda,
                       B, ldb,
                beta,  C, ldc );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_zgemm_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                        parsec_gpu_task_t          *gpu_task,
                        parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_trans_t transA;
    cham_trans_t transB;
    int m;
    int n;
    int k;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *A;
    int lda;
    CHAMELEON_Complex64_t *B;
    int ldb;
    CHAMELEON_Complex64_t beta;
    CHAMELEON_Complex64_t *C;
    int ldc;

    parsec_dtd_unpack_args(
        gpu_task->ec, &transA, &transB, &m, &n, &k, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc );

    CUDA_zgemm( transA, transB, m, n, k,
                (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A, lda,
                                           (cuDoubleComplex *)B, ldb,
                (cuDoubleComplex *)&beta,  (cuDoubleComplex *)C, ldc,
                chameleon_parsec_cuda_handles( gpu_stream )->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
zgemm_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "gemm",
        sizeof(cham_trans_t),          PARSEC_VALUE,
        sizeof(cham_trans_t),          PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(CHAMELEON_Complex64_t), PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INPUT,
        sizeof(int),                   PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INPUT,
        sizeof(int),                   PARSEC_VALUE,
        sizeof(CHAMELEON_Complex64_t), PARSEC_VALUE,
        PASSED_BY_REF,                 PARSEC_INOUT | PARSEC_AFFINITY,
        sizeof(int),                   PARSEC_VALUE,
        PARSEC_DTD_ARG_END );

#if defined(CHAMELEON_PARSEC_CUDA)
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_zgemm_parsec_cuda );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_zgemm_parsec );
    return tc;
}

void
INSERT_TASK_zgemm( const RUNTIME_option_t *options,
                   cham_trans_t transA, cham_trans_t transB,
                   int m, int n, int k, int nb,
                   CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                                                const CHAM_desc_t *B, int Bm, int Bn,
                   CHAMELEON_Complex64_t beta,  const CHAM_desc_t *C, int Cm, int Cn )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_zgemm_parsec, 3, zgemm_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );
    CHAM_tile_t *tileB = B->get_blktile( B, Bm, Bn );
    CHAM_tile_t *tileC = C->get_blktile( C, Cm, Cn );
    int devices = CHAMELEON_PARSEC_DEVICES_OF( tileA, tileB, tileC );

    /* WARNING: CUDA 12.3 has an issue when m or n or k=1 in double complex,
       thus we disable gemm on gpu in these cases */
#if defined(PRECISION_z)
    if ( (k == 1) || (n == 1) || (m == 1) ) {
        devices = PARSEC_DEV_CPU;
    }
#endif

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, devices,
        PARSEC_DTD_EMPTY_FLAG, &transA,
        PARSEC_DTD_EMPTY_FLAG, &transB,
        PARSEC_DTD_EMPTY_FLAG, &m,
        PARSEC_DTD_EMPTY_FLAG, &n,
        PARSEC_DTD_EMPTY_FLAG, &k,
        PARSEC_DTD_EMPTY_FLAG, &alpha,
        chameleon_parsec_get_arena_index( A, Am, An ), RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG, &(tileA->ld),
        chameleon_parsec_get_arena_index( B, Bm, Bn ), RTBLKADDR( B, CHAMELEON_Complex64_t, Bm, Bn ),
        PARSEC_DTD_EMPTY_FLAG, &(tileB->ld),
        PARSEC_DTD_EMPTY_FLAG, &beta,
        chameleon_parsec_get_arena_index( C, Cm, Cn ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( C, CHAMELEON_Complex64_t, Cm, Cn ),
        PARSEC_DTD_EMPTY_FLAG, &(tileC->ld),
        PARSEC_DTD_ARG_END );

    (void)nb;
}

void
INSERT_TASK_zgemm_Astat( const RUNTIME_option_t *options,
                         cham_trans_t transA, cham_trans_t transB,
                         int m, int n, int k, int nb,
                         CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                                                      const CHAM_desc_t *B, int Bm, int Bn,
                         CHAMELEON_Complex64_t beta,  const CHAM_desc_t *C, int Cm, int Cn )
{
    INSERT_TASK_zgemm( options, transA, transB, m, n, k, nb,
                       alpha, A, Am, An, B, Bm, Bn,
                       beta, C, Cm, Cn );
}
