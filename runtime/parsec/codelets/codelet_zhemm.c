/**
 *
 * @file parsec/codelet_zhemm.c
 *
 * @copyright 2009-2015 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zhemm PaRSEC codelet
 *
 * @version 1.4.0
 * @author Reazul Hoque
 * @author Mathieu Faverge
 * @date 2024-02-18
 * @precisions normal z -> c
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "coreblas/coreblas_z.h"

static inline int
CORE_zhemm_parsec( parsec_execution_stream_t *context,
                   parsec_task_t             *this_task )
{
    cham_side_t side;
    cham_uplo_t uplo;
    int M;
    int N;
    CHAMELEON_Complex64_t alpha;
    CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t *B;
    int LDB;
    CHAMELEON_Complex64_t beta;
    CHAMELEON_Complex64_t *C;
    int LDC;

    parsec_dtd_unpack_args(
        this_task, &side, &uplo, &M, &N, &alpha, &A, &LDA, &B, &LDB, &beta, &C, &LDC );

    CORE_zhemm( side, uplo, M, N,
                alpha, A, LDA,
                       B, LDB,
                beta,  C, LDC );

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(CHAMELEON_PARSEC_CUDA)
static int
CORE_zhemm_parsec_cuda( parsec_device_gpu_module_t *gpu_device,
                        parsec_gpu_task_t          *gpu_task,
                        parsec_gpu_exec_stream_t   *gpu_stream )
{
    cham_side_t side;
    cham_uplo_t uplo;
    int M;
    int N;
    cuDoubleComplex alpha; /* 16 bytes aligned: read by cuBLAS with aligned loads */
    CHAMELEON_Complex64_t *A;
    int LDA;
    CHAMELEON_Complex64_t *B;
    int LDB;
    cuDoubleComplex beta;
    CHAMELEON_Complex64_t *C;
    int LDC;
    chameleon_parsec_cuda_handles_t *handles = chameleon_parsec_cuda_handles( gpu_stream );

    parsec_dtd_unpack_args(
        gpu_task->ec, &side, &uplo, &M, &N, &alpha, &A, &LDA, &B, &LDB, &beta, &C, &LDC );

    CUDA_zhemm( side, uplo, M, N,
                (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A, LDA,
                                           (cuDoubleComplex *)B, LDB,
                (cuDoubleComplex *)&beta,  (cuDoubleComplex *)C, LDC,
                handles->cublas );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

static parsec_task_class_t *
zhemm_task_class( parsec_taskpool_t *tp )
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(
        tp, "hemm",
        sizeof(cham_side_t),           PARSEC_VALUE,
        sizeof(cham_uplo_t),           PARSEC_VALUE,
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
    chameleon_parsec_add_cuda_chore( tp, tc, CORE_zhemm_parsec_cuda );
#endif
    parsec_dtd_task_class_add_chore( tp, tc, PARSEC_DEV_CPU, (void *)CORE_zhemm_parsec );
    return tc;
}

void
INSERT_TASK_zhemm( const RUNTIME_option_t *options,
                   cham_side_t side, cham_uplo_t uplo,
                   int m, int n, int nb,
                   CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                                                const CHAM_desc_t *B, int Bm, int Bn,
                   CHAMELEON_Complex64_t beta,  const CHAM_desc_t *C, int Cm, int Cn )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    parsec_task_class_t *tc = chameleon_parsec_task_class( options, CORE_zhemm_parsec, 3, zhemm_task_class );
    CHAM_tile_t *tileA = A->get_blktile( A, Am, An );
    CHAM_tile_t *tileB = B->get_blktile( B, Bm, Bn );
    CHAM_tile_t *tileC = C->get_blktile( C, Cm, Cn );

    parsec_dtd_insert_task_with_task_class(
        PARSEC_dtd_taskpool, tc, options->priority, CHAMELEON_PARSEC_DEVICES_OF( tileA, tileB, tileC ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &side,
        PARSEC_DTD_EMPTY_FLAG,                                                    &uplo,
        PARSEC_DTD_EMPTY_FLAG,                                                    &m,
        PARSEC_DTD_EMPTY_FLAG,                                                    &n,
        PARSEC_DTD_EMPTY_FLAG,                                                    &alpha,
        chameleon_parsec_get_arena_index( A, Am, An ),                            RTBLKADDR( A, CHAMELEON_Complex64_t, Am, An ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileA->ld),
        chameleon_parsec_get_arena_index( B, Bm, Bn ),                            RTBLKADDR( B, CHAMELEON_Complex64_t, Bm, Bn ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileB->ld),
        PARSEC_DTD_EMPTY_FLAG,                                                    &beta,
        chameleon_parsec_get_arena_index( C, Cm, Cn ) | CHAMELEON_PARSEC_GPU_OUT, RTBLKADDR( C, CHAMELEON_Complex64_t, Cm, Cn ),
        PARSEC_DTD_EMPTY_FLAG,                                                    &(tileC->ld),
        PARSEC_DTD_ARG_END );

    (void)nb;
}

void
INSERT_TASK_zhemm_Astat( const RUNTIME_option_t *options,
                         cham_side_t side, cham_uplo_t uplo,
                         int m, int n, int nb,
                         CHAMELEON_Complex64_t alpha, const CHAM_desc_t *A, int Am, int An,
                                                      const CHAM_desc_t *B, int Bm, int Bn,
                         CHAMELEON_Complex64_t beta,  const CHAM_desc_t *C, int Cm, int Cn )
{
    INSERT_TASK_zhemm( options, side, uplo, m, n, nb,
                       alpha, A, Am, An, B, Bm, Bn,
                       beta, C, Cm, Cn );
}
