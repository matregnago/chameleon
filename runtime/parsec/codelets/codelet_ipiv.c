/**
 *
 * @file parsec/codelet_ipiv.c
 *
 * @copyright 2023-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon Parsec codelets to convert pivot to permutations
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Matthieu Kuhn
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2025-12-19
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks.h"
#include "coreblas.h"

/*
 * The reduction of the pivot is done explicitly by the pivot tasks with
 * PaRSEC, and the buffers are reset by the first task of each step.
 */
void INSERT_TASK_ipiv_reducek( const RUNTIME_option_t *options,
                               CHAM_desc_pivot_t *pivot, int k, int h, int rank )
{
    (void)options;
    (void)pivot;
    (void)k;
    (void)h;
    (void)rank;
}

static inline int
CORE_ipiv_to_perm_parsec( parsec_execution_stream_t *context,
                          parsec_task_t             *this_task )
{
    int withidx, m0, m, k, K1, K2, mt;
    int *ipiv, *perm, *invp;

    parsec_dtd_unpack_args(
        this_task, &withidx, &m0, &m, &k, &K1, &K2, &mt, &ipiv, &perm, &invp );

    if ( withidx ) {
        int permtmp[ m ];
        int invptmp[ m ];
        CORE_ipiv_to_perm( m0, m, k, K1, K2, ipiv, permtmp, invptmp );
        CORE_perm_to_idx( m0, m, mt, permtmp, invptmp, perm, invp );
    }
    else {
        CORE_ipiv_to_perm( m0, m, k, K1, K2, ipiv, perm, invp );
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

void INSERT_TASK_ipiv_to_perm( const RUNTIME_option_t *options,
                               int m0, int m, int k, int K1, int K2, int mt,
                               const CHAM_ipiv_t *ipivdesc, int ipivk )
{
    parsec_taskpool_t* PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    int withidx = ipivdesc->withidx;

    parsec_dtd_insert_task(
        PARSEC_dtd_taskpool, CORE_ipiv_to_perm_parsec, options->priority, PARSEC_DEV_CPU, "ipiv_to_perm",
        sizeof(int),         &withidx,      PARSEC_VALUE,
        sizeof(int),         &m0,           PARSEC_VALUE,
        sizeof(int),         &m,            PARSEC_VALUE,
        sizeof(int),         &k,            PARSEC_VALUE,
        sizeof(int),         &K1,           PARSEC_VALUE,
        sizeof(int),         &K2,           PARSEC_VALUE,
        sizeof(int),         &mt,           PARSEC_VALUE,
        PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipivdesc, ChamParsecIpiv, ipivk ),
                       chameleon_parsec_ipiv_arena( ipivdesc, ChamParsecIpiv, ipivk ) | PARSEC_INPUT,
        PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipivdesc, ChamParsecPerm, ipivk ),
                       chameleon_parsec_ipiv_arena( ipivdesc, ChamParsecPerm, ipivk ) | PARSEC_OUTPUT | PARSEC_AFFINITY,
        PASSED_BY_REF, chameleon_parsec_ipiv_tile( options, ipivdesc, ChamParsecInvp, ipivk ),
                       chameleon_parsec_ipiv_arena( ipivdesc, ChamParsecInvp, ipivk ) | PARSEC_OUTPUT,
        PARSEC_DTD_ARG_END );
}
