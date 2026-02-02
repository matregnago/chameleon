/**
 *
 * @file starpu/codelet_ipiv.c
 *
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon StarPU codelets to work with ipiv array
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Matthieu Kuhn
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2025-12-19
 *
 */
#include "chameleon_starpu_internal.h"

void INSERT_TASK_ipiv_reducek( const RUNTIME_option_t *options,
                               CHAM_desc_pivot_t *pivot, int k, int h, int rank )
{
    starpu_data_handle_t prevpiv = RUNTIME_pivot_getaddr( pivot, rank, h-1 );

#if defined(HAVE_STARPU_MPI_REDUX) && defined(CHAMELEON_USE_MPI)
#if !defined(HAVE_STARPU_MPI_REDUX_WRAPUP)
    starpu_data_handle_t nextpiv = RUNTIME_pivot_getaddr( pivot, rank, h );
    if ( h < pivot->n ) {
        starpu_mpi_redux_data_prio_tree( options->sequence->comm, nextpiv,
                                         options->priority, 2 /* Binary tree */ );
    }
#endif
#endif

    /* Invalidate the previous pivot structure for correct initialization in later reuse */
    if ( h > 0 ) {
        starpu_data_invalidate_submit( prevpiv );
    }

    (void)options;
}

#if !defined(CHAMELEON_SIMULATION)
static void cl_ipiv_to_perm_cpu_func( void *descr[], void *cl_arg )
{
    int  m0, m, k, K1, K2;
    int *ipiv, *perm, *invp;

    starpu_codelet_unpack_args( cl_arg, &m0, &m, &k, &K1, &K2 );

    ipiv = (int*)STARPU_VECTOR_GET_PTR(descr[0]);
    perm = (int*)STARPU_VECTOR_GET_PTR(descr[1]);
    invp = (int*)STARPU_VECTOR_GET_PTR(descr[2]);

    CORE_ipiv_to_perm( m0, m, k, K1, K2, ipiv, perm, invp );
}
#endif /* !defined(CHAMELEON_SIMULATION) */

/*
* Codelet definition
*/
static struct starpu_codelet cl_ipiv_to_perm = {
    .where        = STARPU_CPU,
#if defined(CHAMELEON_SIMULATION)
    .cpu_funcs[0] = (starpu_cpu_func_t)1,
#else
    .cpu_funcs[0] = cl_ipiv_to_perm_cpu_func,
#endif
    .nbuffers     = 3,
    .model        = NULL,
    .name         = "ipiv_to_perm"
};

void INSERT_TASK_ipiv_to_perm( const RUNTIME_option_t *options,
                               int m0, int m, int k, int K1, int K2,
                               const CHAM_ipiv_t *ipivdesc, int ipivk )
{

    rt_starpu_insert_task(
        &cl_ipiv_to_perm,
        STARPU_VALUE,             &m0,  sizeof(int),
        STARPU_VALUE,             &m,   sizeof(int),
        STARPU_VALUE,             &k,   sizeof(int),
        STARPU_VALUE,             &K1,  sizeof(int),
        STARPU_VALUE,             &K2,  sizeof(int),
        STARPU_R,                 RUNTIME_ipiv_getaddr( ipivdesc, ipivk ),
        STARPU_W,                 RUNTIME_ipiv_getperm( ipivdesc, ipivk ),
        STARPU_W,                 RUNTIME_ipiv_getinvp( ipivdesc, ipivk ),

        /* Common task arguments */
        INSERT_TASK_COMMON_TASK_PARAMS_NOCB,
        0 );
}

