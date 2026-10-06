/**
 *
 * @file parsec/codelet_zperm_allreduce.c
 *
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon parsec codelets to do the reduction
 *
 * @version 1.4.0
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2026-10-05
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"
#include "control/compute_z.h"

void
chameleon_parsec_zperm_reduce_submit( const RUNTIME_option_t *options,
                                      cham_dir_t              dir,
                                      const CHAM_desc_t      *A,
                                      int                     Am,
                                      int                     An,
                                      CHAM_ipiv_t            *ipiv,
                                      int                     ipivk,
                                      const CHAM_desc_t      *Wu,
                                      int                     Wum,
                                      int                     Wun,
                                      CHAM_perm_t            *ws,
                                      int                     Wm,
                                      int                     Wn,
                                      const CHAM_reduce_t    *reduce,
                                      int                     allreduce );

/*
 * The data are sent by the DTD runtime to the ranks executing the tasks that
 * need them: the explicit sends of StarPU are not needed.
 */
void
INSERT_TASK_zperm_allreduce_send_A( const RUNTIME_option_t *options,
                                    CHAM_desc_t            *A,
                                    int                     Am,
                                    int                     An,
                                    int                     myrank,
                                    int                     np,
                                    int                    *proc_involved )
{
    (void)options;
    (void)A;
    (void)Am;
    (void)An;
    (void)myrank;
    (void)np;
    (void)proc_involved;
}

void
INSERT_TASK_zperm_allreduce_send_perm( const RUNTIME_option_t *options,
                                       cham_dir_t              dir,
                                       CHAM_ipiv_t            *ipiv,
                                       int                     ipivk,
                                       int                     myrank,
                                       int                     np,
                                       int                    *proc_involved )
{
    (void)options;
    (void)dir;
    (void)ipiv;
    (void)ipivk;
    (void)myrank;
    (void)np;
    (void)proc_involved;
}

void
INSERT_TASK_zperm_allreduce_send_invp_row( const RUNTIME_option_t *options,
                                           cham_dir_t              dir,
                                           CHAM_ipiv_t            *ipiv,
                                           int                     ipivk,
                                           const CHAM_desc_t      *A,
                                           int                     k,
                                           int                     n )
{
    (void)options;
    (void)dir;
    (void)ipiv;
    (void)ipivk;
    (void)A;
    (void)k;
    (void)n;
}

void
INSERT_TASK_zperm_allreduce_send_invp_col( const RUNTIME_option_t *options,
                                           cham_dir_t              dir,
                                           CHAM_ipiv_t            *ipiv,
                                           int                     ipivk,
                                           const CHAM_desc_t      *A,
                                           int                     m,
                                           int                     k )
{
    (void)options;
    (void)dir;
    (void)ipiv;
    (void)ipivk;
    (void)A;
    (void)m;
    (void)k;
}

/**
 * Allreduce of the workspaces of the involved ranks: reduction in the
 * workspace of the owner of A(k, n), and copy of the result to the others.
 * With replicated submission, this is called once for all the involved ranks.
 */
void
INSERT_TASK_zperm_allreduce( const RUNTIME_option_t *options,
                             cham_dir_t              dir,
                             const CHAM_desc_t      *A,
                             CHAM_desc_t            *U,
                             int                     Um,
                             int                     Un,
                             CHAM_ipiv_t            *ipiv,
                             int                     ipivk,
                             int                     k,
                             int                     n,
                             void                   *ws )
{
    struct chameleon_pzlaswp_s *tmp = (struct chameleon_pzlaswp_s *)ws;

    chameleon_parsec_zperm_reduce_submit( options, dir, A, k, n, ipiv, ipivk, U, Um, Un,
                                          &(tmp->ws), Um, Un, &(tmp->reduce), 1 );
}
