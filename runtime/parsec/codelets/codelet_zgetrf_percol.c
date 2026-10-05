/**
 *
 * @file parsec/codelet_zgetrf_percol.c
 *
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zgetrf_percol Parsec codelets
 *
 * @version 1.4.0
 * @comment Codelets to perform panel factorization with partial pivoting
 *
 * @author Mathieu Faverge
 * @author Matthieu Kuhn
 * @author Alycia Lisito
 * @author Matteo Marcos
 * @date 2026-10-05
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_parsec.h"
#include "chameleon/tasks_z.h"

/*
 * The per column algorithm is the blocked one with a single block covering the
 * whole tile, and without the Up workspace.
 */
void INSERT_TASK_zgetrf_percol_diag( const RUNTIME_option_t *options,
                                     int m, int n, int h, int m0,
                                     CHAM_desc_t *A, int Am, int An,
                                     CHAM_ipiv_t       *ipiv,
                                     CHAM_desc_pivot_t *pivot )
{
    int ib = A->get_blktile( A, Am, An )->n;

    INSERT_TASK_zgetrf_blocked_diag( options, m, n, h, m0, ib, 0,
                                     A, Am, An, NULL, 0, 0, ipiv, pivot );
}

void INSERT_TASK_zgetrf_percol_offdiag( const RUNTIME_option_t *options,
                                        int m, int n, int h, int m0,
                                        CHAM_desc_t *A, int Am, int An,
                                        CHAM_desc_pivot_t *pivot )
{
    int ib = A->get_blktile( A, Am, An )->n;

    INSERT_TASK_zgetrf_blocked_offdiag( options, m, n, h, m0, ib, 0,
                                        A, Am, An, NULL, 0, 0, pivot );
}
