/**
 *
 * @file context.h
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon context header
 *
 * @version 1.4.0
 * @author Jakub Kurzak
 * @author Cedric Augonnet
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @date 2024-02-18
 *
 */
#ifndef _chameleon_context_h_
#define _chameleon_context_h_

#include "chameleon/struct.h"

/**
 *  Routines to handle threads context
 */
#ifdef __cplusplus
extern "C" {
#endif

CHAM_context_t* chameleon_context_create  ();
CHAM_context_t* chameleon_context_self    ();
int             chameleon_context_destroy ();

/**
 * @brief Return true if every rank submits all the tasks in the same order.
 *
 * In that case, the algorithms must not prune the submission with the local
 * rank: loops cover the whole iteration space and the workspaces are indexed
 * by the owner of the tile instead of the local process.
 */
static inline int
chameleon_replicated_submission( const CHAM_context_t *chamctxt )
{
    return (chamctxt != NULL) && chamctxt->replicated_submission;
}

/**
 * @brief Return true if the runtime supports the reduction access mode
 * (ChamRW|ChamCOMMUTE or STARPU_REDUX).
 */
static inline int
chameleon_runtime_has_reductions( const CHAM_context_t *chamctxt )
{
    return (chamctxt != NULL) && (chamctxt->scheduler != RUNTIME_SCHED_PARSEC);
}

#ifdef __cplusplus
}
#endif

#endif /* _chameleon_context_h_ */
