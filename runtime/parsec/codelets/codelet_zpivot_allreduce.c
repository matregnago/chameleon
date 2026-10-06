/**
 *
 * @file parsec/codelet_zpivot_allreduce.c
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
#include <coreblas/cblas_wrapper.h>

#if defined(CHAMELEON_USE_MPI)

/*
 * Merge the candidates of one rank into the buffer of the root.
 */
static inline int
CORE_zpivot_merge_parsec( parsec_execution_stream_t *context,
                          parsec_task_t             *this_task )
{
    int h, n, nb;
    cham_flttype_t dtyp;
    void *mebuf, *srcbuf;
    chameleon_parsec_pivot_hdr_t *hme, *hsrc;
    CHAM_pivot_t me, src;
    CHAMELEON_Complex64_t *pivrow_me, *pivrow_src;

    parsec_dtd_unpack_args( this_task, &h, &n, &nb, &dtyp, &mebuf, &srcbuf );

    hme  = (chameleon_parsec_pivot_hdr_t *)mebuf;
    hsrc = (chameleon_parsec_pivot_hdr_t *)srcbuf;
    chameleon_parsec_pivot_load( mebuf,  nb, dtyp, &me  );
    chameleon_parsec_pivot_load( srcbuf, nb, dtyp, &src );
    pivrow_me  = (CHAMELEON_Complex64_t *)(me.pivrow);
    pivrow_src = (CHAMELEON_Complex64_t *)(src.pivrow);

    assert( hme->h == hsrc->h );

    if ( cabs( pivrow_src[ h ] ) > cabs( pivrow_me[ h ] ) ) {
        hme->blkm0  = hsrc->blkm0;
        hme->blkidx = hsrc->blkidx;
        cblas_zcopy( n, pivrow_src, 1, pivrow_me, 1 );
    }

    /* Let's copy the diagonal row if needed */
    if ( ( hsrc->has_diag == 1 ) && ( hme->has_diag == -1 ) ) {
        cblas_zcopy( n, src.diagrow, 1, me.diagrow, 1 );
        hme->has_diag = 1;
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Without reduction in DTD, the buffers of all the ranks that contributed to
 * the step h are merged, in rank order, into the buffer of the root, which is
 * read by the next step.
 */
void INSERT_TASK_zpivot_allreduce( const RUNTIME_option_t *options,
                                   CHAM_desc_t            *A,
                                   CHAM_desc_pivot_t      *pivot,
                                   int                     k,
                                   int                     h,
                                   int                     n,
                                   void                   *ws )
{
    parsec_taskpool_t        *PARSEC_dtd_taskpool = (parsec_taskpool_t *)(options->sequence->schedopt);
    chameleon_parsec_pivot_t *ppivot = chameleon_parsec_pivot( pivot );
    int                       nb     = pivot->nb;
    cham_flttype_t            dtyp   = pivot->dtyp;
    int                       p;

    if ( h >= pivot->n ) {
        return;
    }

    for ( p = 0; p < ppivot->NP; p++ ) {
        if ( ( p == ppivot->root ) || ( ppivot->init_stamp[p] != ppivot->stamp ) ) {
            continue;
        }
        parsec_dtd_insert_task(
            PARSEC_dtd_taskpool, CORE_zpivot_merge_parsec, options->priority, PARSEC_DEV_CPU, "pivot_merge",
            sizeof(int),            &h,    PARSEC_VALUE,
            sizeof(int),            &n,    PARSEC_VALUE,
            sizeof(int),            &nb,   PARSEC_VALUE,
            sizeof(cham_flttype_t), &dtyp, PARSEC_VALUE,
            PASSED_BY_REF, chameleon_parsec_pivot_tile( options, pivot, ppivot->root, h ),
                           chameleon_parsec_pivot_arena( pivot ) | PARSEC_INOUT | PARSEC_AFFINITY,
            PASSED_BY_REF, chameleon_parsec_pivot_tile( options, pivot, p, h ),
                           chameleon_parsec_pivot_arena( pivot ) | PARSEC_INPUT,
            PARSEC_DTD_ARG_END );
    }

    (void)A;
    (void)k;
    (void)ws;
}

#endif
