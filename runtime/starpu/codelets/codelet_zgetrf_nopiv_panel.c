/**
 *
 * @file starpu/codelet_zgetrf_nopiv_panel.c
 *
 * @copyright 2025-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon recursive zgetrf_nopiv StarPU panel codelets
 *
 * @version 1.4.0
 * @author Alycia Lisito
 * @author Mathieu Faverge
 * @date 2024-10-18
 * @precisions normal z -> c d s
 *
 */
#include "chameleon_starpu_internal.h"
#include "runtime_codelet_z.h"

#if !defined(CHAMELEON_USE_RECURSIVE_TASKS)
#error "This file should be compiled only when recursive tasks are enabled"
#endif

static inline void
cl_zgetrf_nopiv_panel_facto_rectask_func( struct starpu_task *t, void *_args, void **descrs )
{
    rectask_args_t    *rtargs = (rectask_args_t *)_args;
    RUNTIME_request_t  request = RUNTIME_REQUEST_INITIALIZER;
    int                k;

    (void)descrs;

    starpu_codelet_unpack_args( t->cl_arg, &k );
    starpu_cham_rectask_initrequest( t, &request );

    if ( rtargs->tiles[1] ) {
        chameleon_pzgetrf_nopiv_ws_panel_facto( rtargs->tiles[0]->mat, rtargs->tiles[1]->mat, k,
                                                rtargs->sequence, &request );
    }
    else {
        chameleon_pzgetrf_nopiv_generic_panel_facto( rtargs->tiles[0]->mat, k,
                                                     rtargs->sequence, &request );
    }

    free( rtargs );
}

static struct starpu_codelet cl_zgetrf_nopiv_panel_facto = {
    .nbuffers                    = STARPU_VARIABLE_NBUFFERS,
    .recursive_task_gen_dag_func = cl_zgetrf_nopiv_panel_facto_rectask_func,
    .name                        = "zgetrf_nopiv_panel_facto",
    .model                       = NULL,
    .energy_model                = NULL,
    .cpu_funcs                   = { (starpu_cpu_func_t)1 },
};

void INSERT_TASK_zgetrf_nopiv_panel_facto( const RUNTIME_option_t *options,
                                           int                     k,
                                           const CHAM_desc_t      *A,
                                           int                     An,
                                           const CHAM_desc_t      *WU,
                                           int                     WUm,
                                           int                     WUn )
{
    CHAM_tile_t    *tileA;
    CHAM_tile_t    *tileWU;
    rectask_args_t *rtargs;
    int             workspace  = ( WU != NULL );
    int             access_WU  = workspace ? STARPU_W : STARPU_NONE;
    int             is_rectask = 1;
    int             exec       = 0;

    /* Handle cache */
    exec = chameleon_desc_islocal( A, 0, An ) ||
           ( ( WU != NULL ) && chameleon_desc_islocal( WU, WUm, WUn ) );
    if ( exec == 0 ) {
        return;
    }

    tileA  = A->get_blktile( A, 0, An );
    tileWU = WU ? WU->get_blktile( WU, WUm, WUn ) : NULL;

    /* Make sure we have a descriptor */
    assert( tileA->format & CHAMELEON_TILE_DESC );

    rtargs = starpu_cham_rectask_args_create( options, 2 );
    rtargs->tiles[0] = tileA;
    rtargs->tiles[1] = tileWU;

    rt_starpu_insert_task(
        &cl_zgetrf_nopiv_panel_facto,

        /* Task codelet arguments */
        STARPU_VALUE, &k, sizeof(int),

        /* Handles */
        STARPU_RW, RTBLKADDR( A, ChamComplexDouble, 0, An ),
        access_WU, workspace ? RTBLKADDR( WU, ChamComplexDouble, WUm, WUn ) : NULL,

        /* Common task arguments */
        INSERT_TASK_COMMON_TASK_PARAMS_NOCB( zgetrf_nopiv_panel_facto ),

        /* Recursive task management */
        INSERT_TASK_RECTASK_PARAMS( zgetrf_nopiv_panel_facto )
        0 );
}

static inline void
cl_zgetrf_nopiv_panel_update_rectask_func( struct starpu_task *t, void *_args, void **descrs )
{
    rectask_args_t    *rtargs = (rectask_args_t *)_args;
    RUNTIME_request_t  request = RUNTIME_REQUEST_INITIALIZER;
    int                k;

    (void)descrs;

    starpu_codelet_unpack_args( t->cl_arg, &k );

    starpu_cham_rectask_initrequest( t, &request );

    if ( rtargs->tiles[2] ) {
        chameleon_pzgetrf_nopiv_ws_panel_update( rtargs->tiles[0]->mat, rtargs->tiles[1]->mat,
                                                 rtargs->tiles[2]->mat, k,
                                                 rtargs->sequence, &request );
    }
    else {
        chameleon_pzgetrf_nopiv_generic_panel_update( rtargs->tiles[1]->mat, rtargs->tiles[0]->mat,
                                                      k, rtargs->sequence, &request );
    }

    free( rtargs );
}

/*
 * Codelet definition
 */
static struct starpu_codelet cl_zgetrf_nopiv_panel_update = {
    .nbuffers                    = STARPU_VARIABLE_NBUFFERS,
    .recursive_task_gen_dag_func = cl_zgetrf_nopiv_panel_update_rectask_func,
    .name                        = "zgetrf_nopiv_panel_update",
    .model                       = NULL,
    .energy_model                = NULL,
    .cpu_funcs                   = { (starpu_cpu_func_t)1 },
};

void INSERT_TASK_zgetrf_nopiv_panel_update( const RUNTIME_option_t *options,
                                            int                     k,
                                            const CHAM_desc_t      *A,
                                            int                     An,
                                            const CHAM_desc_t      *L,
                                            int                     Ln,
                                            const CHAM_desc_t      *U,
                                            int                     Um,
                                            int                     Un )
{
    CHAM_tile_t    *tileA;
    CHAM_tile_t    *tileL;
    CHAM_tile_t    *tileU;
    rectask_args_t *rtargs;
    int             accessU    = U ? STARPU_W : STARPU_NONE;
    int             is_rectask = 1;
    int             exec       = 0;

    /* Handle cache */
    exec = chameleon_desc_islocal( A, 0, An ) ||
           chameleon_desc_islocal( L, 0, Ln ) ||
           ( ( U != NULL ) && chameleon_desc_islocal( U, Um, Un ) );
    if ( exec == 0 ) {
        return;
    }

    tileA = A->get_blktile( A, 0, An );
    tileL = L->get_blktile( L, 0, Ln );
    tileU = U ? U->get_blktile( U, Um, Un ) : NULL;

    /* Make sure we have a descriptor */
    assert( tileA->format & CHAMELEON_TILE_DESC );
    assert( tileL->format & CHAMELEON_TILE_DESC );

    rtargs = starpu_cham_rectask_args_create( options, 3 );
    rtargs->tiles[0] = tileA;
    rtargs->tiles[1] = tileL;
    rtargs->tiles[2] = tileU;

    rt_starpu_insert_task(
        &cl_zgetrf_nopiv_panel_update,

        /* Task codelet arguments */
        STARPU_VALUE, &k, sizeof(int),

        /* Handles */
        STARPU_RW, RTBLKADDR( A, ChamComplexDouble, 0, An ),
        STARPU_R,  RTBLKADDR( L, ChamComplexDouble, 0, Ln ),
        accessU,   U ? RTBLKADDR( U, ChamComplexDouble, Um, Un ) : NULL,

        /* Common task arguments */
        INSERT_TASK_COMMON_TASK_PARAMS_NOCB( zgetrf_nopiv_panel_update ),

        /* Recursive task management */
        INSERT_TASK_RECTASK_PARAMS( zgetrf_nopiv_panel_update )
        0 );
}
