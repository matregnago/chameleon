/**
 *
 * @file descriptor_rec.c
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon descriptors routines
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Gwenole Lucas
 * @author Lionel Eyraud-Dubois
 * @date 2025-12-19
 *
 */
#include "control/common.h"
#include "chameleon/runtime.h"

static int
chameleon_recdesc_get_2d_grid( const cham_data_dist_t *data_dist, int *p, int *q )
{
    CHAM_desc_t desc = { 0 };

    if ( data_dist == NULL ) {
        *p = 1;
        *q = 1;
        return CHAMELEON_SUCCESS;
    }

    if ( ( data_dist->get_distrib != (datadist_access_fct_t)chameleon_get_2d_block_cyclic ) ||
         ( data_dist->distrib_array_size < 2 ) )
    {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "only 2D block-cyclic descriptor distributions are supported" );
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }

    desc.data_dist = (cham_data_dist_t*)data_dist;
    *p = chameleon_desc_datadist_get_iparam( &desc, 0 );
    *q = chameleon_desc_datadist_get_iparam( &desc, 1 );
    return CHAMELEON_SUCCESS;
}

/**
 * @brief Initialize and register one level of a recursive descriptor.
 *
 * This mirrors chameleon_desc_init(), but keeps the distribution and tile-rank
 * registration offset-aware so children can later be distributed with their
 * global tile coordinates.
 */
static int
chameleon_recdesc_init_level( const CHAM_context_t *chamctxt,
                              const char *name, CHAM_desc_t *desc, void *mat, cham_flttype_t dtyp,
                              int mb, int nb, int lm, int ln, int m, int n, int p, int q,
                              int dist_it, int dist_jt,
                              blkaddr_fct_t get_blkaddr, blkldd_fct_t get_blkldd,
                              blkrankof_fct_t get_rankof, void* get_rankof_arg )
{
    int rc;

    rc = chameleon_desc_init_base( chamctxt, desc, name, mat, dtyp, mb, nb,
                                   lm, ln, m, n,
                                   get_blkaddr, get_blkldd, get_rankof, get_rankof_arg );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    chameleon_desc_init_2d_distribution_with_offset( desc, p, q, dist_it, dist_jt );

    rc = chameleon_desc_init_storage( chamctxt, desc, mat );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    chameleon_desc_register_with_offset( desc );

    return CHAMELEON_SUCCESS;
}

static int
chameleon_recdesc_create( const CHAM_context_t *chamctxt,
                          const char *name, CHAM_desc_t *desc, void *mat, cham_flttype_t dtyp,
                          cham_rec_t rec, int rarg, int *mb, int *nb,
                          int lm, int ln, int m, int n, int p, int q, int i0, int j0,
                          blkaddr_fct_t get_blkaddr, blkldd_fct_t get_blkldd,
                          blkrankof_fct_t get_rankof, void* get_rankof_arg )
{
    CHAM_desc_t *tiledesc;
    CHAM_tile_t *tile;
    char        *subname;
    int          tempmm, tempnn;
    int          rc, i, j, m, n;

    /* Let's make sure we have at least one couple (mb, nb) defined */
    assert( (mb[0] > 0) && (nb[0] > 0) );

    /* Create the current layer descriptor */
    rc = chameleon_recdesc_init_level( chamctxt, name, desc, mat, dtyp, mb[0], nb[0],
                                       lm, ln, m, n, p, q, i0, j0,
                                       get_blkaddr, get_blkldd, get_rankof, get_rankof_arg );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    /* Move to the next tile size to recurse */
    mb++;
    nb++;
    if ( (mb[0] <= 0) || (nb[0] <= 0) ) {
        return CHAMELEON_SUCCESS;
    }

    for ( n=0; n<desc->nt; n++ ) {
        j = j0 * desc->nt + n; /* Used when rec = diag */

        for ( m=0; m<desc->mt; m++ ) {
            i = i0 * desc->mt + m; /* Used when rec = diag */

            switch (rec) {
            case ChamRecFull:
                break;
            case ChamRecRandom:
                /*
                 * rarg = 1: equivalent to ChamRecFull
                 * rarg = n: every n-th tiles are partitionned
                 */
                if ( random() % rarg ) continue;
                break;
            case ChamRecDiag:
                /*
                 * rarg = n: the first n tiles under and above the diagonal will
                 * be partitionned
                 */
                if ( abs( i - j ) > rarg ) continue;
                break;
            case ChamRecSmart:
                /*
                 * rarg defines the number of tiles that needs to be partitionned
                 */
                if ( n*desc->mt + m >= rarg ) continue;
                break;
            default:
                return CHAMELEON_ERR_UNEXPECTED;
            }

            tile   = desc->get_blktile( desc, m, n );
            tempmm = desc->get_blkdim( desc, m, DIM_m, desc->m );
            tempnn = desc->get_blkdim( desc, n, DIM_n, desc->n );

            chameleon_asprintf( &subname, "%s[%d,%d]", desc->name, m, n );

            tiledesc = (CHAM_desc_t*)malloc(sizeof(CHAM_desc_t));
            rc = chameleon_recdesc_create( chamctxt, subname, tiledesc, tile->mat, desc->dtyp,
                                           rec, rarg, mb, nb,
                                           tile->ld, tempnn, /* Abuse as ln is not used */
                                           tempmm, tempnn,
                                           1, 1,             /* can recurse only on local data */
                                           i, j,             /* Used when rec = diag */
                                           chameleon_getaddr_cm, chameleon_getblkldd_cm,
                                           NULL, NULL );
            free( subname );

            tile->format = CHAMELEON_TILE_DESC;
            tile->mat    = tiledesc;

            if ( rc != CHAMELEON_SUCCESS ) {
                return rc;
            }
        }
    }

    return CHAMELEON_SUCCESS;
}

int
chameleon_desc_create_recursive( CHAM_desc_t **descptr, const CHAM_desc_create_t *args )
{
    CHAM_context_t              *chamctxt;
    const CHAM_desc_storage_t   *storage;
    const CHAM_desc_recursion_t *recargs;
    CHAM_desc_t *desc;
    int status, p, q;

    if ( ( descptr == NULL ) || ( args == NULL ) || ( args->recursive == NULL ) ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "invalid descriptor creation arguments" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    storage = &(args->storage);
    recargs = args->recursive;

    /*
     * The first layer must be allocated, otherwise we will give unitialized
     * pointers to the lower layers
     */
    assert( (storage->mat != CHAMELEON_MAT_ALLOC_TILE) &&
            (storage->mat != CHAMELEON_MAT_OOC) );
    assert( args->layout.i == 0 );
    assert( args->layout.j == 0 );

    if ( ( recargs->mbs == NULL ) || ( recargs->nbs == NULL ) ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "invalid recursive blocking parameters" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    status = chameleon_recdesc_get_2d_grid( args->data_dist, &p, &q );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    chamctxt = chameleon_context_self();
    if ( chamctxt == NULL ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "CHAMELEON not initialized" );
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    if ( chamctxt->scheduler != RUNTIME_SCHED_STARPU ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "CHAMELEON Recursive descriptors only available with StaRPU" );
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    /* Create the current layer descriptor */
    desc = (CHAM_desc_t*)malloc(sizeof(CHAM_desc_t));
    if (desc == NULL) {
        chameleon_error("CHAMELEON_Desc_CreateEx", "malloc() failed");
        return CHAMELEON_ERR_OUT_OF_RESOURCES;
    }

    status = chameleon_recdesc_create( chamctxt, args->name, desc, storage->mat, layout->dtyp,
                                       recargs->kind, recargs->arg,
                                       (int*)recargs->mbs, (int*)recargs->nbs,
                                       layout->lm, layout->ln, layout->m, layout->n, p, q, 0, 0,
                                       storage->get_blkaddr, storage->get_blkldd,
                                       storage->get_rankof, storage->get_rankof_arg );

    *descptr = desc;
    return status;
}

int
CHAMELEON_Recursive_Desc_Create( CHAM_desc_t **descptr, void *mat, cham_flttype_t dtyp,
                                 cham_rec_t rec, int rarg, int *mb, int *nb,
                                 int lm, int ln, int m, int n, int p, int q,
                                 blkaddr_fct_t get_blkaddr, blkldd_fct_t get_blkldd,
                                 blkrankof_fct_t get_rankof, void* get_rankof_arg,
                                 const char *name )
{
    cham_data_dist_t dist = {
        .get_distrib = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib = { p, q }
    };
    CHAM_desc_recursion_t recargs = {
        .kind = rec,
        .arg = rarg,
        .mbs = mb,
        .nbs = nb
    };
    CHAM_desc_create_t args = {
        .name = name,
        .layout = {
            .dtyp = dtyp,
            .mb = ( mb != NULL ) ? mb[0] : 0,
            .nb = ( nb != NULL ) ? nb[0] : 0,
            .lm = lm,
            .ln = ln,
            .i = 0,
            .j = 0,
            .m = m,
            .n = n
        },
        .storage = {
            .mat = mat,
            .get_blkaddr = get_blkaddr,
            .get_blkldd = get_blkldd,
            .get_rankof = get_rankof,
            .get_rankof_arg = get_rankof_arg
        },
        .data_dist = &dist,
        .recursive = &recargs
    };

    return chameleon_desc_create_recursive( descptr, &args );
}
