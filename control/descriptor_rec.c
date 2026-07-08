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

#define CHAMELEON_RECDESC_OWNER_DISTRIBUTED (-1)

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
 * @brief Return the unique owner of every tile in a private recursive level.
 *
 * Below the level where the recursive descriptor is distributed, a child
 * descriptor is a local refinement of one parent tile. All tiles in that child
 * descriptor therefore inherit the parent tile owner instead of using the
 * distributed 2D rank callback.
 *
 * @param[in] desc
 *          Descriptor whose get_rankof_init_arg stores the owner rank.
 *
 * @param[in] m
 *          Ignored tile row index.
 *
 * @param[in] n
 *          Ignored tile column index.
 *
 * @return The MPI rank that owns the whole descriptor level.
 */
static int
chameleon_recdesc_getrankof_single_owner( const CHAM_desc_t *desc, int m, int n )
{
    (void)m;
    (void)n;
    return (int)(intptr_t)( desc->get_rankof_init_arg );
}

/**
 * @brief Check that all declared recursive tile sizes are exact refinements.
 *
 * @param[in] mb
 *          Zero-terminated list of row tile sizes.
 *
 * @param[in] nb
 *          Zero-terminated list of column tile sizes.
 *
 * @retval CHAMELEON_SUCCESS on success.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE if one level cannot be exactly split by
 *         the next one.
 */
static int
chameleon_recdesc_check_blocking( const int *mb, const int *nb )
{
    int k;

    assert( mb != NULL );
    assert( nb != NULL );

    for ( k = 0; ( mb[k] > 0 ) && ( nb[k] > 0 ); k++ ) {
        if ( ( mb[k+1] <= 0 ) || ( nb[k+1] <= 0 ) ) {
            break;
        }

        if ( ( mb[k] % mb[k+1] ) || ( nb[k] % nb[k+1] ) ) {
            chameleon_error( "CHAMELEON_Desc_CreateEx",
                             "recursive tile sizes must be multiples of the next level" );
            return CHAMELEON_ERR_ILLEGAL_VALUE;
        }
    }

    if ( mb[k] != nb[k] ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "recursive row and column tile lists must end together" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    return CHAMELEON_SUCCESS;
}

/**
 * @brief Check that the distributed level exists in the recursive hierarchy.
 *
 * @param[in] recargs
 *          Recursive descriptor creation parameters.
 *
 * @retval CHAMELEON_SUCCESS on success.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE if the requested distributed level does
 *         not exist.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED if the requested distributed level is not
 *         implemented yet.
 */
static int
chameleon_recdesc_check_dist_level( const CHAM_desc_recursion_t *recargs )
{
    int depth = 0;

    if ( recargs->dist_level < 0 ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "recursive distribution level must be non-negative" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    while ( ( recargs->mbs[depth] > 0 ) && ( recargs->nbs[depth] > 0 ) ) {
        depth++;
    }

    if ( recargs->dist_level >= depth ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "recursive distribution level exceeds the descriptor depth" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    if ( recargs->dist_level != 0 ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "recursive distribution below the first level is not supported yet" );
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }

    return CHAMELEON_SUCCESS;
}

/**
 * @brief Initialize descriptor metadata for a level fully owned by one rank.
 *
 * Levels below the distributed level are no longer shared by all MPI ranks.
 * They either belong entirely to the owner of the parent tile, or are empty on
 * all other ranks. This helper initializes both the local dimensions and the
 * matrix pointer according to this single-owner rule.
 *
 * @param[inout] desc
 *          Descriptor to initialize.
 *
 * @param[in] mat
 *          Pointer to the parent tile storage on the owner rank.
 *
 * @param[in] owner
 *          MPI rank owning the parent tile storage.
 */
static void
chameleon_recdesc_init_single_owner_level( CHAM_desc_t *desc, void *mat, int owner )
{
    cham_data_dist_t dist = {
        .get_distrib = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib = { 1, 1 }
    };

    chameleon_desc_set_datadist( desc, &dist );

    desc->alloc_mat    = 0;
    desc->register_mat = 0;
    desc->ooc          = 0;

    if ( desc->myrank == owner ) {
        desc->llmt    = desc->lmt;
        desc->llnt    = desc->lnt;
        desc->llm     = desc->lm;
        desc->lln     = desc->ln;
        desc->llm1    = ( desc->lm % desc->mb ) ? desc->llmt - 1 : desc->llmt;
        desc->lln1    = ( desc->ln % desc->nb ) ? desc->llnt - 1 : desc->llnt;
        desc->mat     = mat;
        desc->use_mat = ( mat != NULL );
    }
    else {
        desc->llmt    = 0;
        desc->llnt    = 0;
        desc->llm     = 0;
        desc->lln     = 0;
        desc->llm1    = 0;
        desc->lln1    = 0;
        desc->mat     = NULL;
        desc->use_mat = 0;
    }

    desc->A21 = (size_t)(desc->llm - desc->llm % desc->mb) * (size_t)(desc->lln - desc->lln % desc->nb);
    desc->A12 = (size_t)(            desc->llm % desc->mb) * (size_t)(desc->lln - desc->lln % desc->nb) + desc->A21;
    desc->A22 = (size_t)(desc->llm - desc->llm % desc->mb) * (size_t)(            desc->lln % desc->nb) + desc->A12;
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
                              int dist_it, int dist_jt, int dist_mstride, int dist_nstride,
                              int owner,
                              blkaddr_fct_t get_blkaddr, blkldd_fct_t get_blkldd,
                              blkrankof_fct_t get_rankof, void* get_rankof_arg )
{
    int rc;

    if ( owner != CHAMELEON_RECDESC_OWNER_DISTRIBUTED ) {
        /*
         * Below the distributed level, every tile of the child descriptor is
         * owned by the rank that owns the parent tile. The rank callback is
         * replaced accordingly before initializing the descriptor invariants.
         */
        get_rankof     = chameleon_recdesc_getrankof_single_owner;
        get_rankof_arg = (void*)(intptr_t)owner;
    }

    rc = chameleon_desc_init_base( chamctxt, desc, name, mat, dtyp, mb, nb,
                                   lm, ln, m, n,
                                   get_blkaddr, get_blkldd, get_rankof, get_rankof_arg );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    if ( owner != CHAMELEON_RECDESC_OWNER_DISTRIBUTED ) {
        chameleon_recdesc_init_single_owner_level( desc, mat, owner );
    }
    else {
        chameleon_desc_init_2d_distribution_with_offset( desc, p, q, dist_it, dist_jt );
        desc->dist_mstride = dist_mstride;
        desc->dist_nstride = dist_nstride;

        rc = chameleon_desc_init_storage( chamctxt, desc, mat );
        if ( rc != CHAMELEON_SUCCESS ) {
            return rc;
        }
    }

    chameleon_desc_register_with_offset( desc );

    return CHAMELEON_SUCCESS;
}

static int
chameleon_recdesc_create( const CHAM_context_t *chamctxt,
                          const char *name, CHAM_desc_t *desc, void *mat, cham_flttype_t dtyp,
                          cham_rec_t rec, int rarg, int *mb, int *nb,
                          int lm, int ln, int m, int n, int p, int q,
                          int level, int dist_level, int i0, int j0, int owner,
                          blkaddr_fct_t get_blkaddr, blkldd_fct_t get_blkldd,
                          blkrankof_fct_t get_rankof, void* get_rankof_arg )
{
    CHAM_desc_t *tiledesc;
    CHAM_tile_t *tile;
    char        *subname;
    int          tempmm, tempnn;
    int          child_p, child_q, child_owner;
    int          dist_mstride, dist_nstride;
    int          rc, i, j, m, n;

    /* Let's make sure we have at least one couple (mb, nb) defined */
    assert( (mb[0] > 0) && (nb[0] > 0) );

    dist_mstride = 1;
    dist_nstride = 1;
    if ( level < dist_level ) {
        dist_mstride = mb[0] / mb[dist_level - level];
        dist_nstride = nb[0] / nb[dist_level - level];
    }

    /* Create the current layer descriptor */
    rc = chameleon_recdesc_init_level( chamctxt, name, desc, mat, dtyp, mb[0], nb[0],
                                       lm, ln, m, n, p, q,
                                       i0 * dist_mstride, j0 * dist_nstride,
                                       dist_mstride, dist_nstride, owner,
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
            if ( tiledesc == NULL ) {
                free( subname );
                chameleon_error("CHAMELEON_Desc_CreateEx", "malloc() failed");
                return CHAMELEON_ERR_OUT_OF_RESOURCES;
            }

            if ( level < dist_level ) {
                child_p     = p;
                child_q     = q;
                child_owner = -1;
            }
            else {
                child_p     = 1;
                child_q     = 1;
                child_owner = tile->rank;
            }

            rc = chameleon_recdesc_create( chamctxt, subname, tiledesc,
                                           ( tile->rank == desc->myrank ) ? tile->mat : NULL,
                                           desc->dtyp,
                                           rec, rarg, mb, nb,
                                           tile->ld, tempnn, /* Abuse as ln is not used */
                                           tempmm, tempnn,
                                           child_p, child_q,
                                           level + 1, dist_level,
                                           i, j, child_owner,
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

/**
 * @brief Create a recursive descriptor from the extended descriptor arguments.
 *
 * @retval CHAMELEON_SUCCESS on success.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE if the recursive descriptor arguments
 *         are invalid.
 * @retval CHAMELEON_ERR_NOT_INITIALIZED if CHAMELEON is not initialized or the
 *         selected runtime cannot support recursive descriptors.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED if the requested distributed recursive
 *         storage mode is not supported yet.
 * @retval CHAMELEON_ERR_OUT_OF_RESOURCES if descriptor allocation fails.
 */
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

    status = chameleon_recdesc_check_blocking( recargs->mbs, recargs->nbs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    status = chameleon_recdesc_check_dist_level( recargs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
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
                                       layout->lm, layout->ln, layout->m, layout->n, p, q,
                                       0, recargs->dist_level, 0, 0,
                                       CHAMELEON_RECDESC_OWNER_DISTRIBUTED,
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
        .kind       = rec,
        .arg        = rarg,
        .mbs        = mb,
        .nbs        = nb,
        .dist_level = 0
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
