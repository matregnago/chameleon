/**
 *
 * @file descriptor_rec.c
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
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

typedef struct chameleon_recdesc_dist_s {
    int myrank;     /**< Rank of the calling process.                           */
    int p;          /**< Number of process rows at the distributed level.       */
    int q;          /**< Number of process columns at the distributed level.    */
    int dist_level; /**< Recursion level where the 2D distribution is applied.  */
    int level;      /**< Current recursion level.                               */
    int i;          /**< Global row tile coordinate at the current level.       */
    int j;          /**< Global column tile coordinate at the current level.    */
    int owner;      /**< Owning rank, or CHAMELEON_RECDESC_OWNER_DISTRIBUTED.   */

    int64_t  tag_root;       /**< First tag reserved for the complete hierarchy.    */
    int64_t  tag_base;       /**< First tag assigned to the current descriptor.     */
    int64_t  tag_count;      /**< Total number of tags reserved by the root.        */
    int64_t  tag_span;       /**< Tags reserved for one tile at the current level.  */
    int64_t  tag_tree_span;  /**< Primary-handle tags below one current-level tile. */
    int64_t *tag_tree_spans; /**< Primary-handle subtree span at every level.       */
    int      tag_ld;         /**< Leading dimension of the distributed tag grid.    */
} chameleon_recdesc_dist_t;

/**
 * @brief Precompute the maximum subtree tag span at every recursive level.
 *
 * A leaf consumes one tag. At every coarser level, one tag is used by the
 * current tile handle and a complete child span is reserved for each tile in
 * its exact refinement. Partial recursion policies may leave some tags unused.
 */
static int
chameleon_recdesc_tag_tree_spans_init( chameleon_recdesc_dist_t *dist,
                                       const int *mb, const int *nb )
{
    int64_t span = 1;
    int     depth = 0;
    int     level;

    while ( ( mb[depth] > 0 ) && ( nb[depth] > 0 ) ) {
        depth++;
    }
    assert( depth > 0 );

    dist->tag_tree_spans = malloc( (size_t)depth * sizeof(int64_t) );
    if ( dist->tag_tree_spans == NULL ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "malloc() failed" );
        return CHAMELEON_ERR_OUT_OF_RESOURCES;
    }

    dist->tag_tree_spans[depth-1] = span;
    for ( level = depth - 2; level >= 0; level-- ) {
        span = 1 + (int64_t)( mb[level] / mb[level+1] ) *
                           ( nb[level] / nb[level+1] ) * span;
        dist->tag_tree_spans[level] = span;
    }
    return CHAMELEON_SUCCESS;
}

/**
 * @brief Initialize the deterministic MPI-tag layout of a recursive hierarchy.
 *
 * Each tile at the distributed level receives an equally sized contiguous
 * range. Its primary-handle subtree occupies @c tag_tree_span entries, and a
 * mixed-precision descriptor reserves two additional conversion tags. The root
 * reserves all distributed-tile ranges in one operation.
 */
static int
chameleon_recdesc_tag_layout_init( chameleon_recdesc_dist_t *dist,
                                   cham_flttype_t dtyp, int m, int n,
                                   const int *mbs, const int *nbs )
{
    int dist_level = dist->dist_level;
    int tag_lnt;
    int rc;

    rc = chameleon_recdesc_tag_tree_spans_init( dist, mbs, nbs );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    dist->tag_root      = -1;
    dist->tag_base      = -1;
    dist->tag_ld        = ( m + mbs[dist_level] - 1 ) / mbs[dist_level];
    dist->tag_tree_span = dist->tag_tree_spans[dist_level];
    dist->tag_span      = dist->tag_tree_span + ( cham_is_mixed( dtyp ) ? 2 : 0 );

    tag_lnt         = ( n + nbs[dist_level] - 1 ) / nbs[dist_level];
    dist->tag_count = (int64_t)dist->tag_ld * tag_lnt * dist->tag_span;
    return CHAMELEON_SUCCESS;
}

/**
 * @brief Check that the active runtime can manage recursive descriptors.
 */
static int
chameleon_recdesc_check_runtime( const CHAM_context_t *chamctxt, const char *funcname )
{
#if !defined(CHAMELEON_USE_RECURSIVE_TASKS)
    (void)chamctxt;
    chameleon_error( funcname, "recursive descriptors require recursive-task support" );
    return CHAMELEON_ERR_NOT_SUPPORTED;
#else
    if ( chamctxt->scheduler != RUNTIME_SCHED_STARPU ) {
        chameleon_error( funcname, "recursive descriptors are only available with StarPU" );
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }
    return CHAMELEON_SUCCESS;
#endif
}

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
 * @brief Return the unique owner of every tile in a single-owner recursive level.
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
 * @param[in] layout
 *          Matrix layout parameters.
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
chameleon_recdesc_check_blocking( const CHAM_desc_layout_t *layout,
                                  const int *mb, const int *nb )
{
    int k;

    assert( layout != NULL );
    assert( mb != NULL );
    assert( nb != NULL );

    for ( k = 0; ( mb[k] > 0 ) && ( nb[k] > 0 ); k++ ) {
        if ( ( mb[k+1] <= 0 ) || ( nb[k+1] <= 0 ) ) {
            if ( ( mb[k+1] <= 0 ) != ( nb[k+1] <= 0 ) ) {
                chameleon_error( "CHAMELEON_Desc_CreateEx",
                                 "recursive row and column tile lists must end together" );
                return CHAMELEON_ERR_ILLEGAL_VALUE;
            }
            break;
        }

        if ( ( ( mb[k] % mb[k+1] ) && ( mb[k] != layout->m ) ) ||
             ( ( nb[k] % nb[k+1] ) && ( nb[k] != layout->n ) ) )
        {
            chameleon_error( "CHAMELEON_Desc_CreateEx",
                             "recursive tile sizes must be multiples of the next level" );
            return CHAMELEON_ERR_ILLEGAL_VALUE;
        }
    }

    if ( mb[k] != nb[k] ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "finest recursive tiles must be square" );
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

    return CHAMELEON_SUCCESS;
}

/**
 * @brief Validate storage modes supported by recursive descriptors.
 */
static int
chameleon_recdesc_check_storage( const CHAM_desc_storage_t   *storage,
                                 const CHAM_desc_recursion_t *recargs )
{
#if !defined(CHAMELEON_USE_MPI)
    if ( CHAMELEON_MAT_IS_RUNTIME_ALLOC( storage->mat ) ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx",
                         "runtime-allocated recursive descriptors require MPI support" );
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }
#endif

    if ( ( recargs->dist_level > 0 ) &&
         !CHAMELEON_MAT_IS_RUNTIME_ALLOC( storage->mat ) &&
         ( storage->mat != CHAMELEON_MAT_ALLOC_GLOBAL ) )
    {
        chameleon_error(
            "CHAMELEON_Desc_CreateEx",
            "recursive distribution below level 0 requires global, tile, or OOC allocation" );
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }

    return CHAMELEON_SUCCESS;
}

/**
 * @brief Validate a recursive partitioning policy and its argument.
 */
static int
chameleon_recdesc_check_policy( const CHAM_desc_recursion_t *recargs )
{
    switch ( recargs->kind ) {
    case ChamRecFull:
        return CHAMELEON_SUCCESS;

    case ChamRecRandom:
        if ( recargs->arg > 0 ) {
            return CHAMELEON_SUCCESS;
        }
        break;

    case ChamRecDiag:
    case ChamRecSmart:
        if ( recargs->arg >= 0 ) {
            return CHAMELEON_SUCCESS;
        }
        break;

    default:
        break;
    }

    chameleon_error( "CHAMELEON_Desc_CreateEx",
                     "invalid recursive partitioning policy or argument" );
    return CHAMELEON_ERR_ILLEGAL_VALUE;
}

/**
 * @brief Return whether a descriptor contains data owned by the calling rank.
 *
 * Shared tiles marked with @c CHAMELEON_MPI_WITH_ME also count as local.
 *
 * @param[in] desc
 *          Descriptor level to inspect.
 *
 * @return 1 if at least one tile is local, or 0 otherwise.
 */
static int
chameleon_recdesc_contains_local_tile( const CHAM_desc_t *desc )
{
    int m, n;

    for ( n = 0; n < desc->nt; n++ ) {
        for ( m = 0; m < desc->mt; m++ ) {
            int rank = desc->get_rankof( desc, m, n );

            if ( ( rank == desc->myrank ) ||
                 ( rank == CHAMELEON_MPI_WITH_ME ) )
            {
                return 1;
            }
        }
    }

    return 0;
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
 * @brief Initialize a hierarchy-only descriptor level without physical data.
 *
 * Levels above the distributed level describe recursive task decomposition.
 * Their tiles are parent handles whose storage is provided by their children.
 *
 * @param[inout] desc
 *          Descriptor to initialize.
 */
static void
chameleon_recdesc_init_hierarchy_storage( CHAM_desc_t *desc )
{
    desc->alloc_mat    = 0;
    desc->use_mat      = 0;
    desc->register_mat = 0;
    desc->ooc          = 0;
    desc->mat          = NULL;
    desc->get_blkaddr  = chameleon_getaddr_null;
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
                              CHAM_desc_t *desc, int myrank, const char *name,
                              const CHAM_desc_storage_t      *storage,
                              const CHAM_desc_layout_t       *layout,
                              const chameleon_recdesc_dist_t *dist,
                              int dist_it,      int dist_jt,
                              int dist_mstride, int dist_nstride )
{
    CHAM_desc_storage_t rank_storage = *storage;
    int rc;

    if ( dist->owner != CHAMELEON_RECDESC_OWNER_DISTRIBUTED ) {
        /*
         * Below the distributed level, every tile of the child descriptor is
         * owned by the rank that owns the parent tile. The rank callback is
         * replaced accordingly before initializing the descriptor invariants.
         */
        rank_storage.get_rankof     = chameleon_recdesc_getrankof_single_owner;
        rank_storage.get_rankof_arg = (void*)(intptr_t)(dist->owner);
    }

    rc = chameleon_desc_init_base( desc, myrank, name, &rank_storage, layout );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    desc->mpitag           = ( dist->level < dist->dist_level ) ? dist->tag_root
                                                                : dist->tag_base;
    desc->mpitag_size      = 0;
    desc->mpitag_tile_span = dist->tag_span;
    desc->mpitag_tree_span = dist->tag_tree_span;
    desc->mpitag_ld        = ( dist->level == dist->dist_level ) ? dist->tag_ld
                                                                 : desc->mt;
    desc->mpitag_use       = ( dist->level >= dist->dist_level );
    desc->mpitag_dist      = ( dist->level == dist->dist_level );
    if ( dist->level == 0 ) {
        desc->mpitag_size  = dist->tag_count;
        desc->mpitag_owner = 1;
        desc->mpitag       = -1;
    }

    if ( dist->owner != CHAMELEON_RECDESC_OWNER_DISTRIBUTED ) {
        chameleon_recdesc_init_single_owner_level( desc, storage->mat, dist->owner );
    }
    else {
        chameleon_desc_init_2d_distribution_with_offset( desc, dist->p, dist->q, dist_it, dist_jt );
        desc->dist_mstride = dist_mstride;
        desc->dist_nstride = dist_nstride;

        if ( dist->level < dist->dist_level ) {
            chameleon_recdesc_init_hierarchy_storage( desc );
        }
        else {
            rc = chameleon_desc_init_storage( chamctxt, desc, storage->mat );
            if ( rc != CHAMELEON_SUCCESS ) {
                return rc;
            }
        }
    }

    chameleon_desc_register_with_offset( desc );

    return CHAMELEON_SUCCESS;
}

static int
chameleon_recdesc_create( const CHAM_context_t *chamctxt,
                          const CHAM_desc_create_t *args, CHAM_desc_t *desc,
                          chameleon_recdesc_dist_t *dist )
{
    const CHAM_desc_storage_t   *storage = &(args->storage);
    const CHAM_desc_recursion_t *recargs = args->recursive;
    const int                   *mb      = recargs->mbs + dist->level;
    const int                   *nb      = recargs->nbs + dist->level;
    CHAM_desc_create_t           child_args;
    chameleon_recdesc_dist_t     child_dist;
    CHAM_desc_t *tiledesc;
    CHAM_tile_t *tile;
    char        *subname;
    void        *child_mat;
    int          tempmm, tempnn;
    int          dist_mstride, dist_nstride;
    int          rc, i, j, m, n;

    /* At least one pair of recursive tile sizes is required. */
    assert( ( mb[0] > 0 ) && ( nb[0] > 0 ) );

    dist_mstride = 1;
    dist_nstride = 1;
    if ( dist->level < dist->dist_level ) {
        dist_mstride = mb[0] / mb[dist->dist_level - dist->level];
        dist_nstride = nb[0] / nb[dist->dist_level - dist->level];
    }

    /* Create the current layer descriptor */
    rc = chameleon_recdesc_init_level( chamctxt, desc, dist->myrank, args->name,
                                       storage, &(args->layout), dist,
                                       dist->i * dist_mstride,
                                       dist->j * dist_nstride,
                                       dist_mstride, dist_nstride );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    if ( dist->level == 0 ) {
        dist->tag_root = desc->mpitag;
    }

    /* Stop once the finest recursive level has been initialized. */
    if ( ( mb[1] <= 0 ) || ( nb[1] <= 0 ) ) {
        return CHAMELEON_SUCCESS;
    }

    for ( n = 0; n < desc->nt; n++ ) {
        /* Coordinate of the first child tile, used by ChamRecDiag. */
        j = ( dist->j + n ) * ( nb[0] / nb[1] );

        for ( m = 0; m < desc->mt; m++ ) {
            /* Coordinate of the first child tile, used by ChamRecDiag. */
            i = ( dist->i + m ) * ( mb[0] / mb[1] );

            switch ( recargs->kind ) {
            case ChamRecFull:
                break;
            case ChamRecRandom:
                /*
                 * arg = 1: equivalent to ChamRecFull
                 * arg = n: every n-th tile is partitioned
                 */
                if ( random() % recargs->arg ) continue;
                break;
            case ChamRecDiag:
                /*
                 * arg = n: the first n tiles under and above the diagonal will
                 * be partitioned
                 */
                if ( abs( i - j ) > recargs->arg ) continue;
                break;
            case ChamRecSmart:
                /*
                 * arg defines the number of tiles to partition
                 */
                if ( n*desc->mt + m >= recargs->arg ) continue;
                break;
            default:
                return CHAMELEON_ERR_UNEXPECTED;
            }

            tile   = desc->get_blktile( desc, m, n );
            tempmm = desc->get_blkdim( desc, m, DIM_m, desc->m );
            tempnn = desc->get_blkdim( desc, n, DIM_n, desc->n );

            chameleon_asprintf( &subname, "%s[%d,%d]", desc->name, m, n );

            tiledesc = (CHAM_desc_t*)malloc( sizeof(CHAM_desc_t) );
            if ( tiledesc == NULL ) {
                free( subname );
                chameleon_error( "CHAMELEON_Desc_CreateEx", "malloc() failed" );
                return CHAMELEON_ERR_OUT_OF_RESOURCES;
            }

            /*
             * Hierarchy levels forward the requested allocation mode until it
             * is applied at dist_level. Below that level, StarPU partition
             * handles provide views into the parent tile.
             */
            child_mat = ( dist->level < dist->dist_level )
                      ? storage->mat
                      : ( ( tile->rank == desc->myrank ) ? tile->mat : NULL );

            child_args = *args;
            child_args.name       = subname;
            child_args.layout.mb  = mb[1];
            child_args.layout.nb  = nb[1];
            child_args.layout.lm  = tile->ld;
            child_args.layout.ln  = tempnn;
            child_args.layout.m   = tempmm;
            child_args.layout.n   = tempnn;
            child_args.storage.mat            = child_mat;
            child_args.storage.get_blkaddr    = ( (intptr_t)child_mat > 0 )
                                              ? chameleon_getaddr_cm
                                              : ( ( dist->level >= dist->dist_level )
                                                ? chameleon_getaddr_null
                                                : NULL );
            child_args.storage.get_blkldd     = ( (intptr_t)child_mat > 0 )
                                              ? chameleon_getblkldd_cm : NULL;
            child_args.storage.get_rankof     = NULL;
            child_args.storage.get_rankof_arg = NULL;

            child_dist = *dist;
            child_dist.level = dist->level + 1;
            child_dist.i     = i;
            child_dist.j     = j;
            child_dist.owner = ( dist->level < dist->dist_level )
                             ? CHAMELEON_RECDESC_OWNER_DISTRIBUTED : tile->rank;
            child_dist.tag_root      = dist->tag_root;
            child_dist.tag_tree_span = dist->tag_tree_spans[child_dist.level];
            child_dist.tag_span      = child_dist.tag_tree_span;
            if ( child_dist.level == child_dist.dist_level ) {
                child_dist.tag_span += cham_is_mixed( desc->dtyp ) ? 2 : 0;
                child_dist.tag_base  = dist->tag_root;
            }
            else if ( child_dist.level > child_dist.dist_level ) {
                child_dist.tag_base = chameleon_desc_get_mpi_tag( desc, m, n, 0 ) + 1;
            }

            rc = chameleon_recdesc_create( chamctxt, &child_args, tiledesc, &child_dist );
            free( subname );

            if ( rc != CHAMELEON_SUCCESS ) {
                return rc;
            }

            tile->format |= CHAMELEON_TILE_DESC;
            tile->mat    = tiledesc;
            if ( dist->level < dist->dist_level ) {
                tile->rank = chameleon_recdesc_contains_local_tile( tiledesc )
                           ? CHAMELEON_MPI_WITH_ME
                           : CHAMELEON_MPI_WITHOUT_ME;
                assert( ( tile->rank == CHAMELEON_MPI_WITH_ME ) ||
                        ( tile->rank == CHAMELEON_MPI_WITHOUT_ME ) );
            }
        }
    }

    return CHAMELEON_SUCCESS;
}

/**
 * @brief Initialize a recursive descriptor from validated extended arguments.
 *
 * This internal entry point builds recursive workspaces without repeating the
 * public context lookup and descriptor validation performed by
 * CHAMELEON_Desc_CreateEx(). The caller must provide a valid initialized
 * context and mutually consistent descriptor parameters.
 *
 * @param[in] chamctxt
 *          Initialized CHAMELEON context.
 * @param[inout] desc
 *          Descriptor storage to initialize.
 * @param[in] name
 *          Descriptor name used for debugging.
 * @param[in] mat
 *          Matrix pointer or CHAMELEON allocation mode.
 * @param[in] dtyp
 *          Matrix precision.
 * @param[in] rec
 *          Recursive partitioning policy.
 * @param[in] rarg
 *          Policy-specific recursive partitioning argument.
 * @param[in] mbs
 *          Zero-terminated row tile sizes for each recursive level.
 * @param[in] nbs
 *          Zero-terminated column tile sizes for each recursive level.
 * @param[in] lm
 *          Number of rows in the complete matrix storage.
 * @param[in] ln
 *          Number of columns in the complete matrix storage.
 * @param[in] m
 *          Number of rows in the described matrix.
 * @param[in] n
 *          Number of columns in the described matrix.
 * @param[in] p
 *          Number of process rows in the 2D distribution.
 * @param[in] q
 *          Number of process columns in the 2D distribution.
 * @param[in] dist_level
 *          Recursive level at which the 2D distribution applies.
 * @param[in] get_blkaddr
 *          Optional tile-address callback.
 * @param[in] get_blkldd
 *          Optional tile-leading-dimension callback.
 * @param[in] get_rankof
 *          Optional tile-rank callback.
 * @param[in] get_rankof_arg
 *          Argument passed to @p get_rankof.
 *
 * @retval CHAMELEON_SUCCESS on success.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE if the recursive descriptor arguments
 *         are invalid.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED if the requested storage mode is not
 *         supported by the active runtime.
 * @retval CHAMELEON_ERR_OUT_OF_RESOURCES if descriptor allocation fails.
 */
int
chameleon_recdesc_init( const CHAM_context_t *chamctxt,
                        CHAM_desc_t *desc, const char *name, void *mat,
                        cham_flttype_t dtyp, cham_rec_t rec, int rarg,
                        const int *mbs, const int *nbs,
                        int lm, int ln, int m, int n,
                        int p, int q, int dist_level,
                        blkaddr_fct_t    get_blkaddr,
                        blkldd_fct_t     get_blkldd,
                        blkrankof_fct_t  get_rankof,
                        void            *get_rankof_arg )
{
    cham_data_dist_t data_dist = {
        .get_distrib        = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib            = { p, q }
    };
    CHAM_desc_recursion_t recursion = {
        .kind       = rec,
        .arg        = rarg,
        .mbs        = mbs,
        .nbs        = nbs,
        .dist_level = dist_level
    };
    CHAM_desc_create_t args = {
        .name = name,
        .layout = {
            .dtyp = dtyp,
            .mb   = mbs[0],
            .nb   = nbs[0],
            .lm   = lm,
            .ln   = ln,
            .i    = 0,
            .j    = 0,
            .m    = m,
            .n    = n
        },
        .storage = {
            .mat            = mat,
            .get_blkaddr    = get_blkaddr,
            .get_blkldd     = get_blkldd,
            .get_rankof     = get_rankof,
            .get_rankof_arg = get_rankof_arg
        },
        .data_dist = &data_dist,
        .recursive = &recursion
    };
    chameleon_recdesc_dist_t dist = {
        .myrank     = RUNTIME_comm_rank( chamctxt ),
        .p          = p,
        .q          = q,
        .dist_level = dist_level,
        .level      = 0,
        .i          = 0,
        .j          = 0,
        .owner      = CHAMELEON_RECDESC_OWNER_DISTRIBUTED
    };
    int rc;

    {
        rc = chameleon_recdesc_check_storage( &(args.storage), &recursion );

        if ( rc != CHAMELEON_SUCCESS ) {
            return rc;
        }
    }
#if !defined(NDEBUG)
    {
        rc = chameleon_recdesc_check_blocking( &(args.layout), mbs, nbs );

        if ( rc != CHAMELEON_SUCCESS ) {
            return rc;
        }
    }
#endif

    rc = chameleon_recdesc_tag_layout_init( &dist, dtyp, m, n, mbs, nbs );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    rc = chameleon_recdesc_create( chamctxt, &args, desc, &dist );
    free( dist.tag_tree_spans );

    return rc;
}

/**
 * @brief Create a recursive descriptor from the extended descriptor arguments.
 *
 * @retval CHAMELEON_SUCCESS on success.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE if the recursive descriptor arguments
 *         are invalid.
 * @retval CHAMELEON_ERR_NOT_INITIALIZED if CHAMELEON is not initialized.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED if recursive-task support is unavailable
 *         or the requested distributed storage mode is unsupported.
 * @retval CHAMELEON_ERR_OUT_OF_RESOURCES if descriptor allocation fails.
 */
int
chameleon_desc_create_recursive( CHAM_desc_t **descptr, const CHAM_desc_create_t *args )
{
    CHAM_context_t              *chamctxt;
    const CHAM_desc_storage_t   *storage;
    const CHAM_desc_recursion_t *recargs;
    chameleon_recdesc_dist_t     dist;
    CHAM_desc_t *desc;
    int status, p, q;

    chamctxt = chameleon_context_self();
    if ( chamctxt == NULL ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "CHAMELEON not initialized" );
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    status = chameleon_recdesc_check_runtime( chamctxt, "CHAMELEON_Desc_CreateEx" );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    if ( ( descptr == NULL ) || ( args == NULL ) || ( args->recursive == NULL ) ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "invalid descriptor creation arguments" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    storage = &(args->storage);
    recargs = args->recursive;

    assert( args->layout.i == 0 );
    assert( args->layout.j == 0 );

    if ( ( recargs->mbs == NULL ) || ( recargs->nbs == NULL ) ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "invalid recursive blocking parameters" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    status = chameleon_recdesc_check_policy( recargs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    status = chameleon_recdesc_check_blocking( &(args->layout), recargs->mbs, recargs->nbs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    status = chameleon_recdesc_check_dist_level( recargs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    status = chameleon_recdesc_check_storage( storage, recargs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    status = chameleon_recdesc_get_2d_grid( args->data_dist, &p, &q );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    dist.myrank     = RUNTIME_comm_rank( chamctxt );
    dist.p          = p;
    dist.q          = q;
    dist.dist_level = recargs->dist_level;
    dist.level      = 0;
    dist.i          = 0;
    dist.j          = 0;
    dist.owner      = CHAMELEON_RECDESC_OWNER_DISTRIBUTED;
    status = chameleon_recdesc_tag_layout_init( &dist, args->layout.dtyp,
                                                args->layout.m, args->layout.n,
                                                recargs->mbs, recargs->nbs );
    if ( status != CHAMELEON_SUCCESS ) {
        return status;
    }

    /* Create the current layer descriptor */
    desc = (CHAM_desc_t*)malloc( sizeof(CHAM_desc_t) );
    if ( desc == NULL ) {
        free( dist.tag_tree_spans );
        chameleon_error( "CHAMELEON_Desc_CreateEx", "malloc() failed" );
        return CHAMELEON_ERR_OUT_OF_RESOURCES;
    }

    status = chameleon_recdesc_create( chamctxt, args, desc, &dist );
    free( dist.tag_tree_spans );

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
