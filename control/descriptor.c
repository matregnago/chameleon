/**
 *
 * @file descriptor.c
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
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @author Guillaume Sylvand
 * @author Raphael Boucherie
 * @author Samuel Thibault
 * @author Lionel Eyraud-Dubois
 * @author Pierre Esterie
 * @author Atte Torri
 * @author Brieuc Nicolas
 * @date 2025-12-19
 *
 ***
 *
 * @defgroup Descriptor
 * @brief Group descriptor routines exposed to users
 *
 */
#include "control/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "control/descriptor.h"
#include "chameleon/runtime.h"

static int nbdesc = 0;

/**
 * @brief Generate a default descriptor name for unnamed descriptors.
 */
static char *
chameleon_desc_get_name( void )
{
    static int counter = 0;
    char      *name    = malloc( sizeof(char) * 4 );
    int        idx     = 0;

    name[idx] = 'x';
    idx++;

    if ( counter > 26 ) {
        name[idx] = 'A' + ( ( counter / 26 ) % 26 );
        idx++;
    }

    name[idx] = 'A' + counter % 26;
    idx++;

    name[idx] = '\0';

    counter++;
    return name;
}

void chameleon_desc_init_tiles_with_offset( CHAM_desc_t *desc, blkrankof_fct_t rankof,
                                            int dist_it, int dist_jt )
{
    CHAM_tile_t *tile;
    int8_t flttype = cham_get_flttype( desc->dtyp );
    int ii, jj;

    assert( rankof != chameleon_getrankof_tile );
    desc->tiles = malloc( sizeof(CHAM_tile_t) * desc->lmt * desc->lnt );

    tile = desc->tiles;
    for( jj=0; jj<desc->lnt; jj++ ) {
        for( ii=0; ii<desc->lmt; ii++, tile++ ) {
            int rank = rankof( desc,
                               dist_it + ii * desc->dist_mstride,
                               dist_jt + jj * desc->dist_nstride );
            tile->format  = CHAMELEON_TILE_FULLRANK;
            tile->flttype = flttype;
            tile->rank    = rank;
            tile->m       = ii == desc->lmt-1 ? desc->lm - ii * desc->mb : desc->mb;
            tile->n       = jj == desc->lnt-1 ? desc->ln - jj * desc->nb : desc->nb;
            tile->mat     = (rank == desc->myrank) ? desc->get_blkaddr( desc, ii, jj ) : NULL;
            tile->ld      = desc->get_blkldd( desc, ii );
#if defined(CHAMELEON_KERNELS_TRACE)
            chameleon_asprintf( &(tile->name), "%s(%d,%d)", desc->name, ii, jj );
#endif
        }
    }
}

void chameleon_desc_init_tiles( CHAM_desc_t *desc, blkrankof_fct_t rankof )
{
    chameleon_desc_init_tiles_with_offset( desc, rankof, 0, 0 );
}

/* Get access to data dist */
int chameleon_desc_datadist_get_iparam( const CHAM_desc_t *desc, int i )
{
    return desc->data_dist->distrib[desc->data_dist->get_distrib(desc, i)];
}

int chameleon_get_2d_block_cyclic( const CHAM_desc_t *desc, int i )
{
    (void)desc;
    return i;
}

void chameleon_desc_set_datadist( CHAM_desc_t *to, cham_data_dist_t *from )
{
    int i;
    to->data_dist = malloc(sizeof(cham_data_dist_t));
    to->data_dist->get_distrib = from->get_distrib;
    to->data_dist->distrib_array_size = from->distrib_array_size;

    for (i = 0; i < to->data_dist->distrib_array_size; i++) {
        to->data_dist->distrib[i] = from->distrib[i];
    }
}

/**
 ******************************************************************************
 *
 * @ingroup Descriptor
 *
 * @brief Initialize descriptor invariants shared by all descriptor creation
 * paths.
 *
 ******************************************************************************
 *
 * @param[in] name
 *          Name of the descriptor for debug purpose.
 *
 * @param[in] dtyp
 *          Data type of the matrix:
 *          @arg ChamRealFloat:     single precision real (S),
 *          @arg ChamRealDouble:    double precision real (D),
 *          @arg ChamComplexFloat:  single precision complex (C),
 *          @arg ChamComplexDouble: double precision complex (Z).
 *
 * @param[in] mb
 *          Number of rows in a tile.
 *
 * @param[in] nb
 *          Number of columns in a tile.
 *
 * @param[in] lm
 *          Number of rows of the entire matrix.
 *
 * @param[in] ln
 *          Number of columns of the entire matrix.
 *
 * @param[in] m
 *          Number of rows of the submatrix.
 *
 * @param[in] n
 *          Number of columns of the submatrix.
 *
 * @param[in] get_blkaddr
 *          A function which return the address of the data corresponding to
 *          the tile A(m,n).
 *
 * @param[in] get_blkldd
 *          A function that return the leading dimension of the tile A(m,*).
 *
 * @param[in] get_rankof
 *          A function that return the MPI rank of the tile A(m,n).
 *
 * @param[in] get_rankof_arg
 *          A pointer to custom data that can be used by the get_rankof function
 *
 ******************************************************************************
 *
 * @return  The descriptor with the common matrix description parameters set.
 *
 */
int chameleon_desc_init_base( const CHAM_context_t *chamctxt,
                              CHAM_desc_t *desc, const char *name, void *mat,
                              cham_flttype_t dtyp, int mb, int nb,
                              int lm, int ln, int m, int n,
                              blkaddr_fct_t   get_blkaddr,
                              blkldd_fct_t    get_blkldd,
                              blkrankof_fct_t get_rankof,
                              void           *get_rankof_arg )
{
    assert( chamctxt );
    memset( desc, 0, sizeof(CHAM_desc_t) );

    if ( name ) {
        desc->name = strdup( name );
    }
    else {
        desc->name = chameleon_desc_get_name();
    }

    /* If one of the function get_* is NULL, we switch back to the default */
    desc->get_blktile = chameleon_desc_gettile;
    desc->get_blkdim  = chameleon_getblkdim;

    /* Data addresses */
    if ( get_blkaddr ) {
        desc->get_blkaddr = get_blkaddr;
    }
    else {
        if ( (intptr_t)mat > 0 ) {
            desc->get_blkaddr = chameleon_getaddr_cm;
        }
        else {
            desc->get_blkaddr = chameleon_getaddr_ccrb;
        }
    }

    /* Data leading dimensions */
    if ( get_blkldd ) {
        desc->get_blkldd = get_blkldd;
    }
    else {
        if ( (intptr_t)mat > 0 ) {
            desc->get_blkldd = chameleon_getblkldd_cm;
        }
        else {
            desc->get_blkldd = chameleon_getblkldd_ccrb;
        }
    }

    /* Data distribution */
    desc->get_rankof          = chameleon_getrankof_tile;
    desc->get_rankof_init     = get_rankof ? get_rankof : chameleon_getrankof_2d;
    desc->get_rankof_init_arg = get_rankof_arg;

    /* Matrix properties */
    desc->dtyp = dtyp;
    /* Should be given as parameter to follow get_blkaddr (unused) */
    desc->styp = (get_blkaddr == chameleon_getaddr_cm ) ? ChamCM : ChamCCRB;
    desc->mb   = mb;
    desc->nb   = nb;
    desc->bsiz = mb * nb;

    /* Matrix parameters */
    desc->i = 0;
    desc->j = 0;
    desc->m = m;
    desc->n = n;

    /* Global tile origin in the data distribution */
    desc->dist_it = 0;
    desc->dist_jt = 0;
    desc->dist_mstride = 1;
    desc->dist_nstride = 1;

    /* Matrix stride parameters */
    desc->lm = lm;
    desc->ln = ln;

    /* Matrix derived parameters */
    desc->mt  = chameleon_ceil( m, mb );
    desc->nt  = chameleon_ceil( n, nb );
    desc->lmt = chameleon_ceil( lm, mb );
    desc->lnt = chameleon_ceil( ln, nb );

    desc->id = nbdesc;
    nbdesc++;
    desc->occurences = 0;

    desc->myrank = RUNTIME_comm_rank( chamctxt );

    return CHAMELEON_SUCCESS;
}

/**
 * @brief Count how many tile indices are owned by a process coordinate.
 *
 * Counts indices i in the interval [0, nt) such that
 * (tile0 + i) is congruent to myp modulo p.
 *
 * @param[in] nt
 *          Number of tile indices in the interval.
 *
 * @param[in] p
 *          Number of processes in the cyclic distribution.
 *
 * @param[in] myp
 *          Process coordinate in @f$[0, p)@f.
 *
 * @param[in] tile0
 *          Global tile index of the first local interval element.
 *
 * @return Number of local tiles for that process coordinate.
 */
static int
chameleon_desc_count_local_tiles_1d( int nt, int p, int myp, int dist_it )
{
    int shifted_myp;

    assert( p > 0 );
    assert( ( myp >= 0 ) && ( myp < p ) );
    assert( dist_it >= 0 );

    if ( nt <= 0 ) {
        return 0;
    }

    /* Shift p based on the index of the first tile at the distributed level */
    shifted_myp = ( myp - ( dist_it % p ) + p ) % p;

    /* Returns the local number of tiles while using cyclic distribution over p */
    return ( nt / p ) + ( ( nt % p ) > shifted_myp );
}

/**
 * @brief Initialize the 2D block-cyclic distribution and derived local
 * dimensions.
 *
 * @param[inout] desc
 *          Descriptor to initialize.
 *
 * @param[in] p
 *          Number of process rows in the 2D block-cyclic distribution.
 *
 * @param[in] q
 *          Number of process columns in the 2D block-cyclic distribution.
 *
 * @param[in] dist_it
 *          Global row tile index corresponding to local tile row 0.
 *
 * @param[in] dist_jt
 *          Global column tile index corresponding to local tile column 0.
 */
void chameleon_desc_init_2d_distribution_with_offset( CHAM_desc_t *desc, int p, int q,
                                                      int dist_it, int dist_jt )
{
    /* Grid size */
    cham_data_dist_t dist = {
        .get_distrib = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib = {p, q} };
    chameleon_desc_set_datadist( desc, &dist );

    desc->dist_it = dist_it;
    desc->dist_jt = dist_jt;
    desc->dist_mstride = 1;
    desc->dist_nstride = 1;

    /* Local dimensions in tiles */
    if ( desc->myrank < (p*q) ) {
        int myp = desc->myrank / q;
        int myq = desc->myrank % q;
        int lastm = desc->dist_it + desc->lmt - 1;
        int lastn = desc->dist_jt + desc->lnt - 1;

        /* Compute the total number of local tiles */
        desc->llmt = chameleon_desc_count_local_tiles_1d( desc->lmt, p, myp, desc->dist_it );
        desc->llnt = chameleon_desc_count_local_tiles_1d( desc->lnt, q, myq, desc->dist_jt );
        desc->llm1 = desc->llmt;
        desc->lln1 = desc->llnt;

        /* If leading row dimension does not match mb, and I own the last tiles row */
        if ( ( (desc->lm % desc->mb) != 0 ) && ( ( lastm % p ) == myp ) )
        {
            desc->llm1 = desc->llmt - 1;
            desc->llm  = desc->llm1 * desc->mb + (desc->lm % desc->mb);
        } else {
            desc->llm  = desc->llmt * desc->mb;
        }

        /* If leading column dimension does not match nb, and I own the last tiles column */
        if ( ( (desc->ln % desc->nb) != 0 ) && ( ( lastn % q ) == myq ) )
        {
            desc->lln1 = desc->llnt - 1;
            desc->lln  = desc->lln1 * desc->nb + (desc->ln % desc->nb);
        } else {
            desc->lln  = desc->llnt * desc->nb;
        }
    }
    else {
        desc->llmt = 0;
        desc->llnt = 0;
        desc->llm  = 0;
        desc->lln  = 0;
        desc->llm1 = 0;
        desc->lln1 = 0;
    }

    desc->A21 = (size_t)(desc->llm - desc->llm % desc->mb) * (size_t)(desc->lln - desc->lln % desc->nb);
    desc->A12 = (size_t)(            desc->llm % desc->mb) * (size_t)(desc->lln - desc->lln % desc->nb) + desc->A21;
    desc->A22 = (size_t)(desc->llm - desc->llm % desc->mb) * (size_t)(            desc->lln % desc->nb) + desc->A12;
}

void chameleon_desc_init_2d_distribution( CHAM_desc_t *desc, int p, int q )
{
    chameleon_desc_init_2d_distribution_with_offset( desc, p, q, 0, 0 );
}

/**
 * @brief Initialize descriptor memory ownership and allocation mode.
 *
 * @param[in] chamctxt
 *          CHAMELEON context.
 *
 * @param[inout] desc
 *          Descriptor to initialize.
 *
 * @param[in] mat
 *          Matrix storage mode or user-provided matrix pointer.
 *
 * @retval CHAMELEON_SUCCESS on success.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED if out-of-core is requested without StarPU.
 */
int chameleon_desc_init_storage( const CHAM_context_t *chamctxt, CHAM_desc_t *desc, void *mat )
{
    int rc = CHAMELEON_SUCCESS;

    /* memory of the matrix is handled by the user */
    desc->alloc_mat    = 0;
    /* if the user gives a pointer to the overall data (tiles) we can use it */
    desc->use_mat      = 0;
    /* users data can have multiple forms: let him register tiles */
    desc->register_mat = 0;
    /* The matrix is alocated tile by tile with out of core */
    desc->ooc = 0;

    switch ( (intptr_t)mat ) {
    case CHAMELEON_MAT_CASE_ALLOC_TILE:
        if ( chamctxt->scheduler == RUNTIME_SCHED_STARPU ) {
            /* Let's use the allocation on the fly as in OOC */
            desc->get_blkaddr = chameleon_getaddr_null;
            desc->mat = NULL;
            break;
        }
        /* Otherwise we switch back to the full allocation */
        chameleon_attr_fallthrough;

    case CHAMELEON_MAT_CASE_ALLOC_GLOBAL:
        rc = chameleon_desc_mat_alloc( desc );
        desc->alloc_mat = 1;
        desc->use_mat   = 1;
        break;

    case CHAMELEON_MAT_CASE_OOC:
        if ( chamctxt->scheduler != RUNTIME_SCHED_STARPU ) {
            chameleon_error("CHAMELEON_Desc_Create", "CHAMELEON Out-of-Core descriptors are supported only with StarPU");
            return CHAMELEON_ERR_NOT_SUPPORTED;
        }
        desc->get_blkaddr = chameleon_getaddr_null;
        desc->mat = NULL;
        desc->ooc = 1;
        break;

    default:
        /* memory of the matrix is handled by users */
        desc->mat     = mat;
        desc->use_mat = 1;
    }

    return rc;
}

/**
 * @brief Fill descriptor tiles with rank-aware metadata and register them in
 * the runtime.
 *
 * @param[inout] desc
 *          Descriptor to register.
 *
 * The global tile origin is taken from desc->dist_it and desc->dist_jt.
 */
void chameleon_desc_register_with_offset( CHAM_desc_t *desc )
{
    chameleon_desc_init_tiles_with_offset( desc, desc->get_rankof_init,
                                           desc->dist_it, desc->dist_jt );

    /* Create runtime specific structure like registering data */
    RUNTIME_desc_create( desc );
}

void chameleon_desc_register( CHAM_desc_t *desc )
{
    chameleon_desc_register_with_offset( desc );
}

/**
 * @brief Initialize and register a tiled matrix descriptor.
 */
int chameleon_desc_init( const CHAM_context_t *chamctxt,
                         CHAM_desc_t *desc, const char *name, void *mat,
                         cham_flttype_t dtyp, int mb, int nb,
                         int lm, int ln, int m, int n, int p, int q,
                         blkaddr_fct_t   get_blkaddr,
                         blkldd_fct_t    get_blkldd,
                         blkrankof_fct_t get_rankof,
                         void           *get_rankof_arg )
{
    int rc;

    rc = chameleon_desc_init_base( chamctxt, desc, name, mat, dtyp, mb, nb,
                                   lm, ln, m, n,
                                   get_blkaddr, get_blkldd, get_rankof, get_rankof_arg );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    chameleon_desc_init_2d_distribution( desc, p, q );

    rc = chameleon_desc_init_storage( chamctxt, desc, mat );
    if ( rc != CHAMELEON_SUCCESS ) {
        return rc;
    }

    chameleon_desc_register( desc );

    return rc;
}

/**
 *  Internal static descriptor initializer for submatrices
 */
CHAM_desc_t* chameleon_desc_submatrix( CHAM_desc_t *descA, int i, int j, int m, int n )
{
    CHAM_desc_t *descB = malloc(sizeof(CHAM_desc_t));
    int mb, nb;

    if ( (descA->i + i + m) > descA->m ) {
        chameleon_error("chameleon_desc_submatrix", "The number of rows (i+m) of the submatrix doesn't fit in the parent matrix");
        assert((descA->i + i + m) > descA->m);
    }
    if ( (descA->j + j + n) > descA->n ) {
        chameleon_error("chameleon_desc_submatrix", "The number of rows (j+n) of the submatrix doesn't fit in the parent matrix");
        assert((descA->j + j + n) > descA->n);
    }

    memcpy( descB, descA, sizeof(CHAM_desc_t) );
    mb = descA->mb;
    nb = descA->nb;
    /*
     * Submatrix parameters. dist_it and dist_jt are inherited from descA:
     * this is a view into the parent's storage and runtime descriptor, not a
     * new descriptor in the global data distribution.
     */
    descB->i = descA->i + i;
    descB->j = descA->j + j;
    descB->m = m;
    descB->n = n;
    // Submatrix derived parameters
    descB->mt = (m == 0) ? 0 : (descB->i+m-1)/mb - descB->i/mb + 1;
    descB->nt = (n == 0) ? 0 : (descB->j+n-1)/nb - descB->j/nb + 1;

    // Increase the number of occurences to avoid multiple free of runtime specific data structures.
    descB->occurences++;

    return descB;
}

void chameleon_desc_destroy_submit( CHAM_desc_t              *desc,
                                    const RUNTIME_sequence_t *sequence )
{
    int m, n;

    for ( n=0; n<desc->nt; n++ ) {
        for ( m=0; m<desc->mt; m++ ) {
            CHAM_tile_t *tile;

            tile = desc->get_blktile( desc, m, n );

            if ( tile->format == CHAMELEON_TILE_DESC ) {
                CHAM_desc_t *tiledesc = tile->mat;
                chameleon_desc_destroy_submit( tiledesc, sequence );
            }
        }
    }

    RUNTIME_desc_destroy_submit( desc, sequence );

    /*
     * Note that global free operation can't be done here, since the data can
     * still be used until the next call to wait
     */
}

void chameleon_desc_destroy( CHAM_desc_t *desc )
{
    int m, n;

    for ( n=0; n<desc->nt; n++ ) {
        for ( m=0; m<desc->mt; m++ ) {
            CHAM_tile_t *tile;

            tile = desc->get_blktile( desc, m, n );

            if ( tile->format == CHAMELEON_TILE_DESC ) {
                CHAM_desc_t *tiledesc = tile->mat;

                chameleon_desc_destroy( tiledesc );
                free( tiledesc );
                tile->mat = NULL;
            }
        }
    }

    /* Decrease the number of occurences using the descrptor */
    desc->occurences--;

    RUNTIME_desc_destroy( desc );
    chameleon_desc_mat_free( desc );
    if ( ( desc->occurences == 0 ) && desc->name ) {
        free( desc->name );
        desc->name = NULL;
    }
}

/**
 *  Check for descriptor correctness
 */
int chameleon_desc_check(const CHAM_desc_t *desc)
{
    cham_flttype_t flttype;

    if (desc == NULL) {
        chameleon_error("chameleon_desc_check", "NULL descriptor");
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }
    if (desc->mat == NULL && desc->use_mat == 1) {
        chameleon_error("chameleon_desc_check", "NULL matrix pointer");
        return CHAMELEON_ERR_UNALLOCATED;
    }

    flttype = cham_get_flttype( desc->dtyp );
    if ( (flttype != ChamInteger       ) &&
         (flttype != ChamRealHalf      ) &&
         (flttype != ChamRealFloat     ) &&
         (flttype != ChamRealDouble    ) &&
         (flttype != ChamComplexHalf   ) &&
         (flttype != ChamComplexFloat  ) &&
         (flttype != ChamComplexDouble ) )
    {
        chameleon_error("chameleon_desc_check", "invalid matrix type");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    if (desc->mb <= 0 || desc->nb <= 0) {
        chameleon_error("chameleon_desc_check", "negative tile dimension");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    if (desc->bsiz < desc->mb*desc->nb) {
        chameleon_error("chameleon_desc_check", "tile memory size smaller than the product of dimensions");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    if (desc->lm <= 0 || desc->ln <= 0) {
        chameleon_error("chameleon_desc_check", "negative matrix dimension");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    if ((desc->lm < desc->m) || (desc->ln < desc->n)) {
        chameleon_error("chameleon_desc_check", "matrix dimensions larger than leading dimensions");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    if ((desc->i > 0 && desc->i >= desc->lm) || (desc->j > 0 && desc->j >= desc->ln)) {
        chameleon_error("chameleon_desc_check", "beginning of the matrix out of scope");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    if (desc->i+desc->m > desc->lm || desc->j+desc->n > desc->ln) {
        chameleon_error("chameleon_desc_check", "submatrix out of scope");
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }
    return CHAMELEON_SUCCESS;
}

CHAM_desc_t *
CHAMELEON_Desc_SubMatrix( CHAM_desc_t *descA, int i, int j, int m, int n )
{
    return chameleon_desc_submatrix( descA, i, j, m, n );
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Create - Create tiled matrix descriptor.
 *
 ******************************************************************************
 *
 * @param[out] desc
 *          On exit, descriptor of the matrix.
 *
 * @param[in] mat
 *          Memory location of the matrix. If mat is NULL, the space to store
 *          the data is automatically allocated by the call to the function.
 *
 * @param[in] dtyp
 *          Data type of the matrix:
 *          @arg ChamInteger:       integer (i),
 *          @arg ChamRealHalf:      half precision real (H),
 *          @arg ChamRealFloat:     single precision real (S),
 *          @arg ChamRealDouble:    double precision real (D),
 *          @arg ChamComplexHalf:   half precision complex (),
 *          @arg ChamComplexFloat:  single precision complex (C),
 *          @arg ChamComplexDouble: double precision complex (Z).
 *
 * @param[in] mb
 *          Number of rows in a tile.
 *
 * @param[in] nb
 *          Number of columns in a tile.
 *
 * @param[in] bsiz
 *          Size in number of elements of each tile, including internal padding.
 *
 * @param[in] lm
 *          Number of rows of the entire matrix.
 *
 * @param[in] ln
 *          Number of columns of the entire matrix.
 *
 * @param[in] i
 *          Row index to the beginning of the submatrix.
 *
 * @param[in] j
 *          Column indes to the beginning of the submatrix.
 *
 * @param[in] m
 *          Number of rows of the submatrix.
 *
 * @param[in] n
 *          Number of columns of the submatrix.
 *
 * @param[in] p
 *          Number of processes rows for the 2D block-cyclic distribution.
 *
 * @param[in] q
 *          Number of processes columns for the 2D block-cyclic distribution.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Create( CHAM_desc_t **descptr, void *mat, cham_flttype_t dtyp, int mb, int nb, int bsiz,
                           int lm, int ln, int i, int j, int m, int n, int p, int q )
{
    return CHAMELEON_Desc_Create_User( descptr, mat, dtyp, mb, nb, bsiz,
                                       lm, ln, i, j, m, n, p, q,
                                       NULL, NULL, NULL, NULL );
}

/**
 ******************************************************************************
 *
 * @ingroup Descriptor
 *
 * @brief Create a tiled matrix descriptor from structured creation
 * parameters.
 *
 * This interface groups matrix layout, storage, data distribution, and
 * optional recursive tiling parameters in a CHAM_desc_create_t structure.
 * It is intended for descriptor configurations that would otherwise require
 * extending the positional argument list of the legacy creation functions.
 *
 * When @p args->recursive is NULL, a classic tiled descriptor is created.
 * Otherwise, creation is delegated to the recursive descriptor path using
 * the parameters in @p args->recursive.
 *
 * The matrix layout is described by @p args->layout. The @c i and @c j fields
 * are reserved for future submatrix support and must currently be zero.
 *
 * Storage is described by @p args->storage. Its @c mat field may be a
 * user-provided matrix pointer or one of the CHAMELEON allocation modes.
 * NULL callbacks select the standard address, leading-dimension, and rank
 * functions where applicable.
 *
 * @p args->data_dist describes the process grid and data distribution. A NULL
 * value selects a local 1-by-1 distribution. Currently, only the 2D
 * block-cyclic distribution is supported.
 *
 ******************************************************************************
 *
 * @param[out] descptr
 *          On success, descriptor of the matrix. The descriptor must
 *          eventually be released with CHAMELEON_Desc_Destroy().
 *
 * @param[in] args
 *          Descriptor creation parameters. The structure and all referenced
 *          arrays must remain valid for the duration of this call. When a
 *          custom rank callback uses @c storage.get_rankof_arg, that argument
 *          must remain valid for the lifetime of the descriptor.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS
 *          Descriptor successfully created.
 * @retval CHAMELEON_ERR_NOT_INITIALIZED
 *          CHAMELEON has not been initialized.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE
 *          @p descptr, @p args, or one of the descriptor parameters is
 *          invalid.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED
 *          The requested data distribution, storage mode, or recursive
 *          configuration is not supported.
 * @retval CHAMELEON_ERR_OUT_OF_RESOURCES
 *          Memory allocation failed.
 *
 */
int CHAMELEON_Desc_CreateEx( CHAM_desc_t **descptr, const CHAM_desc_create_t *args )
{
    CHAM_context_t *chamctxt;
    CHAM_desc_t *desc = NULL;
    const CHAM_desc_layout_t  *layout;
    const CHAM_desc_storage_t *storage;
    const cham_data_dist_t    *data_dist;
    cham_data_dist_t           default_dist = {
        .get_distrib = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib = { 1, 1 }
    };
    int status;
    int p, q;

    if ( ( descptr == NULL ) || ( args == NULL ) ) {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "invalid descriptor creation arguments" );
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    if ( args->recursive != NULL ) {
        return chameleon_desc_create_recursive( descptr, args );
    }

    layout    = &(args->layout);
    storage   = &(args->storage);
    data_dist = args->data_dist ? args->data_dist : &default_dist;

    assert( layout->i == 0 );
    assert( layout->j == 0 );

    chamctxt = chameleon_context_self();
    if (chamctxt == NULL) {
        chameleon_error("CHAMELEON_Desc_CreateEx", "CHAMELEON not initialized");
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    if ( ( data_dist->get_distrib != (datadist_access_fct_t)chameleon_get_2d_block_cyclic ) ||
         ( data_dist->distrib_array_size < 2 ) )
    {
        chameleon_error( "CHAMELEON_Desc_CreateEx", "only 2D block-cyclic descriptor distributions are supported" );
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }

    p = data_dist->distrib[0];
    q = data_dist->distrib[1];

    /* Allocate memory and initialize the descriptor */
    desc = (CHAM_desc_t*)malloc(sizeof(CHAM_desc_t));
    if (desc == NULL) {
        chameleon_error("CHAMELEON_Desc_CreateEx", "malloc() failed");
        return CHAMELEON_ERR_OUT_OF_RESOURCES;
    }

    status = chameleon_desc_init( chamctxt, desc, args->name, storage->mat, layout->dtyp,
                                  layout->mb, layout->nb,
                                  layout->lm, layout->ln, layout->m, layout->n, p, q,
                                  storage->get_blkaddr, storage->get_blkldd,
                                  storage->get_rankof, storage->get_rankof_arg );
    if (status != CHAMELEON_SUCCESS) {
        if ( desc->name ) {
            free( desc->name );
        }
        chameleon_desc_mat_free( desc );
        free( desc );
        return status;
    }

    status = chameleon_desc_check( desc );
    if (status != CHAMELEON_SUCCESS) {
        chameleon_error("CHAMELEON_Desc_CreateEx", "invalid descriptor");
        CHAMELEON_Desc_Destroy( &desc );
        return status;
    }

    *descptr = desc;

    return CHAMELEON_SUCCESS;
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Create_User - Create generic tiled matrix descriptor for general
 *  applications.
 *
 ******************************************************************************
 *
 * @param[out] desc
 *          On exit, descriptor of the matrix.
 *
 * @param[in] mat
 *          Memory location of the matrix. If mat is NULL, the space to store
 *          the data is automatically allocated by the call to the function.
 *
 * @param[in] dtyp
 *          Data type of the matrix:
 *          @arg ChamRealFloat:     single precision real (S),
 *          @arg ChamRealDouble:    double precision real (D),
 *          @arg ChamComplexFloat:  single precision complex (C),
 *          @arg ChamComplexDouble: double precision complex (Z).
 *
 * @param[in] mb
 *          Number of rows in a tile.
 *
 * @param[in] nb
 *          Number of columns in a tile.
 *
 * @param[in] bsiz
 *          Size in number of elements of each tile, including internal padding.
 *
 * @param[in] lm
 *          Number of rows of the entire matrix.
 *
 * @param[in] ln
 *          Number of columns of the entire matrix.
 *
 * @param[in] i
 *          Row index to the beginning of the submatrix.
 *
 * @param[in] j
 *          Column indes to the beginning of the submatrix.
 *
 * @param[in] m
 *          Number of rows of the submatrix.
 *
 * @param[in] n
 *          Number of columns of the submatrix.
 *
 * @param[in] p
 *          Number of processes rows for the 2D block-cyclic distribution.
 *
 * @param[in] q
 *          Number of processes columns for the 2D block-cyclic distribution.
 *
 * @param[in] get_blkaddr
 *          A function which return the address of the data corresponding to
 *          the tile A(m,n).
 *
 * @param[in] get_blkldd
 *          A function that return the leading dimension of the tile A(m,*).
 *
 * @param[in] get_rankof
 *          A function that return the MPI rank of the tile A(m,n).
 *
 * @param[in] get_rankof_arg
 *          A pointer to custom data that can be used by the get_rankof function
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Create_User( CHAM_desc_t **descptr, void *mat, cham_flttype_t dtyp, int mb, int nb, int bsiz,
                                int lm, int ln, int i, int j, int m, int n, int p, int q,
                                blkaddr_fct_t   get_blkaddr,
                                blkldd_fct_t    get_blkldd,
                                blkrankof_fct_t get_rankof,
                                void* get_rankof_arg )
{
    cham_data_dist_t dist = {
        .get_distrib = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib = { p, q }
    };
    CHAM_desc_create_t args = {
        .name = NULL,
        .layout = {
            .dtyp = dtyp,
            .mb = mb,
            .nb = nb,
            .lm = lm,
            .ln = ln,
            .i = i,
            .j = j,
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
        .recursive = NULL
    };

    (void)i;
    (void)j;
    (void)bsiz;

    return CHAMELEON_Desc_CreateEx( descptr, &args );
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Create_OOC_User - Create matrix descriptor for tiled matrix which
 *  may not fit memory.
 *
 ******************************************************************************
 *
 * @param[out] desc
 *          On exit, descriptor of the matrix.
 *
 * @param[in] dtyp
 *          Data type of the matrix:
 *          @arg ChamRealFloat:     single precision real (S),
 *          @arg ChamRealDouble:    double precision real (D),
 *          @arg ChamComplexFloat:  single precision complex (C),
 *          @arg ChamComplexDouble: double precision complex (Z).
 *
 * @param[in] mb
 *          Number of rows in a tile.
 *
 * @param[in] nb
 *          Number of columns in a tile.
 *
 * @param[in] bsiz
 *          Size in number of elements of each tile, including internal padding.
 *
 * @param[in] lm
 *          Number of rows of the entire matrix.
 *
 * @param[in] ln
 *          Number of columns of the entire matrix.
 *
 * @param[in] i
 *          Row index to the beginning of the submatrix.
 *
 * @param[in] j
 *          Column indes to the beginning of the submatrix.
 *
 * @param[in] m
 *          Number of rows of the submatrix.
 *
 * @param[in] n
 *          Number of columns of the submatrix.
 *
 * @param[in] p
 *          Number of processes rows for the 2D block-cyclic distribution.
 *
 * @param[in] q
 *          Number of processes columns for the 2D block-cyclic distribution.
 *
 * @param[in] get_rankof
 *          A function that return the MPI rank of the tile A(m,n).
 *
 * @param[in] get_rankof_arg
 *          A pointer to custom data that can be used by the get_rankof function
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Create_OOC_User(CHAM_desc_t **descptr, cham_flttype_t dtyp, int mb, int nb, int bsiz,
                                   int lm, int ln, int i, int j, int m, int n, int p, int q,
                                   blkrankof_fct_t get_rankof, void* get_rankof_arg )
{
#if !defined (CHAMELEON_SCHED_STARPU)
    (void)descptr; (void)dtyp; (void)mb; (void)nb; (void)bsiz;
    (void)lm; (void)ln; (void)i; (void)j; (void)m; (void)n; (void)p; (void)q;
    (void)get_rankof;(void)get_rankof_arg;

    chameleon_error("CHAMELEON_Desc_Create_OOC_User", "Only StarPU supports on-demand tile allocation");
    return CHAMELEON_ERR_NOT_SUPPORTED;
#else
    int rc;
    rc = CHAMELEON_Desc_Create_User( descptr, CHAMELEON_MAT_OOC, dtyp, mb, nb, bsiz,
                                     lm, ln, i, j, m, n, p, q,
                                     chameleon_getaddr_null, NULL, get_rankof, get_rankof_arg );
    return rc;
#endif
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Create_OOC - Create matrix descriptor for tiled matrix which may
 *  not fit memory.
 *
 ******************************************************************************
 *
 * @param[out] desc
 *          On exit, descriptor of the matrix.
 *
 * @param[in] dtyp
 *          Data type of the matrix:
 *          @arg ChamRealFloat:     single precision real (S),
 *          @arg ChamRealDouble:    double precision real (D),
 *          @arg ChamComplexFloat:  single precision complex (C),
 *          @arg ChamComplexDouble: double precision complex (Z).
 *
 * @param[in] mb
 *          Number of rows in a tile.
 *
 * @param[in] nb
 *          Number of columns in a tile.
 *
 * @param[in] bsiz
 *          Size in number of elements of each tile, including internal padding.
 *
 * @param[in] lm
 *          Number of rows of the entire matrix.
 *
 * @param[in] ln
 *          Number of columns of the entire matrix.
 *
 * @param[in] i
 *          Row index to the beginning of the submatrix.
 *
 * @param[in] j
 *          Column indes to the beginning of the submatrix.
 *
 * @param[in] m
 *          Number of rows of the submatrix.
 *
 * @param[in] n
 *          Number of columns of the submatrix.
 *
 * @param[in] p
 *          Number of processes rows for the 2D block-cyclic distribution.
 *
 * @param[in] q
 *          Number of processes columns for the 2D block-cyclic distribution.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Create_OOC(CHAM_desc_t **descptr, cham_flttype_t dtyp, int mb, int nb, int bsiz,
                              int lm, int ln, int i, int j, int m, int n, int p, int q)
{
    return CHAMELEON_Desc_Create_User( descptr, CHAMELEON_MAT_OOC, dtyp, mb, nb, bsiz,
                                       lm, ln, i, j, m, n, p, q,
                                       chameleon_getaddr_null, NULL, NULL, NULL );
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 * @brief Creates a new descriptor with the same properties as the one given as
 * input.
 *
 * @warning This function copies the descriptor structure, but does not copy the
 * matrix data.
 *
 ******************************************************************************
 *
 * @param[in] descin
 *          The descriptor structure to duplicate.
 *
 * @param[in] mat
 *          Memory location for the copy. If mat is NULL, the space to store
 *          the data is automatically allocated by the call to the function.
 *
 ******************************************************************************
 *
 * @retval The new matrix descriptor.
 *
 */
CHAM_desc_t *CHAMELEON_Desc_Copy( const CHAM_desc_t *descin, void *mat )
{
    CHAM_desc_t *descout = NULL;
    CHAMELEON_Desc_Create_User( &descout, mat,
                                descin->dtyp, descin->mb, descin->nb, descin->bsiz,
                                descin->lm, descin->ln, descin->i, descin->j, descin->m, descin->n,
                                chameleon_desc_datadist_get_iparam(descin, 0), chameleon_desc_datadist_get_iparam(descin, 1),
                                NULL, NULL, descin->get_rankof_init, descin->get_rankof_init_arg );
    return descout;
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 * @brief Creates a new descriptor with the same properties as the one given as
 * input and restricted on node 0.
 *
 * @warning This function copies the descriptor structure, but does not copy the
 * matrix data.
 *
 ******************************************************************************
 *
 * @param[in] descin
 *          The descriptor structure to duplicate.
 *
 * @param[in] mat
 *          Memory location for the copy. If mat is NULL, the space to store
 *          the data is automatically allocated by the call to the function.
 *
 ******************************************************************************
 *
 * @retval The new matrix descriptor.
 *
 */
CHAM_desc_t *CHAMELEON_Desc_CopyOnZero( const CHAM_desc_t *descin, void *mat )
{
    CHAM_desc_t *descout = NULL;
    CHAMELEON_Desc_Create_User( &descout, mat,
                                descin->dtyp, descin->mb, descin->nb, descin->bsiz,
                                descin->lm, descin->ln, descin->i, descin->j, descin->m, descin->n, 1, 1,
                                NULL, NULL, descin->get_rankof_init, descin->get_rankof_init_arg );
    return descout;
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Destroy - Destroys matrix descriptor.
 *
 ******************************************************************************
 *
 * @param[in] desc
 *          Matrix descriptor.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Destroy(CHAM_desc_t **descptr)
{
    CHAM_context_t *chamctxt;
    CHAM_desc_t *desc;

    chamctxt = chameleon_context_self();
    if (chamctxt == NULL) {
        chameleon_error("CHAMELEON_Desc_Destroy", "CHAMELEON not initialized");
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    if ((descptr == NULL) || (*descptr == NULL)) {
        chameleon_error("CHAMELEON_Desc_Destroy", "attempting to destroy a NULL descriptor");
        return CHAMELEON_ERR_UNALLOCATED;
    }

    desc = *descptr;
    chameleon_desc_destroy( desc );
    free(desc);
    *descptr = NULL;
    return CHAMELEON_SUCCESS;
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Destroy_Submit - Submit the unregisytration/destruction of
 *  the matrix descriptor, data is unregistered when all tasks accessing it are
 *  done. It still needs a call to destroy after a CHAMELEON_Sequence_Wait() to
 *  be fully freed.
 *
 ******************************************************************************
 *
 * @param[inout] desc
 *          Matrix descriptor.
 *
 * @param[in] sequence
 *          Identifies the set of routines to which submit the
 *          destruction/unregistration.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Destroy_Submit( CHAM_desc_t *desc,
                                   const RUNTIME_sequence_t *sequence )
{
    CHAM_context_t *chamctxt;

    chamctxt = chameleon_context_self();
    if (chamctxt == NULL) {
        chameleon_error("CHAMELEON_Desc_Destroy_Submit", "CHAMELEON not initialized");
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    if (desc == NULL) {
        chameleon_error("CHAMELEON_Desc_Destroy_Submit", "attempting to destroy a NULL descriptor");
        return CHAMELEON_ERR_UNALLOCATED;
    }

    if (sequence == NULL) {
        chameleon_error("CHAMELEON_Desc_Destroy_Submit", "incorrect sequence provided to submit unregistration");
        return CHAMELEON_ERR_UNALLOCATED;
    }

    chameleon_desc_destroy_submit( desc, sequence );
    return CHAMELEON_SUCCESS;
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Acquire - Ensures that all data of the descriptor are
 *  up-to-date.
 *
 ******************************************************************************
 *
 * @param[in] desc
 *          Matrix descriptor.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Acquire( const CHAM_desc_t *desc ) {
    return RUNTIME_desc_acquire( desc );
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Release - Release the data of the descriptor acquired by the
 *  application. Should be called if CHAMELEON_Desc_Acquire has been called on the
 *  descriptor and if you do not need to access to its data anymore.
 *
 ******************************************************************************
 *
 * @param[in] desc
 *          Matrix descriptor.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Release( const CHAM_desc_t *desc ) {
    return RUNTIME_desc_release( desc );
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  CHAMELEON_Desc_Flush - Flushes the data in the sequence when they won't be
 *  reused. This calls cleans up the distributed communication caches, and
 *  transfer the data back to the CPU.
 *
 ******************************************************************************
 *
 * @param[in,out] desc
 *          Matrix descriptor.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit
 *
 */
int CHAMELEON_Desc_Flush( CHAM_desc_t              *desc,
                          const RUNTIME_sequence_t *sequence )
{
    RUNTIME_desc_flush( desc, sequence );
    return CHAMELEON_SUCCESS;
}

static void
chameleon_desc_print( const CHAM_desc_t *desc, int shift )
{
    intptr_t base = (intptr_t)desc->mat;
    int m, n, rank;
    CHAM_context_t *chamctxt = chameleon_context_self();

    rank = CHAMELEON_Comm_rank();

    for ( n=0; n<desc->nt; n++ ) {
        for ( m=0; m<desc->mt; m++ ) {
            const CHAM_tile_t *tile;
            const CHAM_desc_t *tiledesc;
            intptr_t ptr;
            int      trank;

            trank    = desc->get_rankof( desc, m, n );
            tile     = desc->get_blktile( desc, m, n );
            tiledesc = tile->mat;
            assert( trank == tile->rank );

            ptr = ( tile->format == CHAMELEON_TILE_DESC ) ? (intptr_t)(tiledesc->mat) : (intptr_t)(tile->mat);

            if ( trank == rank ) {
                fprintf( stdout, "[%2d]%*s%s(%3d,%3d): %d * %d / ld = %d / offset= %ld\n",
                         rank, shift, " ", desc->name, m, n, tile->m, tile->n, tile->ld, ptr - base );

                if ( tile->format == CHAMELEON_TILE_DESC ) {
                    chameleon_desc_print( tiledesc, shift+2 );
                }
            }
            else {
                assert( ptr == 0 );
            }

            if ( chamctxt->scheduler != RUNTIME_SCHED_OPENMP ) {
                RUNTIME_barrier(chamctxt);
            }
        }
    }
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 *  @brief Print descriptor structure for debug purpose
 *
 ******************************************************************************
 *
 * @param[in] desc
 *          The input desc for which to describe to print the tile structure
 */
void
CHAMELEON_Desc_Print( const CHAM_desc_t *desc )
{
    chameleon_desc_print( desc, 2 );
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 * @brief Change the data distribution of the given matrix.
 *
 ******************************************************************************
 *
 * @param[in] uplo
 *          = ChamUpper: Upper triangle of A is moved; lower part is never referenced.
 *          = ChamLower: Lower triangle of A is moved; upper part is never referenced.
 *          = ChamUpperLower: The full matrix A is moved.
 *
 * @param[inout] desc The matrix descriptor to move around. On exit, the new
 *          distribution is registered.
 *
 * @param[in] new_get_rankof
 *          The function that describes the new data distribution.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE on error.
 *
 */
int CHAMELEON_Desc_Change_Distribution( cham_uplo_t      uplo,
                                        CHAM_desc_t     *desc,
                                        blkrankof_fct_t  new_get_rankof,
                                        void*            new_get_rankof_arg )
{
    int                 status;
    CHAM_context_t     *chamctxt;
    RUNTIME_sequence_t *sequence = NULL;

    chamctxt = chameleon_context_self();
    if (chamctxt == NULL) {
        chameleon_fatal_error("CHAMELEON_Desc_Change_Distribution", "CHAMELEON not initialized");
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    chameleon_sequence_create( chamctxt, &sequence );

    CHAMELEON_Desc_Change_Distribution_Async( uplo, desc, new_get_rankof, new_get_rankof_arg, sequence );

    RUNTIME_desc_flush( desc, sequence );
    chameleon_sequence_wait( chamctxt, sequence );
    status = sequence->status;
    chameleon_sequence_destroy( chamctxt, sequence );

    return status;
}

/**
 *****************************************************************************
 *
 * @ingroup Descriptor
 *
 * @brief Change the data distribution of the given matrix.
 *
 ******************************************************************************
 *
 * @param[in] uplo
 *          = ChamUpper: Upper triangle of A is moved; lower part is never referenced.
 *          = ChamLower: Lower triangle of A is moved; upper part is never referenced.
 *          = ChamUpperLower: The full matrix A is moved.
 *
 * @param[inout] desc The matrix descriptor to move around. On exit, the new
 *          distribution is registered.
 *
 * @param[in] new_get_rankof
 *          The function that describes the new data distribution.
 *
 ******************************************************************************
 *
 * @retval CHAMELEON_SUCCESS successful exit.
 * @retval CHAMELEON_ERR_ILLEGAL_VALUE on error.
 * @retval CHAMELEON_ERR_NOT_INITIALIZED if chameleon is not initialized.
 * @retval CHAMELEON_ERR_NOT_SUPPORTED if CHAMELEON_USE_MIGRATE is not enabled
 *
 */
int CHAMELEON_Desc_Change_Distribution_Async( cham_uplo_t         uplo,
                                              CHAM_desc_t        *desc,
                                              blkrankof_fct_t     new_get_rankof,
                                              void*               new_get_rankof_arg,
                                              RUNTIME_sequence_t *sequence )
{
    CHAM_context_t *chamctxt;
    int m, n, mmin, mmax;

    chamctxt = chameleon_context_self();
    if ( chamctxt == NULL ) {
        chameleon_error("CHAMELEON_Desc_Change_Distribution_Async", "CHAMELEON not initialized");
        sequence->status = CHAMELEON_ERR_NOT_INITIALIZED;
        return CHAMELEON_ERR_NOT_INITIALIZED;
    }

    /* Nothing to do if the new mapping is the same as the original one */
    if ( ( ( new_get_rankof     == desc->get_rankof_init    ) &&
           ( new_get_rankof_arg == desc->get_rankof_init_arg) ) ||
         ( RUNTIME_comm_size( chamctxt ) == 1 ) )
    {
        return CHAMELEON_SUCCESS;
    }

    if ( chamctxt->scheduler != RUNTIME_SCHED_STARPU ) {
        chameleon_error("CHAMELEON_Desc_Change_Distribution_Async", "Distribution change is only supported by StarPU");
        sequence->status = CHAMELEON_ERR_ILLEGAL_VALUE;
        return CHAMELEON_ERR_ILLEGAL_VALUE;
    }

    /* Check that the function is enabled */
#if !defined(CHAMELEON_USE_MIGRATE)
    {
        chameleon_error("CHAMELEON_Desc_Change_Distribution_Async", "Distribution change is only supported by StarPU");
        sequence->status = CHAMELEON_ERR_NOT_SUPPORTED;
        return CHAMELEON_ERR_NOT_SUPPORTED;
    }
#endif

    /* Update the get_rankof function and argument for the new one */
    desc->get_rankof_init     = new_get_rankof;
    desc->get_rankof_init_arg = new_get_rankof_arg;

    for ( n = 0; n < desc->nt; n++ ) {
        mmin = ( uplo == ChamLower ) ? chameleon_min( n,   desc->mt ) : 0;
        mmax = ( uplo == ChamUpper ) ? chameleon_min( n+1, desc->mt ) : desc->mt;
        for ( m = mmin; m < mmax; m++ ) {
            CHAM_tile_t *tile = desc->get_blktile( desc, m, n );
            int         rank  = new_get_rankof( desc, m, n );

            if ( rank != tile->rank ) {
                RUNTIME_data_migrate( sequence, desc, m, n, rank );
                tile->rank = rank;
            }
        }
    }

    return CHAMELEON_SUCCESS;
}
