/**
 *
 * @file descriptor.h
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon descriptor header
 *
 * @version 1.4.0
 * @author Jakub Kurzak
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @author Guillaume Sylvand
 * @author Raphael Boucherie
 * @author Samuel Thibault
 * @author Lionel Eyraud-Dubois
 * @author Alycia Lisito
 * @author Pierre Esterie
 * @author Matteo Marcos
 * @author Brieuc Nicolas
 * @date 2025-12-19
 *
 */
#ifndef _chameleon_descriptor_h_
#define _chameleon_descriptor_h_

#include "chameleon/config.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "chameleon/struct.h"
#include "control/auxiliary.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 */
static inline int chameleon_desc_mat_alloc( CHAM_desc_t *desc )
{
    size_t size = (size_t)(desc->llm) * (size_t)(desc->lln)
        * (size_t)CHAMELEON_Element_Size(desc->dtyp);
    if ((desc->mat = RUNTIME_malloc(size)) == NULL) {
        chameleon_error("chameleon_desc_mat_alloc", "malloc() failed");
        return CHAMELEON_ERR_OUT_OF_RESOURCES;
    }

    /* The matrix has already been registered by the Runtime alloc */
    desc->register_mat = 0;

    return CHAMELEON_SUCCESS;
}

/**
 *
 */
static inline int chameleon_desc_mat_free( CHAM_desc_t *desc )
{
    if ( (desc->mat       != NULL) &&
         (desc->use_mat   == 1   ) &&
         (desc->alloc_mat == 1   ) )
    {
        size_t size = (size_t)(desc->llm) * (size_t)(desc->lln)
            * (size_t)CHAMELEON_Element_Size(desc->dtyp);

        RUNTIME_free(desc->mat, size);
        desc->mat = NULL;
    }

    if ( desc->tiles ) {
#if defined(CHAMELEON_KERNELS_TRACE)
        CHAM_tile_t *tile = desc->tiles;
        int ii, jj;
        for( jj=0; jj<desc->lnt; jj++ ) {
            for( ii=0; ii<desc->lmt; ii++, tile++ ) {
                if ( tile->name ) {
                    free( tile->name );
                }
            }
        }
#endif
        free( desc->tiles );
    }
    free( desc->data_dist );
    return CHAMELEON_SUCCESS;
}

/**
 *  Internal routines
 */
static inline int chameleon_getrankof_tile(const CHAM_desc_t *desc, int m, int n) {
    CHAM_tile_t *tile = desc->get_blktile( desc, m, n );
    assert( tile != NULL );
    return tile->rank;
}

/**
 * @brief Compute the deterministic MPI tag of a descriptor tile handle.
 *
 * @param[in] desc
 *          Descriptor owning the tile handle.
 *
 * @param[in] mm
 *          Tile row in the full tile grid represented by @p desc. This is not
 *          necessarily the row in the distributed MPI-tag grid.
 *
 * @param[in] nn
 *          Tile column in the full tile grid represented by @p desc. This is
 *          not necessarily the column in the distributed MPI-tag grid.
 *
 * @param[in] fltshift
 *          Precision-copy index: zero selects the primary handle, while one
 *          and two select the conversion handles.
 *
 * The descriptor MPI-tag origin lifts the input coordinates into the tag grid.
 * Classic descriptors and distributed recursive levels use the global tile
 * grid. Levels below the distributed level use a compact grid local to their
 * single-owner subtree.
 *
 * @return The MPI tag, or -1 when handles at this level do not use MPI tags.
 */
static inline int64_t
chameleon_desc_get_mpi_tag( const CHAM_desc_t *desc, int64_t mm, int64_t nn,
                            int fltshift )
{
    int64_t index;
    int64_t tag;

    if ( !desc->mpitag_use ) {
        return -1;
    }
    assert( desc->mpitag_ld > 0 );
    index = ( desc->dist_jt + nn ) * desc->mpitag_ld + desc->dist_it + mm;
    tag = desc->mpitag + index * desc->mpitag_tile_span;
    if ( fltshift != 0 ) {
        tag += desc->mpitag_tree_span + fltshift - 1;
    }
    return tag;
}

int chameleon_desc_init( const CHAM_context_t *chamctxt,
                         CHAM_desc_t *desc, const char *name, void *mat,
                         cham_flttype_t dtyp, int mb, int nb,
                         int lm, int ln, int m, int n, int p, int q,
                         void* (*get_blkaddr)( const CHAM_desc_t*, int, int ),
                         int   (*get_blkldd) ( const CHAM_desc_t*, int      ),
                         int   (*get_rankof) ( const CHAM_desc_t*, int, int ),
                         void* get_rankof_arg );

int chameleon_desc_init_base( CHAM_desc_t *desc, int myrank, const char *name,
                              const CHAM_desc_storage_t *storage,
                              const CHAM_desc_layout_t  *layout );

void chameleon_desc_init_tiles( CHAM_desc_t *desc, blkrankof_fct_t rankof );
void chameleon_desc_init_2d_distribution( CHAM_desc_t *desc, int p, int q );
void chameleon_desc_init_2d_distribution_with_offset( CHAM_desc_t *desc, int p, int q,
                                                      int dist_it, int dist_jt );
int  chameleon_desc_init_storage( const CHAM_context_t *chamctxt, CHAM_desc_t *desc, void *mat );
void chameleon_desc_register( CHAM_desc_t *desc );
void chameleon_desc_register_with_offset( CHAM_desc_t *desc );
int  chameleon_recdesc_init( const CHAM_context_t *chamctxt,
                             CHAM_desc_t *desc, const char *name, void *mat,
                             cham_flttype_t dtyp, cham_rec_t rec, int rarg,
                             const int *mbs, const int *nbs,
                             int lm, int ln, int m, int n, int p, int q,
                             int dist_level,
                             blkaddr_fct_t get_blkaddr,
                             blkldd_fct_t get_blkldd,
                             blkrankof_fct_t get_rankof,
                             void *get_rankof_arg );
int  chameleon_desc_create_recursive( CHAM_desc_t **desc, const CHAM_desc_create_t *args );

static inline int
chameleon_desc_init_2dtile( const CHAM_context_t *chamctxt,
                            CHAM_desc_t *desc, const char *name,
                            cham_flttype_t dtyp, int mb, int nb,
                            int m, int n, int p, int q )
{
    return chameleon_desc_init( chamctxt, desc, name, CHAMELEON_MAT_ALLOC_TILE,
                                dtyp, mb, nb, m, n, m, n, p, q,
                                NULL, NULL, NULL, NULL );
}

static inline int
chameleon_desc_init_2dlap( const CHAM_context_t *chamctxt,
                           CHAM_desc_t *desc, const char *name,
                           cham_flttype_t dtyp, int mb, int nb,
                           int m, int n, int p, int q )
{
    return chameleon_desc_init( chamctxt, desc, name, CHAMELEON_MAT_ALLOC_GLOBAL,
                                dtyp, mb, nb, m, n, m, n, p, q,
                                NULL, NULL, NULL, NULL );
}

static inline int
chameleon_desc_init_local( const CHAM_context_t *chamctxt,
                           CHAM_desc_t *desc, const char *name,
                           cham_flttype_t dtyp, int mb, int nb, int m, int n )
{
    return chameleon_desc_init( chamctxt, desc, name, CHAMELEON_MAT_ALLOC_GLOBAL,
                                dtyp, mb, nb, m, n, m, n, 1, 1,
                                NULL, NULL, NULL, NULL );
}

CHAM_desc_t* chameleon_desc_submatrix( CHAM_desc_t *descA, int i, int j, int m, int n );
int          chameleon_desc_check    ( const CHAM_desc_t *desc );
void         chameleon_desc_destroy  ( CHAM_desc_t *desc );
void         chameleon_desc_destroy_submit( CHAM_desc_t *desc, const RUNTIME_sequence_t *sequence );

int chameleon_desc_has_recursive_tiles( const CHAM_desc_t *desc );
int chameleon_desc_has_square_recursive_tiling( const CHAM_desc_t *desc );
int chameleon_desc_is_recursive_panel( const CHAM_desc_t *desc );

int chameleon_ipiv_init( CHAM_ipiv_t *ipiv, cham_side_t side, int mb, int m,
                         int withidx, int max_m, int max_mt,
                         int p, int np, void *data,
                         blkrankof_ipiv_fct_t get_rankof );
void chameleon_ipiv_destroy( CHAM_ipiv_t *ipiv );

int chameleon_pivot_init( CHAM_desc_pivot_t *pivot,
                          const CHAM_desc_t *desc );
int chameleon_pivot_destroy( CHAM_desc_pivot_t *pivot );
int chameleon_pivot_destroy_submit( CHAM_desc_pivot_t  *pivot,
                                    RUNTIME_sequence_t *sequence );

/**
 *  Internal function to return address of block (m,n) with m,n = block indices
 */
inline static size_t chameleon_getaddr_cm_offset( const CHAM_desc_t *A, int m, int n, size_t ld )
{
    size_t mm = m + A->i / A->mb;
    size_t nn = n + A->j / A->nb;
    size_t offset = 0;

#if defined(CHAMELEON_USE_MPI)
    assert( A->myrank == A->get_rankof( A, mm, nn ) );
    mm = mm / chameleon_desc_datadist_get_iparam(A, 0);
    nn = nn / chameleon_desc_datadist_get_iparam(A, 1);
#endif

    offset = (size_t)(ld * A->nb) * nn + (size_t)(A->mb) * mm;
    return offset;
}

/**
 *  Internal function to return address of block (m,n) with m,n = block indices
 */
inline static CHAM_tile_t *chameleon_desc_gettile(const CHAM_desc_t *A, int m, int n )
{
    size_t mm = m + A->i / A->mb;
    size_t nn = n + A->j / A->nb;
    size_t offset = 0;

    assert( A->tiles != NULL );

    offset = A->lmt * nn + mm;
    return A->tiles + offset;
}

/**
 *  Internal function to return address of element A(m,n) with m,n = matrix indices
 */
inline static void* chameleon_geteltaddr(const CHAM_desc_t *A, int m, int n, int eltsize) // Not used anywhere ?!
{
    size_t mm = (m + A->i)/A->mb;
    size_t nn = (n + A->j)/A->nb;
    size_t offset = 0;

#if defined(CHAMELEON_USE_MPI)
    assert( A->myrank == A->get_rankof( A, mm, nn) );
    mm = mm / chameleon_desc_datadist_get_iparam(A, 0);
    nn = nn / chameleon_desc_datadist_get_iparam(A, 1);
#endif

    if (mm < (size_t)(A->llm1)) {
        if (nn < (size_t)(A->lln1))
            offset = A->bsiz*(mm+A->llm1*nn) + m%A->mb + A->mb*(n%A->nb);
        else
            offset = A->A12 + (A->mb*(A->lln%A->nb)*mm) + m%A->mb + A->mb*(n%A->nb);
    }
    else {
        if (nn < (size_t)(A->lln1))
            offset = A->A21 + ((A->llm%A->mb)*A->nb*nn) + m%A->mb + (A->llm%A->mb)*(n%A->nb);
        else
            offset = A->A22 + m%A->mb  + (A->llm%A->mb)*(n%A->nb);
    }
    return (void*)((intptr_t)A->mat + (offset*eltsize) );
}

/**
 * Detect if the tile is local or not
 */
inline static int chameleon_desc_islocal( const CHAM_desc_t *A, int m, int n )
{
#if defined(CHAMELEON_USE_MPI)
    int rank = A->get_rankof(A, m, n);
    return ( ( rank == A->myrank ) || ( rank == CHAMELEON_MPI_WITH_ME ) );
#else
    (void)A; (void)m; (void)n;
    return 1;
#endif /* defined(CHAMELEON_USE_MPI) */
}

/**
 * Declare data accesses of codelets using these macros, for instance:
 * CHAMELEON_BEGIN_ACCESS_DECLARATION
 * CHAMELEON_ACCESS_R(A, Am, An)
 * CHAMELEON_ACCESS_R(B, Bm, Bn)
 * CHAMELEON_ACCESS_RW(C, Cm, Cn)
 * CHAMELEON_END_ACCESS_DECLARATION
 */
#define CHAMELEON_BEGIN_ACCESS_DECLARATION {                    \
    unsigned __chameleon_need_exec = 0;                         \
    unsigned __chameleon_need_submit = options->forcesub;       \
    RUNTIME_BEGIN_ACCESS_DECLARATION

#define CHAMELEON_ACCESS_R(A, Am, An) do {                              \
        if (chameleon_desc_islocal(A, Am, An)) __chameleon_need_submit = 1; \
        RUNTIME_ACCESS_R(A, Am, An);                                    \
    } while(0)

#define CHAMELEON_ACCESS_W(A, Am, An) do {              \
        if (chameleon_desc_islocal(A, Am, An)) {        \
            __chameleon_need_exec = 1;                  \
            __chameleon_need_submit = 1;                \
        }                                               \
        RUNTIME_ACCESS_W(A, Am, An);                    \
    } while(0)

#define CHAMELEON_ACCESS_RW(A, Am, An) do {             \
        if (chameleon_desc_islocal(A, Am, An)) {        \
            __chameleon_need_exec = 1;                  \
            __chameleon_need_submit = 1;                \
        }                                               \
        RUNTIME_ACCESS_RW(A, Am, An);                   \
    } while(0)

#define CHAMELEON_RANK_CHANGED(rank) do {       \
        __chameleon_need_submit = 1;            \
        RUNTIME_RANK_CHANGED(rank);             \
    } while (0)

#define CHAMELEON_END_ACCESS_DECLARATION        \
    RUNTIME_END_ACCESS_DECLARATION;             \
    if (!__chameleon_need_submit) return;       \
    (void)__chameleon_need_exec;                \
}

/**
 * @brief Helpers to iterate over the 2D block cyclic indices of a process.
 *
 * Without replicated submission, a process only iterates over its own
 * indices: my, my+np, my+2np, ... With replicated submission, every process
 * iterates over all of them: 0, 1, 2, ...
 *
 *   for ( m = chameleon_2dbc_first( repl, myp, P ); m < MT; m += chameleon_2dbc_stride( repl, P ) )
 *
 * chameleon_2dbc_last() bounds the loops over the processes themselves:
 *
 *   for ( p = chameleon_2dbc_first( repl, myp, P ); p < chameleon_2dbc_last( repl, myp, P ); p++ )
 */
static inline int chameleon_2dbc_first( int repl, int my, int np ) {
    (void)np;
    return repl ? 0 : my;
}

static inline int chameleon_2dbc_stride( int repl, int np ) {
    return repl ? 1 : np;
}

static inline int chameleon_2dbc_last( int repl, int my, int np ) {
    return repl ? np : my + 1;
}

#ifdef __cplusplus
}
#endif

#endif /* _chameleon_descriptor_h_ */
