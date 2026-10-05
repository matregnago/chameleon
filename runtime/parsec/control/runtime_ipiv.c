/**
 *
 * @file parsec/runtime_ipiv.c
 *
 * @copyright 2022-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon PaRSEC descriptor routines
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Matthieu Kuhn
 * @author Alycia Lisito
 * @author Florent Pruvost
 * @author Matteo Marcos
 * @date 2026-10-05
 *
 */
#include "chameleon_parsec.h"

/*
 * The keys are the tile indices of the full vector (offset by i/mb), the owner
 * of a key is the owner of the diagonal tile.
 */
static int
chameleon_parsec_ipiv_owner( const chameleon_parsec_vdc_t *vdc, int key )
{
    const CHAM_ipiv_t *ipiv = (const CHAM_ipiv_t *)(vdc->args);
    int m = key - ipiv->i / ipiv->mb;
    return ipiv->get_rankof( ipiv, m, m );
}

/* The ipiv family works in place in the user array when it is given */
static void *
chameleon_parsec_ipiv_userptr( const chameleon_parsec_vdc_t *vdc, int key )
{
    const CHAM_ipiv_t *ipiv = (const CHAM_ipiv_t *)(vdc->args);
    int m = key - ipiv->i / ipiv->mb;

    if ( ipiv->data == NULL ) {
        return NULL;
    }
    return ipiv->data + ipiv->i + m * ipiv->mb;
}

/**
 *  Create ipiv runtime structures
 */
void RUNTIME_ipiv_create( CHAM_ipiv_t *ipiv )
{
    chameleon_parsec_ipiv_t *pipiv;
    size_t size, size_last, size_perm;
    int    last;

    assert( ipiv );
    pipiv = calloc( 1, sizeof(chameleon_parsec_ipiv_t) );
    pipiv->ipiv = ipiv;

    last      = ipiv->m - (ipiv->mt - 1) * ipiv->mb;
    size      = sizeof(int) * ipiv->mb;
    size_last = sizeof(int) * last;
    size_perm = ipiv->withidx ? sizeof(int) * ( (ipiv->max_mt + 1) + 2 * ipiv->mb ) : sizeof(int) * ipiv->mb;

    chameleon_parsec_vdc_init( pipiv->vdc + ChamParsecIpiv, ipiv->mt, size, size_last,
                               chameleon_parsec_ipiv_owner, chameleon_parsec_ipiv_userptr,
                               NULL, ipiv );
    chameleon_parsec_vdc_init( pipiv->vdc + ChamParsecPerm, ipiv->mt, size_perm, size_perm,
                               chameleon_parsec_ipiv_owner, NULL, NULL, ipiv );
    chameleon_parsec_vdc_init( pipiv->vdc + ChamParsecInvp, ipiv->mt, size_perm, size_perm,
                               chameleon_parsec_ipiv_owner, NULL, NULL, ipiv );

    ipiv->ipiv = pipiv;
    ipiv->perm = pipiv->vdc + ChamParsecPerm;
    ipiv->invp = pipiv->vdc + ChamParsecInvp;
}

/**
 *  Destroy ipiv runtime structures
 */
void RUNTIME_ipiv_destroy( CHAM_ipiv_t *ipiv )
{
    chameleon_parsec_ipiv_t *pipiv = (chameleon_parsec_ipiv_t *)(ipiv->ipiv);
    int i;

    if ( pipiv == NULL ) {
        return;
    }
    for ( i = 0; i < 3; i++ ) {
        chameleon_parsec_vdc_fini( pipiv->vdc + i );
    }
    free( pipiv );
    ipiv->ipiv = NULL;
    ipiv->perm = NULL;
    ipiv->invp = NULL;
}

/*
 * The getters return the DTD tile without registering the collection for the
 * deferred flush: the codelets use chameleon_parsec_ipiv_tile() which does.
 */
void *RUNTIME_ipiv_getaddr( const CHAM_ipiv_t *ipiv, int m )
{
    return parsec_dtd_tile_of( (parsec_data_collection_t *)chameleon_parsec_ipiv_vdc( ipiv, ChamParsecIpiv ),
                               chameleon_parsec_ipiv_key( ipiv, m ) );
}

void *RUNTIME_ipiv_getperm( const CHAM_ipiv_t *ipiv, int m )
{
    return parsec_dtd_tile_of( (parsec_data_collection_t *)chameleon_parsec_ipiv_vdc( ipiv, ChamParsecPerm ),
                               chameleon_parsec_ipiv_key( ipiv, m ) );
}

void *RUNTIME_ipiv_getinvp( const CHAM_ipiv_t *ipiv, int m )
{
    return parsec_dtd_tile_of( (parsec_data_collection_t *)chameleon_parsec_ipiv_vdc( ipiv, ChamParsecInvp ),
                               chameleon_parsec_ipiv_key( ipiv, m ) );
}

/*
 * A flushed tile can not be reused before the wait of the sequence: the
 * flushes only register the collections, they are flushed at the wait.
 */
void RUNTIME_ipiv_flushone( RUNTIME_sequence_t *sequence,
                            CHAM_ipiv_e         which,
                            const CHAM_ipiv_t  *ipiv,
                            int                 m )
{
    RUNTIME_ipiv_flushall( sequence, which, ipiv );
    (void)m;
}

void RUNTIME_ipiv_flushall( RUNTIME_sequence_t *sequence,
                            CHAM_ipiv_e         which,
                            const CHAM_ipiv_t  *ipiv )
{
    parsec_taskpool_t *tp = (parsec_taskpool_t *)(sequence->schedopt);

    if ( which & CHAMIPIV_IPIV ) {
        chameleon_parsec_flush_defer( tp, (parsec_data_collection_t *)chameleon_parsec_ipiv_vdc( ipiv, ChamParsecIpiv ) );
    }
    if ( which & CHAMIPIV_PERM ) {
        chameleon_parsec_flush_defer( tp, (parsec_data_collection_t *)chameleon_parsec_ipiv_vdc( ipiv, ChamParsecPerm ) );
    }
    if ( which & CHAMIPIV_INVP ) {
        chameleon_parsec_flush_defer( tp, (parsec_data_collection_t *)chameleon_parsec_ipiv_vdc( ipiv, ChamParsecInvp ) );
    }
}

static inline int
CORE_ipiv_gather_parsec( parsec_execution_stream_t *context,
                         parsec_task_t             *this_task )
{
    int  node, ncols;
    int *dst, *src;

    parsec_dtd_unpack_args( this_task, &node, &ncols, &dst, &src );
    if ( dst != src ) {
        memcpy( dst, src, ncols * sizeof(int) );
    }

    (void)context;
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 *  Gather the ipiv array on the rank node. The user array is only accessed on
 *  node, where the task is executed.
 */
void RUNTIME_ipiv_gather( RUNTIME_sequence_t *sequence,
                          const CHAM_ipiv_t  *desc,
                          int                *ipiv,
                          int                 node )
{
    parsec_taskpool_t *tp = (parsec_taskpool_t *)(sequence->schedopt);
    RUNTIME_option_t   options;
    int64_t            mt = desc->mt;
    int64_t            mb = desc->mb;
    int                m;

    memset( &options, 0, sizeof(RUNTIME_option_t) );
    options.sequence = sequence;

    for ( m = 0; m < mt; m++, ipiv += mb ) {
        int ncols = ( m == (mt-1) ) ? desc->m - m * mb : mb;

        parsec_dtd_insert_task(
            tp, CORE_ipiv_gather_parsec, 0, PARSEC_DEV_CPU, "ipiv_gather",
            sizeof(int),    &node,  PARSEC_VALUE | PARSEC_AFFINITY,
            sizeof(int),    &ncols, PARSEC_VALUE,
            sizeof(int *),  &ipiv,  PARSEC_VALUE,
            PASSED_BY_REF,  chameleon_parsec_ipiv_tile( &options, desc, ChamParsecIpiv, m ),
                            chameleon_parsec_ipiv_arena( desc, ChamParsecIpiv, m ) | PARSEC_INPUT,
            PARSEC_DTD_ARG_END );
    }
}
