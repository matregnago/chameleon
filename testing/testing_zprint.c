/**
 *
 * @file testing_zprint.c
 *
 * @copyright 2019-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon descriptor printing and recursive flat-view testing
 *
 * @version 1.4.0
 * @author Lucas Barros de Assis
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @author Lionel Eyraud-Dubois
 * @date 2025-12-19
 * @precisions normal z -> c d s
 *
 */
#include <chameleon.h>
#include <chameleon_lapack.h>
#include "testings.h"
#include "testing_zcheck.h"

static int
testing_zprint_check_desc( run_arg_list_t *args, CHAM_desc_t *desc,
                           unsigned long long int seed, int check )
{
    int rc;

    CHAMELEON_Desc_Print( desc );
    rc = CHAMELEON_zplrnt_Tile( desc, seed );
    if ( ( rc == CHAMELEON_SUCCESS ) && check ) {
        rc = check_zgenerate( args, ChamGeneral, ChamUpperLower, desc, seed, 0. );
    }

    return rc;
}

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
/**
 * @brief Select a useful exact refinement of a recursive tile size.
 *
 * The largest proper divisor keeps the number of generated leaf tiles small.
 * Prime tile sizes are left unchanged instead of being split into individual
 * matrix elements merely for this descriptor test.
 */
static int
testing_zprint_child_blocking( int parent )
{
    int factor;

    for ( factor = 2; factor <= parent / factor; factor++ ) {
        if ( ( parent % factor ) == 0 ) {
            return parent / factor;
        }
    }

    return parent;
}

static int
testing_zprint_create_recursive( CHAM_desc_t **descptr, const char *name,
                                  void *mat, int M, int N, int P, int Q,
                                  cham_rec_t rec, int rarg,
                                  const int *mbs, const int *nbs, int dist_level )
{
    cham_data_dist_t data_dist = {
        .get_distrib        = (datadist_access_fct_t)chameleon_get_2d_block_cyclic,
        .distrib_array_size = 2,
        .distrib            = { P, Q }
    };
    CHAM_desc_recursion_t recursion = {
        .kind       = rec,
        .arg        = rarg,
        .mbs        = mbs,
        .nbs        = nbs,
        .dist_level = dist_level
    };
    CHAM_desc_create_t create = {
        .name = name,
        .layout = {
            .dtyp = ChamComplexDouble,
            .mb   = mbs[0],
            .nb   = nbs[0],
            .lm   = M,
            .ln   = N,
            .i    = 0,
            .j    = 0,
            .m    = M,
            .n    = N
        },
        .storage = {
            .mat            = mat,
            .get_blkaddr    = NULL,
            .get_blkldd     = NULL,
            .get_rankof     = NULL,
            .get_rankof_arg = NULL
        },
        .data_dist = &data_dist,
        .recursive = &recursion
    };

    return CHAMELEON_Desc_CreateEx( descptr, &create );
}

static int
testing_zprint_recursive_case( run_arg_list_t *args, const char *name,
                               void *mat, int M, int N, int P, int Q,
                               cham_rec_t rec, int rarg,
                               const int *mbs, const int *nbs, int dist_level,
                               unsigned long long int seed, int check )
{
    CHAM_desc_t *desc = NULL;
    CHAM_desc_t *view = NULL;
    int          depth;
    int          rc;

    for ( depth = 0; ( mbs[depth] > 0 ) && ( nbs[depth] > 0 ); depth++ ) {
    }

    if ( CHAMELEON_Comm_rank() == 0 ) {
        fprintf( stdout, "--- %s (dist_level=%d, leaf=%dx%d) ---\n",
                 name, dist_level, mbs[depth-1], nbs[depth-1] );
    }

    rc = testing_zprint_create_recursive( &desc, name, mat, M, N, P, Q, rec, rarg,
                                          mbs, nbs, dist_level );
    if ( rc == CHAMELEON_SUCCESS ) {
        rc = testing_zprint_check_desc( args, desc, seed, 0 );
    }
    if ( rc == CHAMELEON_SUCCESS ) {
        rc = CHAMELEON_Desc_Create_FlatView( &view, desc,
                                             mbs[depth-1], nbs[depth-1], "flat" );
    }
    if ( rc == CHAMELEON_SUCCESS ) {
        CHAMELEON_Desc_Print( view );
    }
    if ( ( rc == CHAMELEON_SUCCESS ) && check ) {
        rc = check_zgenerate( args, ChamGeneral, ChamUpperLower, view, seed, 0. );
    }

    if ( view != NULL ) {
        CHAMELEON_Desc_Destroy( &view );
    }
    if ( desc != NULL ) {
        CHAMELEON_Desc_Destroy( &desc );
    }

    return rc;
}
#endif

int
testing_zprint_desc( run_arg_list_t *args, int check )
{
    unsigned long long int seed = run_arg_get_int( args, "seedA", 0x1234 );
    int                    P    = parameters_getvalue_int( "P" );
    int                    Q    = parameters_compute_q( P );
    int                    M    = run_arg_get_int( args, "M", 48 );
    int                    N    = run_arg_get_int( args, "N", M );
    int                    LDA  = run_arg_get_int( args, "LDA", M );
    int                    nb   = run_arg_get_nb( args );
    CHAM_desc_t           *desc = NULL;
    int                    hres = 0;
    int                    rc;

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
    intptr_t    mtxfmt = parameters_getvalue_int( "mtxfmt" );
    cham_rec_t  rec    = run_arg_get_rec( args, "rec", ChamRecNone );
    int         rarg   = run_arg_get_int( args, "rarg", 1 );
    int         child1 = testing_zprint_child_blocking( nb );
    int         child2 = testing_zprint_child_blocking( child1 );

    const int   levels0[] = {        nb, 0 };
    const int   levels1[] = { 2 * nb, nb, 0 };
    const int   levels2[] = { 4 * nb, 2 * nb, nb, 0 };
    const int   levels3[] = { 4 * nb, 2 * nb, nb, child1, 0 };
    const int   levels4[] = { 4 * nb, 2 * nb, nb, child1, child2, 0 };
    const int  *levels[]  = { levels0, levels1, levels2, levels3, levels4 };
    const int   dist[]    = { 0, 1, 2, 2, 2 };
    const char *names[]   = {
        "recursive_dist",
        "recursive_above1",
        "recursive_above2",
        "recursive_below1",
        "recursive_below2"
    };

    void       *recursive_mat = (void *)(-mtxfmt);
    const char *alloc_name    = ( mtxfmt == 0 ) ? "global" : "tile";
    int         i;
#endif

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    if ( CHAMELEON_Comm_rank() == 0 ) {
        fprintf( stdout, "--- Tile layout ---\n" );
    }
    rc = CHAMELEON_Desc_Create( &desc, CHAMELEON_MAT_ALLOC_TILE,
                                ChamComplexDouble, nb, nb, nb * nb,
                                LDA, N, 0, 0, M, N, P, Q );
    if ( rc == CHAMELEON_SUCCESS ) {
        rc = testing_zprint_check_desc( args, desc, seed, check );
    }
    hres += ( rc == CHAMELEON_SUCCESS ) ? 0 : 1;
    if ( desc != NULL ) {
        CHAMELEON_Desc_Destroy( &desc );
    }

    if ( CHAMELEON_Comm_rank() == 0 ) {
        fprintf( stdout, "--- LAPACK layout ---\n" );
    }
    rc = CHAMELEON_Desc_Create_User(
        &desc, CHAMELEON_MAT_ALLOC_GLOBAL, ChamComplexDouble,
        nb, nb, nb * nb, LDA, N, 0, 0, M, N, P, Q,
        chameleon_getaddr_cm, chameleon_getblkldd_cm, NULL, NULL );
    if ( rc == CHAMELEON_SUCCESS ) {
        rc = testing_zprint_check_desc( args, desc, seed, check );
    }
    hres += ( rc == CHAMELEON_SUCCESS ) ? 0 : 1;
    if ( desc != NULL ) {
        CHAMELEON_Desc_Destroy( &desc );
    }

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
    if ( rec != ChamRecNone ) {
        for ( i = 0; i < 5; i++ ) {
            char name[64];

            snprintf( name, sizeof(name), "%s_%s", names[i], alloc_name );
            rc = testing_zprint_recursive_case( args, name, recursive_mat, M, N, P, Q,
                                                rec, rarg, levels[i], levels[i], dist[i],
                                                seed, check );
            hres += ( rc == CHAMELEON_SUCCESS ) ? 0 : 1;
        }
    }
#endif

    run_arg_add_fixdbl( args, "time", 1. );
    run_arg_add_fixdbl( args, "gflops", 1. );
    return hres;
}

testing_t   test_zprint;
const char *zprint_params[] = {
    "mtxfmt", "nb", "m", "n", "lda", "seedA", "rec", "rarg", NULL
};
const char *zprint_output[] = { NULL };
const char *zprint_outchk[] = { "RETURN", NULL };

/**
 * @brief Testing registration function
 */
void testing_zprint_init( void ) __attribute__( ( constructor ) );
void
testing_zprint_init( void )
{
    test_zprint.name      = "zprint";
    test_zprint.helper    = "Print and validate descriptors";
    test_zprint.params    = zprint_params;
    test_zprint.output    = zprint_output;
    test_zprint.outchk    = zprint_outchk;
    test_zprint.fptr_desc = testing_zprint_desc;
    test_zprint.fptr_std  = NULL;
    test_zprint.next      = NULL;

    testing_register( &test_zprint );
}
