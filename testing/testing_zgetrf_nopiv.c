/**
 *
 * @file testing_zgetrf_nopiv.c
 *
 * @copyright 2019-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zgetrf_nopiv testing
 *
 * @version 1.4.0
 * @author Lucas Barros de Assis
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @author Lucas Barros De Assis
 * @author Matthieu Kuhn
 * @date 2025-12-19
 * @precisions normal z -> c d s
 *
 */
#include <chameleon.h>
#include <chameleon_lapack.h>
#include "testings.h"
#include "testing_zcheck.h"
#include <chameleon/flops.h>

int
testing_zgetrf_nopiv_desc( run_arg_list_t *args, int check )
{
    testdata_t test_data = TESTINGS_TESTDATA_INITIALIZER;
    int        hres      = 0;

    /* Read arguments */
    int                   async = parameters_getvalue_int( "async" );
    int                   nb    = run_arg_get_nb( args );
    int                   N     = run_arg_get_int( args, "N", 1000 );
    int                   M     = run_arg_get_int( args, "M", N );
    int                   LDA   = run_arg_get_int( args, "LDA", M );
    int                   seedA = run_arg_get_int( args, "seedA", testing_ialea() );
    CHAMELEON_Complex64_t bump  = run_arg_get_complex64( args, "bump", (CHAMELEON_Complex64_t)N );
    cham_rec_t            rec   = run_arg_get_rec( args, "rec", ChamRecNone );
    int                   rarg  = run_arg_get_int( args, "rarg", 1 );
    int                   l1    = run_arg_get_int( args, "l1", 0 );
    int                   leaf  = nb;

    /* Descriptors */
    CHAM_desc_t *descA = NULL;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrices */
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
    if ( rec != ChamRecNone ) {
        int list_mb[] = { M,  nb, 0 };
        int list_nb[] = { nb, nb, 0 };
        CHAM_desc_recursion_t recursion = {
            .kind       = rec,
            .arg        = rarg,
            .mbs        = list_mb,
            .nbs        = list_nb,
            .dist_level = 1
        };

        if ( l1 > 0 ) {
            list_mb[0]          = nb;
            list_mb[1]          = l1;
            list_nb[1]          = l1;
            recursion.dist_level = 0;
            leaf                 = l1;
        }

        hres = parameters_recdesc_create( "A", &descA, ChamComplexDouble,
                                          LDA, N, M, N,
                                          &recursion );
    }
    else
#endif
    {
        hres = parameters_desc_create( "A", &descA, ChamComplexDouble,
                                       nb, nb, LDA, N, M, N );
    }
    if ( hres != CHAMELEON_SUCCESS ) {
        return 1;
    }

    /* Fills the matrix with random values */
    CHAMELEON_zplgtr_Tile( 0,    ChamUpper, descA, seedA   );
    CHAMELEON_zplgtr_Tile( bump, ChamLower, descA, seedA+1 );

    /* Calculates the solution */
    testing_start( &test_data );
    if ( async ) {
        hres = CHAMELEON_zgetrf_nopiv_Tile_Async( descA, NULL, test_data.sequence, &test_data.request );
        CHAMELEON_Desc_Flush( descA, test_data.sequence );
    }
    else {
        hres = CHAMELEON_zgetrf_nopiv_Tile( descA );
    }
    test_data.hres = hres;
    testing_stop( &test_data, flops_zgetrf( M, N ) );
    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the factorisation and residue */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {
        CHAM_desc_t *descA0 = NULL;
        CHAM_desc_t *descLU = descA;

        hres = parameters_desc_create( "A0", &descA0, ChamComplexDouble,
                                       leaf, leaf, LDA, N, M, N );
        if ( hres == CHAMELEON_SUCCESS ) {
            CHAMELEON_zplgtr_Tile( 0,    ChamUpper, descA0, seedA   );
            CHAMELEON_zplgtr_Tile( bump, ChamLower, descA0, seedA+1 );
        }

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
        if ( ( hres == CHAMELEON_SUCCESS ) && ( rec != ChamRecNone ) ) {
            hres = CHAMELEON_Desc_Create_FlatView( &descLU, descA, leaf, leaf, "LU" );
        }
#endif

        if ( hres == CHAMELEON_SUCCESS ) {
            hres += check_zxxtrf( args, ChamGeneral, ChamUpperLower, descA0, descLU );
        }

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
        if ( ( descLU != descA ) && ( descLU != NULL ) ) {
            CHAMELEON_Desc_Destroy( &descLU );
        }
#endif
        if ( descA0 != NULL ) {
            CHAMELEON_Desc_Destroy( &descA0 );
        }
    }

    parameters_desc_destroy( &descA );

    (void)rec;
    (void)rarg;
    (void)l1;
    (void)leaf;
    return hres;
}

int
testing_zgetrf_nopiv_std( run_arg_list_t *args, int check )
{
    testdata_t test_data = TESTINGS_TESTDATA_INITIALIZER;
    int        hres      = 0;

    /* Read arguments */
    int                   nb    = run_arg_get_nb( args );
    int                   N     = run_arg_get_int( args, "N", 1000 );
    int                   M     = run_arg_get_int( args, "M", N );
    int                   LDA   = run_arg_get_int( args, "LDA", M );
    int                   seedA = run_arg_get_int( args, "seedA", testing_ialea() );
    CHAMELEON_Complex64_t bump  = run_arg_get_complex64( args, "bump", (CHAMELEON_Complex64_t)N );

    /* Descriptors */
    CHAMELEON_Complex64_t *A;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrices */
    A = malloc( sizeof(CHAMELEON_Complex64_t) * LDA*N );

    /* Fills the matrix with random values */
    CHAMELEON_zplgtr( 0,    ChamUpper, M, N, A, LDA, seedA   );
    CHAMELEON_zplgtr( bump, ChamLower, M, N, A, LDA, seedA+1 );

    /* Calculates the solution */
    testing_start( &test_data );
    hres = CHAMELEON_zgetrf_nopiv( M, N, A, LDA );
    test_data.hres = hres;
    testing_stop( &test_data, flops_zgetrf( M, N ) );
    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the factorisation and residue */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {
        CHAMELEON_Complex64_t *A0 = malloc( sizeof(CHAMELEON_Complex64_t) * LDA*N );
        CHAMELEON_zplgtr( 0,    ChamUpper, M, N, A0, LDA, seedA   );
        CHAMELEON_zplgtr( bump, ChamLower, M, N, A0, LDA, seedA+1 );

        hres += check_zxxtrf_std( args, ChamGeneral, ChamUpperLower, M, N, A0, A, LDA );

        free( A0 );
    }

    free( A );

    return hres;
}

testing_t   test_zgetrf_nopiv;
const char *zgetrf_nopiv_params[] = {
    "mtxfmt", "nb", "m", "n", "lda", "seedA", "bump", "rec", "rarg", "l1", NULL
};
const char *zgetrf_nopiv_output[] = { NULL };
const char *zgetrf_nopiv_outchk[] = { "||A||", "||A-fact(A)||", "RETURN", NULL };

/**
 * @brief Testing registration function
 */
void testing_zgetrf_nopiv_init( void ) __attribute__( ( constructor ) );
void
testing_zgetrf_nopiv_init( void )
{
    test_zgetrf_nopiv.name   = "zgetrf_nopiv";
    test_zgetrf_nopiv.helper = "General factorization (LU without pivoting)";
    test_zgetrf_nopiv.params = zgetrf_nopiv_params;
    test_zgetrf_nopiv.output = zgetrf_nopiv_output;
    test_zgetrf_nopiv.outchk = zgetrf_nopiv_outchk;
    test_zgetrf_nopiv.fptr_desc = testing_zgetrf_nopiv_desc;
    test_zgetrf_nopiv.fptr_std  = testing_zgetrf_nopiv_std;
    test_zgetrf_nopiv.next   = NULL;

    testing_register( &test_zgetrf_nopiv );
}
