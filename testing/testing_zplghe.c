/**
 *
 * @file testing_zplghe.c
 *
 * @copyright 2019-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zplghe testing
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2025-12-18
 * @precisions normal z -> c
 *
 */
#include <chameleon.h>
#include <chameleon_lapack.h>
#include "testings.h"
#include "testing_zcheck.h"

int
testing_zplghe_desc( run_arg_list_t *args, int check )
{
    testdata_t test_data = TESTINGS_TESTDATA_INITIALIZER;
    int        hres      = 0;

    /* Read arguments */
    int                    async = parameters_getvalue_int( "async" );
    int                    nb    = run_arg_get_nb( args );
    int                    N     = run_arg_get_int( args, "N", 1000 );
    int                    M     = run_arg_get_int( args, "M", N );
    int                    LDA   = run_arg_get_int( args, "LDA", M );
    unsigned long long int seedA = run_arg_get_int( args, "seedA", testing_ialea() );
    double                 bump  = (double)M;

    /* Descriptors */
    CHAM_desc_t *descA;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrix */
    parameters_desc_create( "A", &descA, ChamComplexDouble, nb, nb, LDA, N, M, N );

    /* Randomly generates the hermitian matrix */
    testing_start( &test_data );
    if ( async ) {
        hres = CHAMELEON_zplghe_Tile_Async( bump, ChamUpperLower, descA, seedA,
                                            test_data.sequence, &test_data.request );
        CHAMELEON_Desc_Flush( descA, test_data.sequence );
    }
    else {
        hres = CHAMELEON_zplghe_Tile( bump, ChamUpperLower, descA, seedA );
    }
    testing_stop( &test_data, 0 );
    test_data.hres = hres;

    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the solution */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {

        hres += check_zgenerate( args, ChamHermitian, ChamUpperLower, descA, seedA, bump );

    }

    parameters_desc_destroy( &descA );

    return hres;
}

int
testing_zplghe_std( run_arg_list_t *args, int check )
{
    testdata_t test_data = TESTINGS_TESTDATA_INITIALIZER;
    int        hres      = 0;

    /* Read arguments */
    int                    nb    = run_arg_get_nb( args );
    int                    N     = run_arg_get_int( args, "N", 1000 );
    int                    M     = run_arg_get_int( args, "M", N );
    int                    LDA   = run_arg_get_int( args, "LDA", M );
    unsigned long long int seedA = run_arg_get_int( args, "seedA", testing_ialea() );
    double                 bump  = (double)M;

    /* Descriptors */
    CHAMELEON_Complex64_t *A;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrix */
    A = malloc( sizeof(CHAMELEON_Complex64_t) * LDA*N );

    /* Randomly generates the hermitian matrix */
    testing_start( &test_data );
    hres = CHAMELEON_zplghe( bump, ChamUpperLower, N, A, LDA, seedA );
    testing_stop( &test_data, 0 );
    test_data.hres = hres;

    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the solution */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {

        hres += check_zgenerate_std( args, ChamHermitian, ChamUpperLower, M, N, A, LDA, seedA, bump );

    }

    free( A );

    return hres;
}

testing_t   test_zplghe;
const char *zplghe_params[] = { "mtxfmt", "nb", "uplo", "m", "n", "lda", "seedA", "bump", NULL };
const char *zplghe_output[] = { NULL };
const char *zplghe_outchk[] = { "RETURN", NULL };

/**
 * @brief Testing registration function
 */
void testing_zplghe_init( void ) __attribute__( ( constructor ) );
void
testing_zplghe_init( void )
{
    test_zplghe.name   = "zplghe";
    test_zplghe.helper = "Random Hermitian Matrix Generation";
    test_zplghe.params = zplghe_params;
    test_zplghe.output = zplghe_output;
    test_zplghe.outchk = zplghe_outchk;
    test_zplghe.fptr_desc = testing_zplghe_desc;
    test_zplghe.fptr_std  = testing_zplghe_std;
    test_zplghe.next   = NULL;

    testing_register( &test_zplghe );
}

