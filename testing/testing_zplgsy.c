/**
 *
 * @file testing_zplgsy.c
 *
 * @copyright 2019-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zplgsy testing
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2025-12-18
 * @precisions normal z -> c d s
 *
 */
#include <chameleon.h>
#include <chameleon_lapack.h>
#include "testings.h"
#include "testing_zcheck.h"

int
testing_zplgsy_desc( run_arg_list_t *args, int check )
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
    CHAMELEON_Complex64_t bump  = run_arg_get_complex64( args, "bump", (CHAMELEON_Complex64_t)N );

    /* Descriptors */
    CHAM_desc_t *descA;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrix */
    parameters_desc_create( "A", &descA, ChamComplexDouble, nb, nb, LDA, N, M, N );

    /* Randomly generates the symmetric matrix */
    testing_start( &test_data );
    if ( async ) {
        hres = CHAMELEON_zplgsy_Tile_Async( bump, ChamUpperLower, descA, seedA,
                                            test_data.sequence, &test_data.request );
        CHAMELEON_Desc_Flush( descA, test_data.sequence );
    }
    else {
        hres = CHAMELEON_zplgsy_Tile( bump, ChamUpperLower, descA, seedA );
    }
    testing_stop( &test_data, 0 );
    test_data.hres = hres;

    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the solution */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {

        hres += check_zgenerate( args, ChamSymmetric, ChamUpperLower, descA, seedA, bump );

    }

    parameters_desc_destroy( &descA );

    return hres;
}

int
testing_zplgsy_std( run_arg_list_t *args, int check )
{
    testdata_t test_data = TESTINGS_TESTDATA_INITIALIZER;
    int        hres      = 0;

    /* Read arguments */
    int                    nb    = run_arg_get_nb( args );
    cham_uplo_t            uplo  = run_arg_get_uplo( args, "uplo", ChamUpper );
    int                    N     = run_arg_get_int( args, "N", 1000 );
    int                    LDA   = run_arg_get_int( args, "LDA", N );
    unsigned long long int seedA = run_arg_get_int( args, "seedA", testing_ialea() );
    CHAMELEON_Complex64_t  bump  = (CHAMELEON_Complex64_t)N;

    /* Descriptors */
    CHAMELEON_Complex64_t *A;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrix */
    A = malloc( sizeof(CHAMELEON_Complex64_t) * LDA*N );

    /* Randomly generates the symmetric matrix */
    testing_start( &test_data );
    hres = CHAMELEON_zplgsy( bump, uplo, N, A, LDA, seedA );
    testing_stop( &test_data, 0 );
    test_data.hres = hres;

    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the solution */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {

        hres += check_zgenerate_std( args, ChamSymmetric, uplo, N, N, A, LDA, seedA, bump );

    }

    free( A );

    return hres;
}

testing_t   test_zplgsy;
const char *zplgsy_params[] = { "mtxfmt", "nb", "uplo", "m", "n", "lda", "seedA", "bump", NULL };
const char *zplgsy_output[] = { NULL };
const char *zplgsy_outchk[] = { "RETURN", NULL };

/**
 * @brief Testing registration function
 */
void testing_zplgsy_init( void ) __attribute__( ( constructor ) );
void
testing_zplgsy_init( void )
{
    test_zplgsy.name   = "zplgsy";
    test_zplgsy.helper = "Random Symmetric Matrix Generation";
    test_zplgsy.params = zplgsy_params;
    test_zplgsy.output = zplgsy_output;
    test_zplgsy.outchk = zplgsy_outchk;
    test_zplgsy.fptr_desc = testing_zplgsy_desc;
    test_zplgsy.fptr_std  = testing_zplgsy_std;
    test_zplgsy.next   = NULL;

    testing_register( &test_zplgsy );
}

