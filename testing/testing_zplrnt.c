/**
 *
 * @file testing_zplrnt.c
 *
 * @copyright 2019-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon zplrnt testing
 *
 * @version 1.4.0
 * @author Brieuc Nicolas
 * @date 2025-12-17
 * @precisions normal z -> c d s
 *
 */
#include <chameleon.h>
#include <chameleon_lapack.h>
#include "testings.h"
#include "testing_zcheck.h"

int
testing_zplrnt_desc( run_arg_list_t *args, int check )
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

    /* Descriptors */
    CHAM_desc_t *descA;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrix */
    parameters_desc_create( "A", &descA, ChamComplexDouble, nb, nb, LDA, N, M, N );

    /* Randomly generates the matrix */
    testing_start( &test_data );
    if ( async ) {
        hres = CHAMELEON_zplrnt_Tile_Async( descA, seedA,
                                            test_data.sequence, &test_data.request );
        CHAMELEON_Desc_Flush( descA, test_data.sequence );
    }
    else {
        hres = CHAMELEON_zplrnt_Tile( descA, seedA );
    }
    testing_stop( &test_data, 0 );
    test_data.hres = hres;

    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the solution */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {
        hres += check_zgenerate( args, ChamGeneral, ChamUpperLower, descA, seedA, 0 );
    }

    parameters_desc_destroy( &descA );

    return hres;
}

int
testing_zplrnt_std( run_arg_list_t *args, int check )
{
    testdata_t test_data = TESTINGS_TESTDATA_INITIALIZER;
    int        hres      = 0;

    /* Read arguments */
    int                    nb    = run_arg_get_nb( args );
    int                    N     = run_arg_get_int( args, "N", 1000 );
    int                    M     = run_arg_get_int( args, "M", N );
    int                    LDA   = run_arg_get_int( args, "LDA", M );
    unsigned long long int seedA = run_arg_get_int( args, "seedA", testing_ialea() );

    /* Descriptors */
    CHAMELEON_Complex64_t *A;

    CHAMELEON_Set( CHAMELEON_TILE_SIZE, nb );

    /* Creates the matrix */
    A = malloc( sizeof(CHAMELEON_Complex64_t) * LDA * N );

    /* Randomly generates the matrix */
    testing_start( &test_data );
    hres = CHAMELEON_zplrnt( M, N, A, LDA, seedA );
    testing_stop( &test_data, 0 );
    test_data.hres = hres;

    hres = ( hres == CHAMELEON_SUCCESS ) ? 0 : 1;

    /* Checks the solution */
    if ( ( hres == CHAMELEON_SUCCESS ) && check ) {
        hres += check_zgenerate_std( args, ChamGeneral, ChamUpperLower, M, N, A, LDA, seedA, 0 );
    }

    free( A );

    return hres;
}

testing_t   test_zplrnt;
const char *zplrnt_params[] = { "mtxfmt", "nb", "m", "n", "lda", "seedA", NULL };
const char *zplrnt_output[] = { NULL };
const char *zplrnt_outchk[] = { "||A||", "||B||", "||R||", "RETURN", NULL };

/**
 * @brief Testing registration function
 */
void testing_zplrnt_init( void ) __attribute__( ( constructor ) );
void
testing_zplrnt_init( void )
{
    test_zplrnt.name   = "zplrnt";
    test_zplrnt.helper = "Random Matrix Generation";
    test_zplrnt.params = zplrnt_params;
    test_zplrnt.output = zplrnt_output;
    test_zplrnt.outchk = zplrnt_outchk;
    test_zplrnt.fptr_desc = testing_zplrnt_desc;
    test_zplrnt.fptr_std  = testing_zplrnt_std;
    test_zplrnt.next   = NULL;

    testing_register( &test_zplrnt );
}
