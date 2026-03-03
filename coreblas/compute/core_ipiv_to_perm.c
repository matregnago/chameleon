/**
 *
 * @file core_ipiv_to_perm.c
 *
 * @copyright 2023-2025 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon core_ipiv_to_perm CPU kernel
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Matteo Marcos
 * @date 2025-12-19
 */
#include "coreblas.h"

/**
 *******************************************************************************
 *
 * The idea here is to generate a permutation from the sequence of
 * pivot.  To avoid storing one whole column at each step, we keep
 * track of two vectors of nb elements, the first one contains the
 * permutation of the first nb elements, and the second one contains
 * the inverse permutation of those same elements.
 *
 * Lets have i the element to pivot with ip. ipiv[i] = ip;
 * We set i_1 as such invp[ i_1  ] = i
 *  and  ip_1 as such invp[ ip_1 ] = ip
 *
 * At each step we want to:
 *   - swap perm[i] and perm[ip]
 *   - set invp[i_1] to ip
 *   - set invp[ip_1] to i
 *
 *******************************************************************************
 *
 * @param[in] m0
 *          The base index for all values in ipiv, perm and invp. m0 >= 0.
 *
 * @param[in] m
 *          The number of elements in perm and invp. m >= 0.
 *
 * @param[in] k
 *          The number of elements in ipiv. k >= 0.
 *
 * @param[in] K1
 *          The first element of IPIV for which an interchange will
 *          be done.
 *
 * @param[in] K2
 *          The last element of ipiv for which an interchange will
 *          be done.
 *
 * @param[in] ipiv
 *          The pivot array of size n. This is a (m0+1)-based indices array to follow
 *          the Fortran standard.
 *
 * @param[out] perm
 *          The permutation array of the destination row indices (m0-based) of the [1,n] set of rows.
 *
 * @param[out] invp
 *          The permutation array of the origin row indices (m0-based) of the [1,n] set of rows.
 *
 */
void CORE_ipiv_to_perm( int m0, int m, int k, int K1, int K2,
                        const int *ipiv, int *perm, int *invp )
{
    /*
     * Let's note that variables suffixed by l are local values (shifted by
     * -m0), and variable suffixed by g are global values.
     */
    int il, ipl, il_1, ipl_1, jl;
    int ig, ipg, ig_1, ipg_1;

    /* Loop through perm and invp to initialize them with no pivoting */
    for( il=0; il < m; il++ ) {
        ig = il + m0;
        perm[il] = ig;
        invp[il] = ig;
    }

    /* Loop through ipiv to compute perm and invp */
    for(il = 0; il < k; il++) {
        ig = il + m0;
        if ( ( ig < K1 ) || ( ig > K2 ) ) {
            continue;
        }

        ipg = ipiv[ il ] - 1;
        ipl = ipg - m0;

        /* Pivot should only returns rows below or equal to the current one */
        assert( ipl >= il );

        /* If the row i is permuted with the row ip (i != ip) */
        if ( ipl > il ) {

            /* Save initial permutation in the cycle */
            ig_1 = perm[ il ];

            /* If the row ipl is in the current block */
            if ( ipl < m ) {
                /* Save final permutation point in the cycle and update intermediate one */
                ipg_1 = perm[ ipl ];
                perm[ ipl ] = ig_1;
            }
            /* If the row ipl is not in the current block */
            else {
                ipg_1 = ipg;
                /* Loop through invp to see if ip has already been set in the current block */
                for( jl=0; jl < m; jl++ ) {
                    if( invp[jl] == ipg ) {
                        ipg_1 = jl + m0;
                        break;
                    }
                }
            }

            /* Save the final permutation */
            perm[ il ] = ipg_1;
            il_1  = ig_1  - m0;
            ipl_1 = ipg_1 - m0;

            /* Update initial and final invp if necessary */
            if (il_1  < m) invp[il_1 ] = ipg;
            if (ipl_1 < m) invp[ipl_1] = ig;
        }
    }
}
