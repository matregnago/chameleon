/**
 *
 * @file quark/core_blas_dag.h
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon Quark DAG generation header
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @date 2024-02-18
 *
 */
#ifndef _core_blas_dag_h_
#define _core_blas_dag_h_

#include "control/dag_task_colors.h"

#if defined(QUARK_DOT_DAG_ENABLE) /* || 1 */
#define DAG_SET_PROPERTIES( _name, _task )                                                   \
    QUARK_Task_Flag_Set( (Quark_Task_Flags*)opt, TASK_LABEL, (intptr_t)(_name)  );           \
    QUARK_Task_Flag_Set( (Quark_Task_Flags*)opt, TASK_COLOR,                                 \
                         (intptr_t)CHAMELEON_DAG_TASK_GET_COLOR_STR( options, _task ) );
#else
#define DAG_SET_PROPERTIES( _name, _task ) do {} while(0)
#endif

#define DAG_CORE_ASUM       DAG_SET_PROPERTIES( "ASUM"      , ASUM      )
#define DAG_CORE_AXPY       DAG_SET_PROPERTIES( "AXPY"      , AXPY      )
#define DAG_CORE_BUILD      DAG_SET_PROPERTIES( "BUILD"     , BUILD     )
#define DAG_CORE_GEADD      DAG_SET_PROPERTIES( "GEADD"     , GEADD     )
#define DAG_CORE_LASCAL     DAG_SET_PROPERTIES( "LASCAL"    , LASCAL    )
#define DAG_CORE_GELQT      DAG_SET_PROPERTIES( "GELQT"     , GELQT     )
#define DAG_CORE_GEMM       DAG_SET_PROPERTIES( "GEMM"      , GEMM      )
#define DAG_CORE_GEQRT      DAG_SET_PROPERTIES( "GEQRT"     , GEQRT     )
#define DAG_CORE_GESSM      DAG_SET_PROPERTIES( "GESSM"     , GESSM     )
#define DAG_CORE_GETRF      DAG_SET_PROPERTIES( "GETRF"     , GETRF     )
#define DAG_CORE_GETRIP     DAG_SET_PROPERTIES( "GETRIP"    , GETRIP    )
#define DAG_CORE_LATRO      DAG_SET_PROPERTIES( "LATRO"     , LATRO     )
#define DAG_CORE_HEMM       DAG_SET_PROPERTIES( "HEMM"      , HEMM      )
#define DAG_CORE_HER2K      DAG_SET_PROPERTIES( "HER2K"     , HER2K     )
#define DAG_CORE_HERFB      DAG_SET_PROPERTIES( "HERFB"     , HERFB     )
#define DAG_CORE_HERK       DAG_SET_PROPERTIES( "HERK"      , HERK      )
#define DAG_CORE_LACPY      DAG_SET_PROPERTIES( "LACPY"     , LACPY     )
#define DAG_CORE_LAG2C      DAG_SET_PROPERTIES( "LAG2C"     , LAG2C     )
#define DAG_CORE_LAG2Z      DAG_SET_PROPERTIES( "LAG2Z"     , LAG2Z     )
#define DAG_CORE_LANGE      DAG_SET_PROPERTIES( "LANGE"     , LANGE     )
#define DAG_CORE_LANGE_MAX  DAG_SET_PROPERTIES( "LANGE_MAX" , LANGE_MAX )
#define DAG_CORE_LANHE      DAG_SET_PROPERTIES( "LANHE"     , LANHE     )
#define DAG_CORE_LANSY      DAG_SET_PROPERTIES( "LANSY"     , LANSY     )
#define DAG_CORE_LANTR      DAG_SET_PROPERTIES( "LANTR"     , LANTR     )
#define DAG_CORE_LASET      DAG_SET_PROPERTIES( "LASET"     , LASET     )
#define DAG_CORE_LASWP      DAG_SET_PROPERTIES( "LASWP"     , LASWP     )
#define DAG_CORE_LAUUM      DAG_SET_PROPERTIES( "LAUUM"     , LAUUM     )
#define DAG_CORE_PLGHE      DAG_SET_PROPERTIES( "PLGHE"     , PLGHE     )
#define DAG_CORE_PLGSY      DAG_SET_PROPERTIES( "PLGSY"     , PLGSY     )
#define DAG_CORE_PLRNT      DAG_SET_PROPERTIES( "PLRNT"     , PLRNT     )
#define DAG_CORE_LASSQ      DAG_SET_PROPERTIES( "LASSQ"     , LASSQ     )
#define DAG_CORE_POTRF      DAG_SET_PROPERTIES( "POTRF"     , POTRF     )
#define DAG_CORE_SHIFT      DAG_SET_PROPERTIES( "SHIFT"     , SHIFT     )
#define DAG_CORE_SHIFTW     DAG_SET_PROPERTIES( "SHIFTW"    , SHIFTW    )
#define DAG_CORE_SSSSM      DAG_SET_PROPERTIES( "SSSSM"     , SSSSM     )
#define DAG_CORE_SWPAB      DAG_SET_PROPERTIES( "SWPAB"     , SWPAB     )
#define DAG_CORE_SYMM       DAG_SET_PROPERTIES( "SYMM"      , SYMM      )
#define DAG_CORE_SYR2K      DAG_SET_PROPERTIES( "SYR2K"     , SYR2K     )
#define DAG_CORE_SYRK       DAG_SET_PROPERTIES( "SYRK"      , SYRK      )
#define DAG_CORE_TRMM       DAG_SET_PROPERTIES( "TRMM"      , TRMM      )
#define DAG_CORE_TRSM       DAG_SET_PROPERTIES( "TRSM"      , TRSM      )
#define DAG_CORE_TRTRI      DAG_SET_PROPERTIES( "TRTRI"     , TRTRI     )
#define DAG_CORE_TPLQT      DAG_SET_PROPERTIES( "TPLQT"     , TPLQT     )
#define DAG_CORE_TPMLQT     DAG_SET_PROPERTIES( "TPMLQT"    , TPMLQT    )
#define DAG_CORE_TPMQRT     DAG_SET_PROPERTIES( "TPMQRT"    , TPMQRT    )
#define DAG_CORE_TPQRT      DAG_SET_PROPERTIES( "TPQRT"     , TPQRT     )
#define DAG_CORE_TSTRF      DAG_SET_PROPERTIES( "TSTRF"     , TSTRF     )
#define DAG_CORE_UNMLQ      DAG_SET_PROPERTIES( "UNMLQ"     , UNMLQ     )
#define DAG_CORE_UNMQR      DAG_SET_PROPERTIES( "UNMQR"     , UNMQR     )
#define DAG_CORE_ORMLQ      DAG_SET_PROPERTIES( "ORMLQ"     , ORMLQ     )
#define DAG_CORE_ORMQR      DAG_SET_PROPERTIES( "ORMQR"     , ORMQR     )

#define DAG_CORE_TSLQT      DAG_CORE_TPLQT
#define DAG_CORE_TSMLQ      DAG_CORE_TPMLQT
#define DAG_CORE_TSMQR      DAG_CORE_TPMQRT
#define DAG_CORE_TSQRT      DAG_CORE_TPQRT
#define DAG_CORE_TTLQT      DAG_CORE_TPLQT
#define DAG_CORE_TTMLQ      DAG_CORE_TPMLQT
#define DAG_CORE_TTMQR      DAG_CORE_TPMQRT
#define DAG_CORE_TTQRT      DAG_CORE_TPQRT

#define DAG_CORE_GESSQ      DAG_SET_PROPERTIES( "GESSQ"    , GESSQ     )
#define DAG_CORE_HESSQ      DAG_SET_PROPERTIES( "HESSQ"    , HESSQ     )
#define DAG_CORE_SYSSQ      DAG_SET_PROPERTIES( "SYSSQ"    , SYSSQ     )
#define DAG_CORE_TRSSQ      DAG_SET_PROPERTIES( "TRSSQ"    , TRSSQ     )
#define DAG_CORE_PLSSQ      DAG_SET_PROPERTIES( "PLSSQ"    , PLSSQ     )
#define DAG_CORE_PLSSQ2     DAG_SET_PROPERTIES( "PLSSQ2"   , PLSSQ2    )

#define DAG_CORE_GESUM      DAG_SET_PROPERTIES( "GESUM"    , GESUM     )
#define DAG_CORE_CESCA      DAG_SET_PROPERTIES( "CESCA"    , CESCA     )
#define DAG_CORE_GRAM       DAG_SET_PROPERTIES( "GRAM"     , GRAM      )

#endif /* _core_blas_dag_h_ */
