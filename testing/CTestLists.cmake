#
# Check testing/
#
set(THREADS 2) # Amount of threads
set(N_GPUS 0) # Amount of graphic cards
set(TEST_CATEGORIES shm)
if (CHAMELEON_USE_MPI AND MPI_C_FOUND)
  set( TEST_CATEGORIES ${TEST_CATEGORIES} mpi cdist )
endif()
if ( CHAMELEON_SCHED_STARPU )
  set( TEST_CATEGORIES ${TEST_CATEGORIES} ooc )
endif()
if (CHAMELEON_USE_CUDA AND CUDAToolkit_FOUND)
  set(N_GPUS 0 1)
endif()
if (CHAMELEON_USE_HIP AND hip_FOUND)
  set(N_GPUS 0 1)
endif()
if (CHAMELEON_SIMULATION)
  set(TEST_CATEGORIES simushm)
  if (CHAMELEON_USE_CUDA)
    set( TEST_CATEGORIES ${TEST_CATEGORIES} simugpu )
  endif()
endif()

set( SINGLE_PRECISIONS s c )
# list all tests that have a specific input file for single precision computations
set( SINGLE_TESTS gepdf_qdwh genm2 )

if (NOT CHAMELEON_SIMULATION)

  foreach(prec ${CHAMELEON_PRECISION})
    if ( ${prec} STREQUAL ds OR ${prec} STREQUAL zc )
      continue()
    endif()

    set (CMD ./chameleon_${prec}testing)

    #
    # Create the list of test based on precision and runtime
    #
    # Norms
    set( TESTS print lacpy laset latro plrnt plgsy lange lantr lansy plrnk )
    if ( ${prec} STREQUAL c OR ${prec} STREQUAL z )
      set( TESTS ${TESTS} plghe lanhe )
    endif()
    # BLAS
    set( TESTS ${TESTS} geadd tradd lascal gemm symm syrk syr2k trmm trsm )
    if ( ${prec} STREQUAL c OR ${prec} STREQUAL z )
      set( TESTS ${TESTS} hemm herk her2k )
    endif()
    # Cholesky
    set( TESTS ${TESTS} potrf potrs posv trtri lauum )
    if ( NOT CHAMELEON_SCHED_PARSEC )
      set( TESTS ${TESTS} potri poinv)
    endif()
    # Symmetric factorization
    if ( ${prec} STREQUAL c OR ${prec} STREQUAL z )
      set( TESTS ${TESTS} sytrf sytrs sysv )
    endif()
    # LU
    set( TESTS ${TESTS} getrf_nopiv getrs_nopiv gesv_nopiv )
    if ( CHAMELEON_SCHED_STARPU OR CHAMELEON_SCHED_PARSEC )
      set( TESTS ${TESTS} laswp )
      if ( CHAMELEON_SCHED_PARSEC OR HAVE_STARPU_NONE_NONZERO )
        set( TESTS ${TESTS} getrf getrs gesv )
      endif()
    endif()
    # QR / LQ
    set( TESTS ${TESTS} geqrf gelqf geqrf_hqr gelqf_hqr )
    if ( ${prec} STREQUAL c OR ${prec} STREQUAL z )
      set( TESTS ${TESTS}
        ungqr     unglq     unmqr     unmlq
        ungqr_hqr unglq_hqr unmqr_hqr unmlq_hqr)
    else()
      set( TESTS ${TESTS}
        orgqr     orglq     ormqr     ormlq
        orgqr_hqr orglq_hqr ormqr_hqr ormlq_hqr)
    endif()
    #set( TESTS ${TESTS} geqrs     gelqs     )
    #set( TESTS ${TESTS} geqrs_hqr gelqs_hqr )
    # Others
    set( TESTS ${TESTS} gels gels_hqr )
    set( TESTS ${TESTS} genm2 gepdf_qr gepdf_qdwh gesvd )
    set( TESTS ${TESTS} cesca gram )

    foreach( cat ${TEST_CATEGORIES} )
      foreach( gpus ${N_GPUS} )

        set( TESTSTMP ${TESTS} )
        if ( ${cat} STREQUAL "cdist" )
            set( NP 3 )
        elseif ( ${cat} STREQUAL "mpi" )
            set( NP 2 )
        else()
            set( NP 1 )
        endif()

        if ( NOT ( ${gpus} EQUAL 0 ) )
          set( cat ${cat}_gpu )
          list( REMOVE_ITEM TESTSTMP gram lanhe lange lansy lantr lascal plrnk print )
        endif()

        if ( ${cat} STREQUAL "shm")
          set( PREFIX "" )
        else()
          set( PREFIX mpiexec --bind-to none -n ${NP} )
          list( REMOVE_ITEM TESTSTMP gesvd )
        endif()

        if( ${cat} STREQUAL "cdist" )
          set( CDIST --custom input/dist_3.txt )
        else()
          set( CDIST "" )
        endif()

        if( ${cat} STREQUAL "ooc" )
          set( OOC -F 2 )
        else()
          set( OOC "" )
        endif()

        set(FULLPREFIX ${PREFIX} ${CMD} ${CDIST} ${OOC} -c -t ${THREADS} -g ${gpus} )

        foreach( test ${TESTSTMP} )
          if ( ${test} IN_LIST SINGLE_TESTS AND ${prec} IN_LIST SINGLE_PRECISIONS )
            add_test( test_${cat}_${prec}${test} ${FULLPREFIX} -P 1 -f input/${test}_32.in )
          else()
            add_test( test_${cat}_${prec}${test} ${FULLPREFIX} -P 1 -f input/${test}.in )
          endif()
          if( ${cat} STREQUAL "ooc" )
            set_tests_properties( test_${cat}_${prec}${test}
              PROPERTIES ENVIRONMENT "STARPU_DISK_SWAP=/tmp;STARPU_DISK_SWAP_BACKEND=unistd;STARPU_LIMIT_CPU_MEM=1" )
          endif()
        endforeach()

        set( trtri_test_cmd ${FULLPREFIX} -P ${NP} -f input/trtri.in )
        foreach( trtri_variant 1 2 3 )
          set( trtri_test_name test_${cat}_${prec}trtri_v${trtri_variant} )
          add_test( ${trtri_test_name} ${trtri_test_cmd} )

          set( trtri_env "CHAMELEON_TRTRI_ALGO=${trtri_variant}" )
          if( ${cat} STREQUAL "ooc" )
            set( trtri_env "${trtri_env};STARPU_DISK_SWAP=/tmp;STARPU_DISK_SWAP_BACKEND=unistd;STARPU_LIMIT_CPU_MEM=1" )
          endif()

          set_tests_properties( ${trtri_test_name}
            PROPERTIES ENVIRONMENT "${trtri_env}" )
        endforeach()

        if ( CHAMELEON_SCHED_STARPU OR CHAMELEON_SCHED_PARSEC )
          set( laswp_test_prefix test_${cat}_${prec}laswp )

          if ( ${cat} STREQUAL "mpi" )
            add_test( test_${cat}_${prec}laswp_allreduce ${FULLPREFIX} -P ${NP} -f input/laswp.in )
            set_tests_properties( ${laswp_test_prefix}_allreduce
                PROPERTIES ENVIRONMENT "CHAMELEON_LASWP_ALLREDUCE=1" )
          endif()

          add_test( test_${cat}_${prec}laswp_batch ${FULLPREFIX} -P ${NP} -f input/laswp.in )
          set_tests_properties( test_${cat}_${prec}laswp_batch
            PROPERTIES ENVIRONMENT "CHAMELEON_BATCH_SIZE=3" )

          # if ( ${cat} STREQUAL "mpi" )
          #   add_test( ${laswp_test_prefix}_ppiv_comm_with_task ${laswp_test_cmd} -P ${NP} )
          # endif()

          if ( CHAMELEON_SCHED_STARPU AND CHAMELEON_USE_RECURSIVE_TASKS AND
               ( ${cat} STREQUAL "shm" OR ${cat} STREQUAL "mpi" ) )
            if ( ${cat} STREQUAL "mpi" )
              set( recursive_mtxfmts 1 )
            else()
              set( recursive_mtxfmts 0 1 )
            endif()

            foreach( recursive_mtxfmt ${recursive_mtxfmts} )
              add_test( test_${cat}_${prec}print_recursive_${recursive_mtxfmt}
                ${FULLPREFIX} -P 1 -f input/print_recursive.in
                --rec=full --mtxfmt=${recursive_mtxfmt} )

              add_test( test_${cat}_${prec}getrf_nopiv_rectile_${recursive_mtxfmt}
                ${FULLPREFIX} -P 1 -f input/getrf_nopiv_rectile.in
                --mtxfmt=${recursive_mtxfmt} --rec=full )

              add_test( test_${cat}_${prec}getrf_nopiv_recpanel_${recursive_mtxfmt}
                ${FULLPREFIX} -P 1 -f input/getrf_nopiv_recpanel.in
                --mtxfmt=${recursive_mtxfmt} --rec=full )

              set( getrf_nopiv_recpanel_la2_test
                test_${cat}_${prec}getrf_nopiv_recpanel_la2_${recursive_mtxfmt} )
              add_test( ${getrf_nopiv_recpanel_la2_test}
                ${FULLPREFIX} -P 1 -f input/getrf_nopiv_recpanel.in
                --mtxfmt=${recursive_mtxfmt} --rec=full )
              set_tests_properties( ${getrf_nopiv_recpanel_la2_test}
                PROPERTIES ENVIRONMENT "CHAMELEON_LOOKAHEAD=2" )

              if ( ${cat} STREQUAL "mpi" )
                add_test( test_${cat}_${prec}getrf_nopiv_recpanel_p${NP}_${recursive_mtxfmt}
                  ${FULLPREFIX} -P ${NP} -f input/getrf_nopiv_recpanel.in
                  --mtxfmt=${recursive_mtxfmt} --rec=full )
              endif()
            endforeach()
          endif()
        endif()

        if ( CHAMELEON_SCHED_PARSEC OR ( CHAMELEON_SCHED_STARPU AND HAVE_STARPU_NONE_NONZERO ) )
          set( getrf_test_prefix test_${cat}_${prec}getrf )
          set( getrf_test_cmd ${FULLPREFIX} -P ${NP} -f input/getrf.in )

          add_test( ${getrf_test_prefix}_ppivpercol ${getrf_test_cmd} )
          set_tests_properties( ${getrf_test_prefix}_ppivpercol
            PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppivpercolumn;CHAMELEON_BATCH_SIZE=0" )

          add_test( ${getrf_test_prefix}_ppivpercol_batch ${getrf_test_cmd} )
          set_tests_properties( ${getrf_test_prefix}_ppivpercol_batch
            PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppivpercolumn;CHAMELEON_BATCH_SIZE=3" )

          add_test( ${getrf_test_prefix}_ppivblocked ${getrf_test_cmd} )
          set_tests_properties( ${getrf_test_prefix}_ppivblocked
            PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppiv;CHAMELEON_BATCH_SIZE=0" )

          add_test( ${getrf_test_prefix}_ppivblocked_batch ${getrf_test_cmd} )
          set_tests_properties( ${getrf_test_prefix}_ppivblocked_batch
            PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppiv;CHAMELEON_BATCH_SIZE=3" )

          add_test( ${getrf_test_prefix}_split_laswp_set_gemm ${getrf_test_cmd} )
          set_tests_properties( ${getrf_test_prefix}_split_laswp_set_gemm
            PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppiv;CHAMELEON_BATCH_SIZE=0;CHAMELEON_SPLIT_LASWP_GEMM=1" )

          add_test( ${getrf_test_prefix}_split_laswp_set_gemm_batch ${getrf_test_cmd} )
          set_tests_properties( ${getrf_test_prefix}_split_laswp_set_gemm_batch
            PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppiv;CHAMELEON_BATCH_SIZE=3;CHAMELEON_SPLIT_LASWP_GEMM=1" )

          # if ( ${cat} STREQUAL "mpi" )
          #   add_test( ${getrf_test_prefix}_ppiv_comm_with_task ${getrf_test_cmd} -P ${NP} )
          #   add_test( test_${cat}_${prec}getrs_ppiv_comm_with_task ${PREFIX} ${CMD} -c -t ${THREADS} -g ${gpus} -P ${NP} -f input/getrs.in )
          #   add_test( test_${cat}_${prec}gesv_ppiv_comm_with_task  ${PREFIX} ${CMD} -c -t ${THREADS} -g ${gpus} -P ${NP} -f input/gesv.in )
          #   set_tests_properties( test_${cat}_${prec}getrf_ppiv_comm_with_task
          #     PROPERTIES ENVIRONMENT "CHAMELEON_GETRF_ALGO=ppiv;CHAMELEON_GETRF_BATCH_SIZE=0;CHAMELEON_ALLREDUCE=cham_spu_tasks" )
          # endif()
        endif()

        list( REMOVE_ITEM TESTSTMP print gepdf_qr laswp )

        if ( ${cat} STREQUAL "shm" )
          foreach( test ${TESTSTMP} )
            if ( ${test} IN_LIST SINGLE_TESTS AND ${prec} IN_LIST SINGLE_PRECISIONS )
              add_test( test_${cat}_${prec}${test}_std ${FULLPREFIX} -P 1 -f input/${test}_32.in --api=1 )
            else()
              add_test( test_${cat}_${prec}${test}_std ${FULLPREFIX} -P 1 -f input/${test}.in --api=1 )
            endif()
          endforeach()
        endif()
      endforeach()
    endforeach()
  endforeach()

else (NOT CHAMELEON_SIMULATION)

  # constraints for which we have perfmodels in simucore/perfmodels/
  set( TESTS "potrf")
  set(CHAMELEON_PRECISIONS_SIMU "s;d")
  set(TEST_CMD_simushm -t ${THREADS} -g 0)
  set(TEST_CMD_simugpu -t ${THREADS} -g 1)
  set(PLATFORMS "sirocco-a100-ctest")
  set(BSIZE_sirocco-a100-ctest "800;1600")

  # loop over constraints
  foreach(cat ${TEST_CATEGORIES})
    foreach(prec ${CHAMELEON_PRECISIONS_SIMU})
      string(TOUPPER ${prec} PREC)
      foreach(test ${TESTS})
        if (CHAMELEON_PREC_${PREC})
          foreach(plat ${PLATFORMS})
            foreach(bsize ${BSIZE_${plat}})
              math(EXPR size "10 * ${bsize}")
              add_test(test_${cat}_${prec}${test}_${plat}_${bsize} ./chameleon_${prec}testing -o ${test} ${TEST_CMD_${cat}} -n ${size} -b ${bsize})
              set_tests_properties(test_${cat}_${prec}${test}_${plat}_${bsize} PROPERTIES
                                  ENVIRONMENT "STARPU_HOME=${CMAKE_SOURCE_DIR}/simucore/perfmodels;STARPU_HOSTNAME=${plat}"
                                  )
            endforeach()
          endforeach()
        endif()
      endforeach()
    endforeach()
  endforeach()

endif (NOT CHAMELEON_SIMULATION)
