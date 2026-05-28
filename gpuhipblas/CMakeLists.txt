###
#
# @file CMakeLists.txt
#
# @copyright 2009-2014 The University of Tennessee and The University of
#                      Tennessee Research Foundation. All rights reserved.
# @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                      Univ. Bordeaux. All rights reserved.
#
###
#
#  @project CHAMELEON
#  CHAMELEON is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     King Abdullah Univesity of Science and Technology
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
#
# @version 1.4.0
#  @author Cedric Castagnede
#  @author Emmanuel Agullo
#  @author Mathieu Faverge
#  @author Florent Pruvost
#  @author Loris Lucido
#  @author Brieuc Nicolas
#  @date 2024-02-18
#
###

include(CheckLanguage)

check_language(HIP)
if (NOT CMAKE_HIP_COMPILER)
  message(FATAL_ERROR "HIP language support is required to build gpuhipblas, but no HIP compiler was detected. Set HIPCXX/CMAKE_HIP_COMPILER or install a ROCm HIP toolchain.")
endif()

enable_language(HIP)

add_subdirectory(include)
add_subdirectory(compute)
add_subdirectory(eztrace_module)

###
### END CMakeLists.txt
###
