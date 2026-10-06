{
  lib,
  stdenv,
  cmake,
  pkg-config,
  gfortran,
  openblas,
  lapack,
  hwloc,
  openmpi,
  starpu,
  parsec,
  python3,
  cudaSupport ? false,
  cudaPackages ? null,
  # CHAMELEON_CUDA_TARGETS: names or sm_xx of the GPUs that will run the kernels
  cudaTargets ? "Volta Ampere Hopper",
}:

stdenv.mkDerivation {
  pname = "chameleon";
  version = "1.4.0";

  src = ./.;

  nativeBuildInputs = [
    cmake
    pkg-config
    gfortran
    python3
  ]
  ++ lib.optionals cudaSupport [ cudaPackages.cuda_nvcc ];

  buildInputs = [
    openblas
    lapack
    hwloc
    openmpi
    # starpu
    parsec
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
    cudaPackages.libcublas
    cudaPackages.libcusolver
    cudaPackages.libcusparse
    cudaPackages.libnvjitlink
  ];
  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DCHAMELEON_SCHED=PARSEC"
    "-DCHAMELEON_USE_MPI=ON"
  ]
  ++ lib.optionals cudaSupport [
    "-DCHAMELEON_USE_CUDA=ON"
    "-DCHAMELEON_CUDA_TARGETS=${cudaTargets}"
  ];
}
