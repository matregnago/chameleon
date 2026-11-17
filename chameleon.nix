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
  # runtime used by chameleon: "parsec" or "starpu" (CHAMELEON_SCHED)
  runtime ? "parsec",
  # StarPU must be built with MPI (enableMPI) when this is on
  mpiSupport ? true,
  cudaSupport ? false,
  cudaPackages ? null,
  # CHAMELEON_CUDA_TARGETS: names or sm_xx of the GPUs that will run the kernels
  cudaTargets ? "Volta Ampere Hopper",
  hipSupport ? false,
  rocmPackages ? null,
  # CMAKE_HIP_ARCHITECTURES: AMD GPUs that will run the kernels (MI200, MI300, RDNA3)
  hipArchitectures ? "gfx90a;gfx942;gfx1100",
}:

assert lib.assertOneOf "runtime" runtime [
  "parsec"
  "starpu"
];
assert !(cudaSupport && hipSupport);
# the StarPU of this flake is not built with HIP
assert lib.assertMsg (!(hipSupport && runtime == "starpu"))
  "chameleon: HIP is only supported with runtime = \"parsec\"";

stdenv.mkDerivation {
  pname = "chameleon-${runtime}";
  version = "1.4.0";

  src = ./.;

  nativeBuildInputs = [
    cmake
    pkg-config
    gfortran
    python3
  ]
  ++ lib.optionals cudaSupport [ cudaPackages.cuda_nvcc ]
  ++ lib.optionals hipSupport [ rocmPackages.clr ];

  buildInputs = [
    openblas
    lapack
    hwloc
  ]
  ++ lib.optionals mpiSupport [ openmpi ]
  ++ lib.optionals (runtime == "parsec") [ parsec ]
  ++ lib.optionals (runtime == "starpu") [ starpu ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
    cudaPackages.libcublas
    cudaPackages.libcusolver
    cudaPackages.libcusparse
    cudaPackages.libnvjitlink
  ]
  ++ lib.optionals hipSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblas-common
    rocmPackages.hipsolver
  ];
  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DCHAMELEON_SCHED=${lib.toUpper runtime}"
    (lib.cmakeBool "CHAMELEON_USE_MPI" mpiSupport)
  ]
  ++ lib.optionals cudaSupport [
    "-DCHAMELEON_USE_CUDA=ON"
    "-DCHAMELEON_CUDA_TARGETS=${cudaTargets}"
  ]
  ++ lib.optionals hipSupport [
    "-DCHAMELEON_USE_HIP=ON"
    "-DCMAKE_HIP_COMPILER=${rocmPackages.clr}/bin/amdclang++"
    "-DCMAKE_HIP_ARCHITECTURES=${hipArchitectures}"
  ];
}
