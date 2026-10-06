{
  lib,
  stdenv,
  cmake,
  pkg-config,
  hwloc,
  fetchFromGitHub,
  openmpi,
  cudaSupport ? false,
  cudaPackages ? null,
  # architectures of the GPUs that will run the kernels (the build machine
  # usually has none, so "native" cannot be used)
  cudaArchitectures ? "70;80;90",
  autoAddDriverRunpath ? null,
  # HIP device (ROCm); PaRSEC cannot drive CUDA and HIP devices at the same time
  hipSupport ? false,
  rocmPackages ? null,
  perl,
}:

assert !(cudaSupport && hipSupport);


stdenv.mkDerivation {
  pname = "parsec";
  version = "1";

  # master with the DTD task classes, the GPU chores and the unpack of the
  # device copy (fadf9d8e1)
  src = fetchFromGitHub {
    owner = "ICLDisco";
    repo = "parsec";
    rev = "92b356425c02bdf7bef03eb85b4929488d19ce8d";
    hash = "sha256-8eAVAzfdtD+f3OMBuPKRZrako1Qw38qLBxgGvR2eJnI=";
  };

  # Races of the DTD interface hit by the Chameleon tests (the former 0001 is
  # upstream since a74104665), and the coherence of the GPU copies with the CPU
  # tasks of the fpointer API (0003)
  patches = [
    ./parsec-patches/0002-dtd-two-races-in-the-tracking-of-tile-users.patch
    ./parsec-patches/0003-dtd-cpu-hook-of-the-fpointer-api-keeps-gpu-copies-coherent.patch
  ];

  postPatch = ''
    substituteInPlace parsec/include/parsec.pc.in \
      --replace-fail 'exec_prefix=''${prefix}/@PARSEC_INSTALL_BINDIR@' 'exec_prefix=@CMAKE_INSTALL_FULL_BINDIR@' \
      --replace-fail 'libdir=''${prefix}/@PARSEC_INSTALL_LIBDIR@' 'libdir=@CMAKE_INSTALL_FULL_LIBDIR@' \
      --replace-fail 'includedir=''${prefix}/@PARSEC_INSTALL_INCLUDEDIR@' 'includedir=@CMAKE_INSTALL_FULL_INCLUDEDIR@'
  '';

  nativeBuildInputs = [
    cmake
    pkg-config
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_nvcc
    # finds libcuda in /run/opengl-driver on NixOS
    autoAddDriverRunpath
  ]
  # the HIP device is generated from the CUDA one by hipify-perl
  ++ lib.optionals hipSupport [
    perl
    rocmPackages.hipify
  ];

  buildInputs = [
    hwloc
    openmpi
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
  ]
  ++ lib.optionals hipSupport [ rocmPackages.clr ];

  cmakeFlags = [
    "-DPARSEC_DIST_WITH_MPI=ON"
    "-DPARSEC_GPU_WITH_CUDA=${if cudaSupport then "ON" else "OFF"}"
    "-DPARSEC_GPU_WITH_HIP=${if hipSupport then "ON" else "OFF"}"
    "-DPARSEC_GPU_WITH_LEVEL_ZERO=OFF"
  ]
  ++ lib.optionals cudaSupport [ "-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures}" ];
}
