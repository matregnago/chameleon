{
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
  ];

  buildInputs = [
    openblas
    lapack
    hwloc
    openmpi
    # starpu
    parsec
  ];
  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DCHAMELEON_SCHED=PARSEC"
    "-DCHAMELEON_USE_MPI=ON"
  ];
}
