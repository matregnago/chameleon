{
  stdenv,
  cmake,
  pkg-config,
  hwloc,
  fetchFromGitHub,
  openmpi
}:

stdenv.mkDerivation {
  pname = "parsec";
  version = "1";

  src = fetchFromGitHub {
    owner = "ICLDisco";
    repo = "parsec";
    rev = "d0fc4c01e6d10c8cc10719c1930ce1a5677a075c";
    hash = "sha256-cgLZeWfznrUjCpEV7YR2uCgoigYVZH63q/Xl35ncpWk=";
  };

  # Races of the DTD interface hit by the Chameleon tests (0001 is upstream a74104665)
  patches = [
    ./parsec-patches/0001-data-fix-use-after-free-race-in-self-contained-data-release.patch
    ./parsec-patches/0002-dtd-two-races-in-the-tracking-of-tile-users.patch
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
  ];

  buildInputs = [
    hwloc
    openmpi
  ];
  cmakeFlags = [
    "-DPARSEC_DIST_WITH_MPI=ON"
  ];
}
