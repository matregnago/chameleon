{
  description = "Chameleon Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    starpu.url = "github:Sacolle/nix-starpu";
    # same nixpkgs (and so the same openmpi) for StarPU and Chameleon
    starpu.inputs.nixpkgs.follows = "nixpkgs";
    # runs the Nix binaries with the GPU driver of a non-NixOS host (PCAD)
    nix-gl-host.url = "github:numtide/nix-gl-host";
    nix-gl-host.inputs.nixpkgs.follows = "nixpkgs";
    self.submodules = true;
  };

  outputs =
    {
      self,
      nixpkgs,
      starpu,
      nix-gl-host,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      # CUDA is unfree; only the packages that ask for it are built with it
      pkgsUnfree = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      mkStarpu =
        enableCUDA:
        starpu.packages.${system}.default.override {
          inherit enableCUDA;
          enableMPI = true;
          enableTrace = true;
          # nix-starpu turns shared libs off with MPI (needed only for SMPI);
          # chameleon is a shared library and needs a shared libstarpumpi.
          # The MPI tests of `make check` hang in the Nix sandbox.
          extraOptions = [
            "--enable-shared"
            "--disable-mpi-check"
          ];
        };
      starpuPkg = mkStarpu false;
      starpuCuda = mkStarpu true;
      parsec = pkgs.callPackage ./parsec.nix { };
      parsecCuda = pkgsUnfree.callPackage ./parsec.nix { cudaSupport = true; };
      # the runtime is chosen with `runtime = "parsec" | "starpu"`, e.g.
      #   chameleon.override { runtime = "starpu"; }
      chameleon = pkgs.callPackage ./chameleon.nix {
        starpu = starpuPkg;
        parsec = parsec;
      };
      chameleonCuda = pkgsUnfree.callPackage ./chameleon.nix {
        starpu = starpuCuda;
        parsec = parsecCuda;
        cudaSupport = true;
      };
      chameleonStarpu = chameleon.override { runtime = "starpu"; };
      chameleonCudaStarpu = chameleonCuda.override { runtime = "starpu"; };
      # HIP is PaRSEC only: the StarPU package is not built with HIP
      parsecHip = pkgs.callPackage ./parsec.nix { hipSupport = true; };
      chameleonHip = pkgs.callPackage ./chameleon.nix {
        starpu = starpuPkg;
        parsec = parsecHip;
        hipSupport = true;
      };
    in
    {
      packages.${system} = {
        default = chameleon;
        # chameleon / chameleon-cuda use PaRSEC; the *-starpu ones use StarPU
        chameleon = chameleon;
        chameleon-parsec = chameleon;
        chameleon-starpu = chameleonStarpu;
        chameleon-cuda = chameleonCuda;
        chameleon-cuda-parsec = chameleonCuda;
        chameleon-cuda-starpu = chameleonCudaStarpu;
        parsec = parsec;
        parsec-cuda = parsecCuda;
        parsec-hip = parsecHip;
        chameleon-hip = chameleonHip;
        starpu = starpuPkg;
        starpu-cuda = starpuCuda;
      };
      devShells.${system} = {
        default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.cmake
          ];

          buildInputs = [
            chameleon
            starpuPkg
            parsec
          ];

        };
        # build tools of the CPU package, to build chameleon by hand
        cpu = pkgs.mkShell {
          inputsFrom = [ chameleon ];
          buildInputs = [ parsec ];
        };
        # outside NixOS, run the GPU binaries with `nixglhost <cmd>`
        cuda = pkgs.mkShell {
          inputsFrom = [ chameleonCuda ];
          buildInputs = [
            parsecCuda
            nix-gl-host.defaultPackage.${system}
          ];
        };
        # ROCm build; outside NixOS, run the GPU binaries with `nixglhost <cmd>`
        hip = pkgs.mkShell {
          inputsFrom = [ chameleonHip ];
          buildInputs = [
            parsecHip
            nix-gl-host.defaultPackage.${system}
          ];
        };
      };
    };
}
