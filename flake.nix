{
  description = "Chameleon Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    starpu.url = "github:Sacolle/nix-starpu";
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
      starpuPkg = (
        starpu.packages.${system}.default.override {
          enableCUDA = false;
          enableTrace = true;
        }
      );
      parsec = pkgs.callPackage ./parsec.nix { };
      chameleon = pkgs.callPackage ./chameleon.nix {
        starpu = starpuPkg;
        parsec = parsec;
      };
      parsecCuda = pkgsUnfree.callPackage ./parsec.nix { cudaSupport = true; };
      chameleonCuda = pkgsUnfree.callPackage ./chameleon.nix {
        starpu = starpuPkg;
        parsec = parsecCuda;
        cudaSupport = true;
      };
    in
    {
      packages.${system} = {
        default = chameleon;
        parsec = parsec;
        parsec-cuda = parsecCuda;
        chameleon-cuda = chameleonCuda;
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
      };
    };
}
