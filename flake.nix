{
  description = "Chameleon Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    starpu.url = "github:Sacolle/nix-starpu";
    self.submodules = true;
  };

  outputs =
    {
      self,
      nixpkgs,
      starpu,
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
        cuda = pkgs.mkShell {
          inputsFrom = [ chameleonCuda ];
          buildInputs = [ parsecCuda ];
        };
      };
    };
}
