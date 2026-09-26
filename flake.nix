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
    in
    {
      packages.${system} = {
        default = chameleon;
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
      };
    };
}
