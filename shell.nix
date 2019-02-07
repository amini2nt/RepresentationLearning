let
  pkgs = import (fetchTarball https://github.com/NixOS/nixpkgs/archive/18.09.tar.gz) {}; 
in with pkgs; {
  simpleEnv = stdenv.mkDerivation {
    name = "ift6135";
    version = "1";
    buildInputs = [ 
      python36Packages.pytorchWithoutCuda
      python36Packages.torchvision
      python36Packages.matplotlib
    ];  
  };  
}

