{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.requests
    python3Packages.numpy
    python3Packages.scikit-learn
    python3Packages.sentence-transformers
    python3Packages.rich
    yt-dlp
  ];
}
