# Development shell for NuCS (macOS).
#
# Nix provides the tools (uv, MiniZinc); uv provides Python and installs the exact
# pins from pyproject.toml (numba 0.68.0rc1, numpy 2.5.3, ...), as the Dockerfile does.
#
# Usage:  nix-shell   (first run creates .venv and installs NuCS in editable mode)
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [
    pkgs.uv
    pkgs.minizinc # for the FlatZinc adapter (fzn-nucs) and its tests
  ];

  # Keep Numba's JIT cache inside the repo, as CI does.
  NUMBA_CACHE_DIR = ".numba/cache";

  shellHook = ''
    if [ ! -d .venv ]; then
      echo "Creating .venv and installing NuCS (editable) with dev and test extras..."
      uv venv --python 3.13 .venv
      uv pip install --python .venv/bin/python -e '.[dev,test]'
    fi
    source .venv/bin/activate
  '';
}
