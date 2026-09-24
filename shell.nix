# Development shell for NuCS (macOS).
#
# Nix provides the tools (uv, MiniZinc); uv provides Python and installs the exact
# pins from pyproject.toml (numba 0.68.0rc1, numpy 2.5.3, ...), as the Dockerfile does.
#
# Usage:  nix-shell, or `use nix` in .envrc with direnv.
# To change Python, edit pythonVersion: .venv is rebuilt automatically on the next load.
{ pkgs ? import <nixpkgs> { } }:

let
  pythonVersion = "3.14"; # CI tests 3.12, 3.13 and 3.14
in
pkgs.mkShell {
  packages = [
    pkgs.uv
    pkgs.minizinc # for the FlatZinc adapter (fzn-nucs) and its tests
  ];

  # Keep Numba's JIT cache inside the repo, as CI does.
  NUMBA_CACHE_DIR = ".numba/cache";

  shellHook = ''
    current=$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
    if [ "$current" != "${pythonVersion}" ]; then
      echo "Building .venv with Python ${pythonVersion} and installing NuCS (editable) with dev and test extras..."
      rm -rf .venv
      uv venv --python ${pythonVersion} .venv
      uv pip install --python .venv/bin/python -e '.[dev,test]'
    fi
    source .venv/bin/activate
  '';
}