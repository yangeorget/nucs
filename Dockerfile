# syntax=docker/dockerfile:1
#
# A MiniZinc image that builds NuCS from this repository's source and uses it as the default solver.
#
# Build from the repo root (the build context must be the NuCS source tree):
#   docker build -t minizinc-nucs .
#
# Run (NuCS is the default, so no --solver flag is needed):
#   docker run --rm -v "$PWD:/work" -w /work minizinc-nucs minizinc model.mzn data.dzn
#   docker run --rm -v "$PWD:/work" -w /work minizinc-nucs minizinc --solver nucs model.mzn

FROM minizinc/mznc2026:latest

# Python version to run NuCS on.
ARG PYTHON_VERSION=3.13

USER root

# uv gives us a portable, prebuilt CPython and a fast installer, independent of the base image's OS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Install the requested Python and create an isolated environment for NuCS.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
ENV VIRTUAL_ENV=/opt/nucs
ENV PATH="/opt/nucs/bin:${PATH}"
RUN uv venv --python "${PYTHON_VERSION}" "${VIRTUAL_ENV}"

# Build and install NuCS from the source tree (the build context) together with its dependencies
# (numba, numpy, llvmlite, ...). The packaged share/minizinc globals library is installed with it.
COPY . /src
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" /src

# Sanity check: the FlatZinc adapter (fzn-nucs console script) was installed.
RUN command -v fzn-nucs >/dev/null \
 || { echo "ERROR: 'fzn-nucs' console script not installed from source." >&2; exit 1; }

# Register NuCS as a MiniZinc solver. fzn-nucs writes a nucs.msc with absolute paths to the fzn-nucs
# executable and to the NuCS globals library, so MiniZinc finds it regardless of PATH or working directory.
RUN fzn-nucs --register

# Make NuCS the default solver: in MiniZinc's preferences, the empty "" tag maps to the global default
# solver id (org.nucs.nucs is the id declared in nucs.msc).
RUN mkdir -p /root/.minizinc \
 && printf '{\n    "tagDefaults": [["", "org.nucs.nucs"]]\n}\n' > /root/.minizinc/Preferences.json

# Bake the Numba JIT cache for ALL of NuCS's @njit code into the image, so no compilation happens at solve
# time. warm_cache.py compiles every propagator, heuristic and the solver core with the exact int32
# signatures the solver uses at run time. It runs from /src/scripts (not /src), so `import nucs` resolves to
# the *installed* package -- the same source the runtime imports -- which is what makes the cache valid then.
# NUMBA_CACHE_DIR persists the cache in the image. Numba keys cached code by (triple, cpu_name, cpu_features);
# left unpinned, cpu_name/cpu_features are auto-detected from the *build* host, so the cache misses on any
# machine with a different CPU model (e.g. a GitHub runner's cache is useless on EC2). Pinning both to fixed
# values makes the baked cache portable across all amd64 hosts: NUMBA_CPU_NAME=x86-64-v3 is the AVX2
# microarchitecture level (Haswell 2013+ / every modern cloud CPU), so codegen stays vectorized while the
# cache key is constant. These ENV values persist into `docker run`, so the runtime computes the same key and
# hits the cache. NOTE: warming takes several minutes (it LLVM-compiles the whole library once).
#
# Numba also stamps each cached entry with the source file's (st_mtime, st_size) and discards the whole index
# if the runtime stat doesn't match. The build host (ext4) records nanosecond mtimes, but the deploy host's
# container storage may report a coarser sub-second precision (e.g. EC2 overlayfs truncates to whole seconds),
# so the baked stamp would never match and EVERY function would recompile. We normalize all source mtimes to a
# fixed whole second BEFORE warming, so the baked stamp has no sub-second component for any filesystem to
# diverge from. This must run before warm_cache.py (the stamp is recorded at compile time).
ENV NUMBA_CACHE_DIR=/opt/numba-cache
ENV NUMBA_CPU_NAME=x86-64-v3
ENV NUMBA_CPU_FEATURES=""
RUN find "$VIRTUAL_ENV/lib" -name '*.py' -exec touch -d @1700000000 {} + \
 && mkdir -p "$NUMBA_CACHE_DIR" \
 && python /src/scripts/warm_cache.py

# Source tree no longer needed.
RUN rm -rf /src

# Fatal: NuCS is registered and selecting it compiles a trivial model (validates the .msc + globals library).
RUN printf 'var 1..3: x;\nconstraint x > 1;\nsolve satisfy;\n' > /tmp/check.mzn \
 && minizinc --solvers | grep -qi 'nucs' \
 && minizinc -c --solver nucs /tmp/check.mzn -o /tmp/check.fzn

# Best-effort (non-fatal): warn if NuCS did not become the default solver. MiniZinc's --verbose output
# format varies, so a mismatch here only prints a warning rather than breaking the build.
RUN minizinc -c --verbose /tmp/check.mzn 2>&1 | grep -qi 'nucs' \
 || echo "WARNING: NuCS is registered but may not be the default solver; run with '--solver nucs' if needed." >&2
