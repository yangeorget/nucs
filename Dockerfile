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
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" /src \
 && rm -rf /src

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

# Bake the Numba JIT cache into the image so the first solve at runtime is fast. The cache is keyed to the
# (installed) package source, so warming it with the installed NuCS makes it valid at runtime too;
# NUMBA_CACHE_DIR persists the cache, NUMBA_CPU_NAME=generic keeps it valid across CPUs of this architecture.
ENV NUMBA_CACHE_DIR=/opt/numba-cache
ENV NUMBA_CPU_NAME=generic
ENV NUMBA_CPU_FEATURES=""

# Warm-up: a small but constraint-diverse model. Solving it JIT-compiles and caches the common propagators
# (alldifferent, increasing, lex, count, element, linear/sum, reified comparisons, global_cardinality).
# This also doubles as a fatal end-to-end smoke test that NuCS solves via MiniZinc. Propagators not exercised
# here still JIT on their first use at runtime; extend this model to warm more of them.
RUN mkdir -p "$NUMBA_CACHE_DIR" \
 && printf 'include "globals.mzn";\n\
array[1..5] of var 0..4: x;\n\
array[1..5] of var 0..4: y;\n\
var 0..5: c;\n\
var 1..5: idx;\n\
var 0..4: v;\n\
constraint all_different(x);\n\
constraint increasing(y);\n\
constraint lex_lesseq(x, y);\n\
constraint count(x, 3, c);\n\
constraint v = x[idx];\n\
constraint int_lin_le([1,1,1,1,1], x, 30);\n\
constraint (x[1] = 0) -> (v >= 1);\n\
constraint global_cardinality(x, [0,1,2,3,4], [0,0,0,0,0], [1,1,1,1,1]);\n\
solve satisfy;\n' > /tmp/warm.mzn \
 && minizinc --solver nucs /tmp/warm.mzn > /dev/null

# Best-effort (non-fatal): warn if NuCS did not become the default solver. MiniZinc's --verbose output
# format varies, so a mismatch here only prints a warning rather than breaking the build.
RUN minizinc -c --verbose /tmp/warm.mzn 2>&1 | grep -qi 'nucs' \
 || echo "WARNING: NuCS is registered but may not be the default solver; run with '--solver nucs' if needed." >&2
