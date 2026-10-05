#!/bin/bash

# Lints, formats and type-checks nucs, tests and scripts. With --check, it only reports, as the CI and the git
# pre-commit hook do: nothing is rewritten.

if [ "$1" = "--check" ]; then
  ruff check nucs tests scripts && \
  ruff format --check nucs tests scripts && \
  mypy nucs tests scripts
else
  ruff check --fix nucs tests scripts && \
  ruff format nucs tests scripts && \
  mypy nucs tests scripts
fi
