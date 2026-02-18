#!/bin/bash

ruff check nucs tests && \
ruff format nucs tests && \
mypy --disallow-untyped-defs --ignore-missing-imports nucs tests
