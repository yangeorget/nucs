#!/bin/bash

echo "Running isort" && \
isort --trailing-comma nucs tests && \
echo "Running black" && \
black nucs tests && \
echo "Running flake8" && \
flake8 nucs tests && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports nucs tests
