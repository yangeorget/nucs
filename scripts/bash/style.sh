#!/bin/bash

echo "Running isort" && \
isort --trailing-comma ncs tests scripts/python && \
echo "Running black" && \
black ncs tests scripts/python && \
echo "Running flake8" && \
flake8 ncs tests scripts/python && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports ncs tests scripts/python
