#!/bin/bash

echo "Running isort" && \
isort --trailing-comma ncs tests && \
echo "Running black" && \
black ncs tests && \
echo "Running flake8" && \
flake8 ncs tests && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports ncs tests
