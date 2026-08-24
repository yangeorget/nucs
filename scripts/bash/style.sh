#!/bin/bash

ruff check --fix nucs tests scripts && \
ruff format nucs tests scripts && \
mypy nucs tests
