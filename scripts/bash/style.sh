#!/bin/bash

ruff check --fix nucs tests && \
ruff format nucs tests && \
mypy nucs tests
