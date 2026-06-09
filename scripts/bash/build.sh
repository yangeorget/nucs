#!/bin/bash

python -c "from nucs.fzn.register import sync_template_msc; sync_template_msc()" && \
rm -rf dist build && \
python -m build && \
twine check dist/*
