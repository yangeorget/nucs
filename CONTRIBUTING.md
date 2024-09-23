# Useful scripts and commands to contribute to NUCS

## Code
### How to check the coding style
```bash
./scripts/bash/style.sh    
```

### How to launch the tests
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/
```

### How to compute the code coverage
```bash
NUMBA_DISABLE_JIT=1 PYTHONPATH=. coverage run --source=nucs,tests -m pytest tests
coverage html
```

## Doc
### How to generate the doc
```bash
sphinx-build -M html docs/source docs/output
```

## Performance
### How to profile the code
```bash
NUMBA_DISABLE_JIT=1 ./scripts/bash/profile.sh tests/examples/test_queens.py | more
```

### How to measure the performance
```bash
time NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_queens.py -n 12
```

## Package
### How to build the package
```bash
python -m build
```

### How to publish the package
```bash
python -m twine upload --verbose dist/*
```

