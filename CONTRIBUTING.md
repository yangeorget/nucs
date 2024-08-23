# Contributing

## How to check the coding style
```bash
./scripts/bash/style.sh    
```

## How to launch the tests
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/
```

## How to profile the code
```bash
NUMBA_DISABLE_JIT=1 ./scripts/bash/profile.sh tests/examples/test_queens.py | more
```

## How to measure the performance
```bash
time NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_queens.py -n 12
```

