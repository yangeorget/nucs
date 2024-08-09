# How to check the coding style
```
./scripts/bash/style.sh    
```

# How to launch the tests
```
PYTHONPATH=. pytest tests/
```

# How to profile the code
```
NUMBA_DISABLE_JIT=1 ./scripts/bash/profile.sh tests/examples/test_queens.py | more
```

# How to measure the performance
```
time PYTHONPATH=. python tests/examples/test_queens.py -n 12
```

