NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 8
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 8 --find-all
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 8 --find-all --no-display-solutions
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 12 --find-all --no-display-solutions

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.jobshop
cat datasets/examples/jobshop/mt06.json

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.tsp
cat datasets/examples/tsp/gr17.json

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.sudoku

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.cryptarithmetic --var-heuristic 2 (greatest domain)
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.cryptarithmetic --var-heuristic 1 (first non instantiated)
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.cryptarithmetic --var-heuristic 6 (smallest domain)