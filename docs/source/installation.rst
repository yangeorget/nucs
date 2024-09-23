Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']





## How to use NUCS ?
It is very simple to get started with NUCS.
You can either install the Pip package or install NUCS from the sources.

### Install the NUCS package
Let's install the Pip package for NUCS:
```bash
pip install nucs
````
Now we can write the following `queens.py` program (please refer to [the technical documentation](DOCUMENTATION.md)
to better understand how NUCS works under the hood):
```python
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.propagators.propagators import ALG_ALLDIFFERENT

n = 8  # the number of queens
problem = Problem(
    [(0, n - 1)] * n,  # these n domains are shared between the 3n variables with different offsets
    list(range(n)) * 3,  # for each variable, its shared domain
    [0] * n + list(range(n)) + list(range(0, -n, -1))  # for each variable, its offset
)
problem.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
problem.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
problem.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))
print(BacktrackSolver(problem).solve_one()[:n])
```
Let's run this model with the following command:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python queens.py
```
The first solution found is:
```bash
[0, 4, 7, 5, 2, 6, 1, 3]
```
> [!TIP]
> Note that the second run will always be __much faster__ since the Python code will already have been compiled and cached by Numba.

### Install NUCS from the sources
Let's install NUCS from the sources by cloning the NUCS Github repository:
```bash
git clone https://github.com/yangeorget/nucs.git
pip install -r requirements.txt
```
From there, we will launch some NUCS examples.

#### Run some examples
Some of the examples come with a command line interface and can be run directly.

Let's find all solutions to the [12-queens problem](https://www.csplib.org/Problems/prob054/):
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_queens.py -n 12
```

Let's find the optimal solution to the [Golomb ruler problem](https://www.csplib.org/Problems/prob006/) with 10 marks:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_golomb.py -n 10
```
