######################
Consistency algorithms
######################

NuCS relies on :ref:`consistency algorithms <consistency_algorithms>`.
Some consistency algorithms are provided, custom consistency algorithms can be defined and used instead.


***************************
Bound consistency algorithm
***************************
NuCS provides :mod:`nucs.solvers.bound_consistency_algorithm` which is the default consistency algorithm.


****************************************
Bound consistency algorithm with shaving
****************************************
NuCS provides :mod:`nucs.solvers.shaving_consistency_algorithm` which performs some shaving of the domains.
Note that this algorithm is experimental and is subject to change.


*****************************
Custom consistency algorithms
*****************************
NuCS makes it possible to define and use custom consistency algorithms.

The :mod:`nucs.examples.golomb.golomb_problem` model defines a custom consistency algorithm adapted to the Golomb ruler problem.

This custom consistency algorithm needs to be registered before it is used.

.. code-block:: python
   :linenos:

   consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
   solver = BacktrackSolver(problem, consistency_alg_idx=consistency_alg_golomb)

