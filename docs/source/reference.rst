#######################
Reference documentation
#######################

.. _consistency_algorithms:

**********************
Consistency algorithms
**********************

NuCS provides the following consistency algorithms.

.. py:module:: nucs.solvers.bound_consistency_algorithm
.. py:function:: nucs.solvers.bound_consistency_algorithm.bound_consistency_algorithm(statistics, algorithms, var_bounds, param_bounds, dom_indices_arr, dom_offsets_arr, props_dom_indices, props_dom_offsets, props_parameters,shr_domains_propagators, shr_domains_arr, not_entailed_propagators, triggered_propagators, compute_domains_addrs)

   This is the default consistency algorithm used by the :mod:`nucs.solvers.backtrack_solver`.

   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param var_bounds: the variable bounds indexed by propagators
   :type var_bounds: NDArray
   :param param_bounds: the parameters bounds indexed by propagators
   :type param_bounds: NDArray
   :param dom_indices_arr: the domain indices indexed by variables
   :type dom_indices_arr: NDArray
   :param dom_offsets_arr: the domain offsets indexed by variables
   :type dom_offsets_arr: NDArray
   :param props_dom_indices: the domain indices indexed by propagator variables
   :type props_dom_indices: NDArray
   :param props_dom_offsets: the domain offsets indexed by propagator variables
   :type props_dom_offsets: NDArray
   :param props_parameters: the parameters indexed by propagator variables
   :type props_parameters: NDArray
   :param shr_domains_propagators: a Numpy array of booleans indexed by shared domain indices, MIN/MAX and propagators; true means that the propagator has to be triggered when the MIN or MAX of the shared domain has changed
   :type shr_domains_propagators: NDArray
   :param shr_domains_stack: a stack of shared domains
   :type shr_domains_stack: NDArray
   :param not_entailed_propagators_stack: a stack of not entailed propagators
   :type not_entailed_propagators_stack: NDArray
   :param dom_update_stack: a stack of domain updates (domain index and bound)
   :type dom_update_stack: NDArray
   :param stacks_height: the height of the 3 previous stacks
   :type stacks_height: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int


.. py:module:: nucs.solvers.shaving_consistency_algorithm
.. py:function:: nucs.solvers.shaving_consistency_algorithm.shaving_consistency_algorithm(statistics, algorithms, var_bounds, param_bounds, dom_indices_arr, dom_offsets_arr, props_dom_indices, props_dom_offsets, props_parameters,shr_domains_propagators, shr_domains_arr, not_entailed_propagators, triggered_propagators, compute_domains_addrs)

   This algorithm reduces the need of searching by shaving the domains.
   It is experimental.

   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param var_bounds: the variable bounds indexed by propagators
   :type var_bounds: NDArray
   :param param_bounds: the parameters bounds indexed by propagators
   :type param_bounds: NDArray
   :param dom_indices_arr: the domain indices indexed by variables
   :type dom_indices_arr: NDArray
   :param dom_offsets_arr: the domain offsets indexed by variables
   :type dom_offsets_arr: NDArray
   :param props_dom_indices: the domain indices indexed by propagator variables
   :type props_dom_indices: NDArray
   :param props_dom_offsets: the domain offsets indexed by propagator variables
   :type props_dom_offsets: NDArray
   :param props_parameters: the parameters indexed by propagator variables
   :type props_parameters: NDArray
   :param shr_domains_propagators: a Numpy array of booleans indexed by shared domain indices, MIN/MAX and propagators; true means that the propagator has to be triggered when the MIN or MAX of the shared domain has changed
   :type shr_domains_propagators: NDArray
   :param shr_domains_stack: a stack of shared domains
   :type shr_domains_stack: NDArray
   :param not_entailed_propagators_stack: a stack of not entailed propagators
   :type not_entailed_propagators_stack: NDArray
   :param dom_update_stack: a stack of domain updates (domain index and bound)
   :type dom_update_stack: NDArray
   :param stacks_height: the height of the 3 previous stacks
   :type stacks_height: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int

.. _propagators:

***********
Propagators
***********

NuCS currently provides the following highly-optimized propagators.


.. py:module:: nucs.propagators.and_propagator
.. py:function:: nucs.propagators.and_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\&_{i \in [0, n-1[} b_i = b_{n-1}`
   where for each :math:`i`, :math:`b_i` is a boolean variable.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`b` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.affine_eq_propagator
.. py:function:: nucs.propagators.affine_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i = a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray


.. py:module:: nucs.propagators.affine_geq_propagator
.. py:function:: nucs.propagators.affine_geq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i \geq a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray


.. py:module:: nucs.propagators.affine_leq_propagator
.. py:function:: nucs.propagators.affine_leq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i \leq a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray


.. py:module:: nucs.propagators.alldifferent_propagator
.. py:function:: nucs.propagators.alldifferent_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\forall i \neq j, x_i \neq x_j`.

   It is adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".

   It has the time complexity: :math:`O(n \times log(n))` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.count_eq_propagator
.. py:function:: nucs.propagators.count_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (x_i = a) = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is the first parameter
   :type parameters: NDArray


.. py:module:: nucs.propagators.element_lic_propagator
.. py:function:: nucs.propagators.element_lic_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = c`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-1` domains,
          :math:`i` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`c` is the first parameter
   :type parameters: NDArray


.. py:module:: nucs.propagators.element_liv_propagator
.. py:function:: nucs.propagators.element_liv_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = v`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-2` domains,
          :math:`i` is the :math:`n-1` th domain,
          :math:`v` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.exactly_eq_propagator
.. py:function:: nucs.propagators.exactly_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (x_i = a) = c`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          :math:`a` is the first parameter,
          :math:`c` is the second parameter
   :type parameters: NDArray


.. py:module:: nucs.propagators.exactly_true_propagator
.. py:function:: nucs.propagators.exactly_true_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (b_i = 1) = c`
   where for each :math:`i`, :math:`b_i` is a boolean variable.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`b` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          :math:`c` is the first parameter
   :type parameters: NDArray


.. py:module:: nucs.propagators.gcc_propagator
.. py:function:: nucs.propagators.gcc_propagator.compute_domains(domains, parameters)

   This propagator (Global Cardinality Constraint) enforces that
   :math:`\forall j,  l_j \leq |\{ i  / x_i = v_j \}| \leq v_j`.

   It is adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".

   It has the time complexity: :math:`O(n \times log(n))` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, there are :math:`1 + 2 \times m` parameters:
    the first domain value :math:`v_0`, then the :math:`m` lower bounds, then the :math:`m` upper bounds (capacities)
   :type parameters: NDArray


.. py:module:: nucs.propagators.lexicographic_leq_propagator
.. py:function:: nucs.propagators.lexicographic_leq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`x <_{leq} y`.

   See https://www.diva-portal.org/smash/record.jsf?pid=diva2:1041533.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is the list of the first :math:`n` domains,
          :math:`y` is the list of the last :math:`n` domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.max_eq_propagator
.. py:function:: nucs.propagators.max_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\max_i x_i = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.max_leq_propagator
.. py:function:: nucs.propagators.max_leq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\max_i x_i \leq x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.min_eq_propagator
.. py:function:: nucs.propagators.min_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\min_i x_i = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.min_geq_propagator
.. py:function:: nucs.propagators.min_geq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\min_i x_i \geq x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:module:: nucs.propagators.relation_propagator
.. py:function:: nucs.propagators.relation_propagator.compute_domains(domains, parameters)

   This propagator implements a relation over :math:`O(n)` variables defined by its allowed tuples.

   It has the time complexity: :math:`O(p)` where :math:`p` is the number of parameters.

   :param domains: the domains of the variables
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          the allowed tuples correspond to:
          :math:`(p_0, ..., p_{n-1}), (p_n, ..., p_{2n-1}), ...` where :math:`p` is an alias for parameters

   :type parameters: NDArray


.. _heuristics:

**********
Heuristics
**********

.. py:module:: nucs.solvers.heuristics

NUCS provides heuristics for selecting a variable (precisely selecting a shared domain)
and for selecting a value (more generally, reducing the shared domain):

Heuristics for selecting a shared domain
########################################

NUCS provides the following functions for selecting a shared domain.


.. py:function:: nucs.solvers.heuristics.first_not_instantiated_var_heuristic(shr_domains)

   This heuristics chooses the first non-instantiated shared domain.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :return: the index of the shared domain
   :rtype: int


.. py:function:: nucs.solvers.heuristics.last_not_instantiated_var_heuristic(shr_domains)

   This heuristics chooses the last non-instantiated shared domain.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :return: the index of the shared domain
   :rtype: int


.. py:function:: nucs.solvers.heuristics.smallest_domain_var_heuristic(shr_domains)

   This heuristics chooses the smallest shared domain and which is not instantiated.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :return: the index of the shared domain
   :rtype: int


.. py:function:: nucs.solvers.heuristics.greatest_domain_var_heuristic(shr_domains)

   This heuristics chooses the greatest shared domain and which is not instantiated.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :return: the index of the shared domain
   :rtype: int


Heuristics for reducing the chosen shared domain
################################################

NUCS provides the following functions for reducing a shared domain.


.. py:function:: nucs.solvers.heuristics.min_value_dom_heuristic(shr_domains, shr_domains_copy)

   This heuristics chooses the first value of the domain.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :param shr_domains_copy: the copy of the shared domains to be added to the choice points
   :type shr_domains_copy: NDArray
   :return: the MAX event
   :rtype: int


.. py:function:: nucs.solvers.heuristics.max_value_dom_heuristic(shr_domains, shr_domains_copy)

   This heuristics chooses the last value of the domain.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :param shr_domains_copy: the copy of the shared domains to be added to the choice points
   :type shr_domains_copy: NDArray
   :return: the MIN event
   :rtype: int


.. py:function:: nucs.solvers.heuristics.split_low_dom_heuristic(shr_domains, shr_domains_copy)

   This heuristics chooses the first half of the domain.

   :param shr_domains: the shared domains of the variables
   :type shr_domains: NDArray
   :param shr_domains_copy: the copy of the shared domains to be added to the choice points
   :type shr_domains_copy: NDArray
   :return: the MAX event
   :rtype: int


.. _solvers:

*******
Solvers
*******

NuCS comes with the following solvers.


.. py:module:: nucs.solvers.backtrack_solver
.. py:function:: nucs.solvers.backtrack_solver.__init__(problem, consistency_alg_idx, var_heuristic_idx, dom_heuristic_idx, stack_max_height, log_level)

   A backtrack-based solver.

   :param problem: the problem to be solved
   :type problem: Problem
   :param consistency_alg_idx: the index of the consistency algorithm
   :type consistency_alg_idx: int
   :param var_heuristic_idx: the index of the heuristic for selecting a variable/domain
   :type var_heuristic_idx: int
   :param dom_heuristic_idx: the index of the heuristic for reducing a domain
   :type dom_heuristic_idx: int
   :param stack_max_height: the maximal height of the choice point stack
   :type stack_max_height: int
   :param log_level: the log level
   :type log_level: str

.. py:module:: nucs.solvers.multiprocessing_solver
.. py:function:: nucs.solvers.multiprocessing_solver.__init__(solvers, log_level)

   A solver relying on the multiprocessing package. This solver delegates resolution to a set of solvers.

   :param solvers: the solvers used in different processes
   :type solvers: List[BacktrackSolver]
   :param log_level: the log level
   :type log_level: str


.. _statistics:

**********
Statistics
**********

NUCS aggregates the following statistics:

* ALG_BC_NB: the number of calls to the bound consistency algorithm
* ALG_BC_WITH_SHAVING_NB: the number of calls to the bound consistency with shaving algorithm
* ALG_SHAVING_NB: the number of attempts to shave a value
* ALG_SHAVING_CHANGE_NB: the number of successes when attempting to shave a value
* ALG_SHAVING_NO_CHANGE_NB: the number of failures when attempting to shave a value
* PROPAGATOR_ENTAILMENT_NB: the number of calls to a propagator's :code:`compute_domains` method resulting in an entailment
* PROPAGATOR_FILTER_NB: the number of calls to a propagator's :code:`compute_domains` method
* PROPAGATOR_FILTER_NO_CHANGE_NB: the number of calls to a propagator's :code:`compute_domains` method resulting in no domain change
* PROPAGATOR_INCONSISTENCY_NB: the number of calls to a propagator's :code:`compute_domains` method resulting in an inconsistency
* SOLVER_BACKTRACK_NB: the number of calls to the solver's :code:`backtrack` method
* SOLVER_CHOICE_NB: the number of choices that have been made
* SOLVER_CHOICE_DEPTH: the maximal depth of choices
* SOLVER_SOLUTION_NB: the number of solutions that have been found


.. _examples:

********
Examples
********

NUCS comes with the following examples.


.. py:module:: nucs.examples.alpha.alpha_problem
.. py:class:: nucs.examples.alpha.alpha_problem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.bibd.bibd_problem
.. py:class:: nucs.examples.bibd.bibd_problem

This problem is problem `028 <https://www.csplib.org/Problems/prob028>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3 --symmetry_breaking

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.and_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. py:module:: nucs.examples.donald.donald_problem
.. py:class:: nucs.examples.donald.donald_problem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.donald

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.golomb.golomb_problem
.. py:class:: nucs.examples.golomb.golomb_problem

This problem is problem `006 <https://www.csplib.org/Problems/prob006>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10 --symmetry_breaking

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.knapsack.knapsack_problem
.. py:class:: nucs.examples.knapsack.knapsack_problem

This problem is problem `133 <https://www.csplib.org/Problems/prob133>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`.


.. py:module:: nucs.examples.magic_sequence.magic_sequence_problem
.. py:class:: nucs.examples.magic_sequence.magic_sequence_problem

This problem is problem `019 <https://www.csplib.org/Problems/prob019>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence -n 100

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.count_eq_propagator`.


.. py:module:: nucs.examples.magic_square.magic_square_problem
.. py:class:: nucs.examples.magic_square.magic_square_problem

This problem is problem `019 <https://www.csplib.org/Problems/prob019>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_square -n 4 --symmetry_breaking

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.quasigroup.quasigroup_problem
.. py:class:: nucs.examples.quasigroup.quasigroup_problem

This problem is problem `003 <https://www.csplib.org/Problems/prob003>`_ on CSPLib.

The problem QG5, a sub-instance of the quasigroup problem, can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup -n 10 --symmetry_breaking

This problem leverages the propagators:

* :mod:`nucs.propagators.element_liv_propagator`,
* :mod:`nucs.propagators.element_lic_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.queens.queens_problem
.. py:class:: nucs.examples.queens.queens_problem

This problem is problem `054 <https://www.csplib.org/Problems/prob054>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 10

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.schur_lemma.schur_lemma_problem
.. py:class:: nucs.examples.schur_lemma.schur_lemma_problem

This problem is problem `015 <https://www.csplib.org/Problems/prob015>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma -n 20 --symmetry_breaking

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. py:module:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem
.. py:class:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem

This problem is problem `026 <https://www.csplib.org/Problems/prob026>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.sports_tournament_scheduling -n 10 --symmetry_breaking

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.exactly_eq_propagator`,
* :mod:`nucs.propagators.gcc_propagator`,
* :mod:`nucs.propagators.relation_propagator`.


.. py:module:: nucs.examples.sudoku.sudoku_problem
.. py:class:: nucs.examples.sudoku.sudoku_problem

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.