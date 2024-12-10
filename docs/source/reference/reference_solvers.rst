.. _solvers:

*******
Solvers
*******

NuCS comes with the following solvers.


.. py:module:: nucs.solvers.backtrack_solver
.. py:function:: nucs.solvers.backtrack_solver.__init__(problem, consistency_alg_idx, decision_domains, var_heuristic_idx, dom_heuristic_idx, stack_max_height, log_level)

   A backtrack-based solver.

   :param problem: the problem to be solved
   :type problem: Problem
   :param consistency_alg_idx: the index of the consistency algorithm
   :type consistency_alg_idx: int
   :type decision_domains: Optional[List[int]]
   :param decision_domains: the list of domain indices on which choices will be made or None in which case all domains are used
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

