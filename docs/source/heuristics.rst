##########
Heuristics
##########

NuCS comes with some pre-defined :ref:`heuristics <heuristics>` and makes it possible to design custom heuristics.


*****************
Custom heuristics
*****************

NuCS makes it possible to define and use custom heuristics.

A heuristic needs to be registered before it is used.
The following code registers the :code:`SPLIT_LOW` heuristic.

.. code-block:: python
   :linenos:

   DOM_HEURISTIC_SPLIT_LOW = register_dom_heuristic(split_low_dom_heuristic)


Variable heuristics
###################

A variable heuristic chooses the variable to branch on, and returns :code:`-1` when it can claim none.

.. code-block:: python
   :linenos:

   @njit(cache=True)
   def my_var_heuristic(
       decision_variables: NDArray,
       domains: NDArray,
       entailed: NDArray,
       offsets: NDArray,
       propagator_variables: NDArray,
       variable_propagators_offsets: NDArray,
       variable_propagators: NDArray,
       propagator_weights: NDArray,
       params: NDArray,
   ) -> int:
       for variable in decision_variables:
           if domains[variable, DOMAIN_MIN] < domains[variable, DOMAIN_MAX]:
               return variable
       return -1

:code:`domains` is the current domains as a :code:`(domain_nb, 2)` array indexed by variable then by
:code:`DOMAIN_MIN` / :code:`DOMAIN_MAX`.

The six arguments between :code:`domains` and :code:`params` describe the constraint network and its failures, for a heuristic
that needs it:

- :code:`entailed[p]` is not zero when propagator :code:`p` is entailed,
- the variables of propagator :code:`p` are
  :code:`propagator_variables[offsets[p, OFFSETS_VARIABLE]:offsets[p + 1, OFFSETS_VARIABLE]]`,
- the propagators of variable :code:`x` are
  :code:`variable_propagators[variable_propagators_offsets[x]:variable_propagators_offsets[x + 1]]`,
- :code:`propagator_weights[p]` is the failure weight of propagator :code:`p`, as the dom/wdeg heuristic reads it.

A heuristic must only read them.

Failure-count search
####################

:code:`VAR_HEURISTIC_DOM_WDEG` is dom/wdeg (Choco's :code:`domOverWDeg`): it chooses the variable with the
smallest ratio of domain size to the sum of the failure weights of its live propagators. The weights are learned
during the search and are never backtracked. The FlatZinc selector :code:`dom_w_deg` uses it.

.. code-block:: python
   :linenos:

   solver = BacktrackSolver(problem, var_heuristic=VAR_HEURISTIC_DOM_WDEG)

With :code:`weight_decay` below 1, the recent failures count more than the old ones, as in Gecode's AFC:

.. code-block:: python
   :linenos:

   solver = BacktrackSolver(problem, var_heuristic=VAR_HEURISTIC_DOM_WDEG, weight_decay=0.95)


Restarts and last-conflict
##########################

dom/wdeg learns which constraints fail, but without restarts it can only change the order of the decisions below the
first bad ones. A restart stops the descent after a number of failures and starts again from the root, where the
learned weights choose better first decisions:

.. code-block:: python
   :linenos:

   solver = BacktrackSolver(
       problem, var_heuristic=VAR_HEURISTIC_DOM_WDEG, restart_policy=RESTART_LUBY, restart_scale=100
   )

The policies of :mod:`nucs.solvers.restarts` are those of MiniZinc: :code:`RESTART_LUBY` (the scale times the
Luby sequence 1, 1, 2, 1, 1, 2, 4, ...), :code:`RESTART_GEOMETRIC` (the scale times :code:`restart_base` to the
power of the restart number), :code:`RESTART_LINEAR`, :code:`RESTART_CONSTANT` and :code:`RESTART_NONE`, the default.
A limit counts failures.

- A restart keeps the search sound. When optimizing, the best solution so far is applied again at the root, so the
  search finds only better ones. When enumerating, the restarts stop at the first solution, so that no solution is
  found twice.
- The Luby, geometric and linear limits have no upper bound, so the search stays complete: it still proves that a
  problem has no solution, or that a solution is optimal. A constant limit does not.
- Restarts help only a search that learns. With :code:`VAR_HEURISTIC_FIRST_NOT_INSTANTIATED` or
  :code:`VAR_HEURISTIC_SMALLEST_DOMAIN`, each descent starts with the same decisions again.

Last-conflict (Lecoutre et al. 2009) is independent of the restarts and of the heuristic. After a decision leads to a
failure, the search branches on the variable of that decision first, as long as it is unbound. When this variable
then fails on all its values, the search refutes an earlier decision and tries the conflict variable again at once,
which tells quickly if the earlier decision was the cause:

.. code-block:: python
   :linenos:

   solver = BacktrackSolver(problem, var_heuristic=VAR_HEURISTIC_DOM_WDEG, last_conflict=True)

In a sequential search, the conflict variable waits until the search that owns it has the decision.


Domain heuristics
#################

A domain heuristic says **where** to split the chosen variable's domain; it does not split it. It returns
the kind of split and the value to split at:

============================ ======================== ==========================================
kind                         explored branch          parked alternatives (resumed in this order)
============================ ======================== ==========================================
:code:`DECISION_LE`          :code:`[min, value]`     :code:`[value + 1, max]`
:code:`DECISION_GT`          :code:`[value + 1, max]` :code:`[min, value]`
:code:`DECISION_EQ`          :code:`[value, value]`   :code:`[min, value - 1]` then :code:`[value + 1, max]`
============================ ======================== ==========================================

.. code-block:: python
   :linenos:

   @njit(cache=True)
   def my_dom_heuristic(domains: NDArray, variable: int, params: NDArray) -> tuple[int, int]:
       return DECISION_LE, (domains[variable, DOMAIN_MIN] + domains[variable, DOMAIN_MAX]) >> 1

The heuristic mutates nothing: the solver applies the decision, maintains the unbound-variable count and
schedules the propagators the split wakes. The pair is :code:`int32`; return values in the domain's range
and no cast is needed. A :code:`DECISION_EQ` value outside the domain is clamped into
it, so a split is always a partition of the domain and the enumeration stays complete.

.. warning::

   Both heuristic signatures changed in NuCS 15. A variable heuristic used to receive the whole stack of
   domains and the index of its top; it now receives the current domains directly. A domain heuristic used
   to receive the stacks and write both branches itself; it is now a pure function of the domains. A
   heuristic written against the old signatures will fail to compile against the new ones.

.. warning::

   The variable heuristic signature changed again in NuCS 17: six arguments that describe the constraint network
   and the failure weights come between :code:`domains` and :code:`params`. A variable heuristic written against
   the old signature will fail to compile. Add the six arguments; a heuristic that does not need them ignores them.
