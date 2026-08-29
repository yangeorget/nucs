######################
Consistency algorithms
######################

NuCS relies on :ref:`consistency algorithms <consistency_algorithms>`.
Some consistency algorithms are provided, custom consistency algorithms can be defined and used instead.


***************************
Bound consistency algorithm
***************************
NuCS provides :mod:`nucs.solvers.bc_algorithm` which is the default consistency algorithm.


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



Writing a domain
################

.. warning::

   In NuCS 15 a consistency algorithm receives the current domains rather than a stack of them, and
   **must** write them through :func:`nucs.solvers.choice_points.tighten` (or
   :func:`~nucs.solvers.choice_points.tighten_at`, which threads the trail size through a loop). Assigning
   a domain directly still compiles and still looks right, but the write is not recorded on the trail:
   it is never undone, and it survives the backtrack into sibling subtrees. That is silent unsoundness,
   with no error to point at.

:code:`tighten` also owns the groundness test and the unbound-variable count, so routing through it is
what keeps the solver's "is this problem solved?" test correct. It returns the events the write raises,
:code:`EVENT_MASK_NONE` when it changed nothing, which is what to pass to
:func:`~nucs.propagators.propagators.update_propagators`. Scheduling stays with the caller.

.. code-block:: python
   :linenos:

   events = tighten(state, trail_log, trail_top, trail_indices, mark, variable, new_min, new_max)
   if events:
       update_propagators(triggered_propagators, entailed, triggers, triggers_offsets,
                          priorities, propagator_nb, variable, events)

where :code:`mark` is :code:`choice_point_stk[choice_point_top[0], CHOICE_POINT_TRAIL_MARK]`, the trail position the current
choice point branched at. :mod:`nucs.examples.golomb.golomb_problem` is the in-tree example.

Propagators need no such care and are unchanged: a :code:`compute_domains_*` function receives a gathered
copy of only its own variables' domains and mutates that copy, and the scatter back through the barrier
happens in the consistency algorithm.
