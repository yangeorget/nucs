######################
Consistency algorithms
######################

NuCS relies on :ref:`consistency algorithms <consistency_algorithms>`.
Some consistency algorithms are provided, custom consistency algorithms can be defined and used instead.


***************************
Bound consistency algorithm
***************************
NuCS provides :mod:`nucs.solvers.backtrack.bc_algorithm` which is the default consistency algorithm.


*****************************
Custom consistency algorithms
*****************************
NuCS makes it possible to define and use custom consistency algorithms.

The :mod:`nucs.examples.golomb.golomb_problem` model defines a custom consistency algorithm adapted to the Golomb ruler problem.

This custom consistency algorithm needs to be registered before it is used.

.. code-block:: python
   :linenos:

   consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
   solver = BacktrackSolver(problem, consistency_algorithm=consistency_alg_golomb)

A custom consistency algorithm is compiled against :code:`SIGN_CONSISTENCY_ALG` in
:mod:`nucs.solvers.backtrack.consistency_algorithms`, so it takes the same arguments as
:func:`nucs.solvers.backtrack.bc_algorithm.bc_algorithm`. One of them, :code:`compute_domains_addrs`, holds the compiled
address of each propagator algorithm's :code:`compute_domains` function: pass it on unchanged, to
:code:`bc_algorithm` or to :func:`~nucs.propagators.propagators.call_compute_domains`. Another one,
:code:`propagator_weights`, holds the failure weight of each propagator: pass it on unchanged to
:code:`bc_algorithm`, which records the failures in it, or record them yourself with
:func:`nucs.solvers.weights.weights_bump`.

.. warning::

   In NuCS 17 :code:`propagator_weights` became the second argument of a consistency algorithm, after
   :code:`statistics`. A consistency algorithm written against the old signature will fail to compile.



Writing a domain
################

.. warning::

   In NuCS 15 a consistency algorithm receives the current domains rather than a stack of them, and
   **must** write them through :func:`nucs.solvers.state.tighten_at`. Assigning
   a domain directly still compiles and still looks right, but the write is not recorded on the undo trail:
   it is never undone, and it survives the backtrack into sibling subtrees. That is silent unsoundness,
   with no error to point at.

:code:`tighten_at` also owns the groundness test and the unbound-variable count, so routing through it is
what keeps the solver's "is this problem solved?" test correct. It returns the events the write raises,
:code:`EVENT_MASK_NONE` when it changed nothing, which is what to pass to
:func:`~nucs.solvers.propagation_steps.update_propagators`, and the new undo trail size. Scheduling stays with
the caller, and so does the undo trail size: load it from :code:`utrail_top` once, thread it through the
writes, and store it back once.

.. code-block:: python
   :linenos:

   utrail_sz = utrail_top[0]
   events, utrail_sz = tighten_at(state, utrail_log, utrail_indices, mark, utrail_sz, variable, new_min, new_max)
   if events:
       update_propagators(triggered_propagators, entailed, triggers, triggers_offsets,
                          priorities, variable, events, -1)
   utrail_top[0] = utrail_sz

where :code:`mark` is :code:`choicepoints[choicepoint_top[0], CHOICEPOINT_UTRAIL_MARK]`, the undo trail position
the current choice point branched at, and the last argument of :code:`update_propagators` is a propagator not to wake,
which the propagation loop uses for an idempotent propagator's own writes: pass -1.
:mod:`nucs.examples.golomb.golomb_problem` is the in-tree example.

Propagators need no such care and are unchanged: a :code:`compute_domains_*` function receives a gathered
copy of only its own variables' domains and mutates that copy, and the scatter back through the barrier
happens in the consistency algorithm.

A propagator can have a variable more than once: :code:`count_eq` in the magic sequence example has
:code:`x[i]` in its list and as its count. The propagator sees each copy as a variable of its own, so it can
narrow the copies to different bounds, even to bounds with no common value, and still not fail.
:code:`offsets[p, OFFSETS_COPIES]` (:mod:`nucs.problems.problem`) is 1 for such a propagator. A consistency
algorithm that writes back the domains of such a propagator itself must:

- intersect each copy with the current bounds, and fail when the intersection is empty;
- not skip a variable that an earlier copy fixed;
- call the propagator again when a copy is wider than the bounds that the write-back leaves.

:func:`~nucs.solvers.propagation_steps.update_domains_of_copies` does this; a consistency algorithm that calls
:func:`~nucs.solvers.backtrack.bc_algorithm.bc_algorithm` for the propagators, as the Golomb one does, gets it with no
change.
