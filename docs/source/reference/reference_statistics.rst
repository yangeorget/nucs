.. _statistics:

**********
Statistics
**********

NUCS aggregates the following statistics:

* ALG_BC_NB: the number of calls to the bound consistency algorithm
* PROPAGATOR_ENTAILMENT_NB: the number of calls to a propagator's :code:`compute_domains` method resulting in an entailment
* PROPAGATOR_FILTER_NB: the number of calls to a propagator's :code:`compute_domains` method
* PROPAGATOR_FILTER_NO_CHANGE_NB: the number of calls to a propagator's :code:`compute_domains` method resulting in no domain change
* PROPAGATOR_INCONSISTENCY_NB: the number of calls to a propagator's :code:`compute_domains` method resulting in an inconsistency
* SOLVER_BACKTRACK_NB: the number of calls to the solver's :code:`backtrack` method
* SOLVER_CHOICE_NB: the number of choices that have been made
* SOLVER_CHOICE_DEPTH: the maximal depth of choices
* SOLUTION_NB: the number of solutions that have been found

