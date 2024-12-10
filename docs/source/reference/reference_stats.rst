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

