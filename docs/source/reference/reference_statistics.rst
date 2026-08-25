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
* SOLVER_ELAPSED_TIME_MS: the time spent searching, in milliseconds, excluding the time spent by the caller between two solutions

The two filtering counters are additionally broken down per propagator algorithm, for the algorithms that ran
at least once. Each breakdown key suffixes the name of the total it partitions with the algorithm name:

* PROPAGATOR_FILTER_NB_<ALGORITHM>: the number of calls to that algorithm's :code:`compute_domains` method
* PROPAGATOR_FILTER_NO_CHANGE_NB_<ALGORITHM>: how many of those calls resulted in no domain change

A high no-change share on a given algorithm is where wasted propagation is concentrated: a call that prunes
nothing still costs a bucket pop, a gather of its variables' domains, an indirect call and a write-back.

