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
   def my_var_heuristic(decision_variables: NDArray, domains: NDArray, params: NDArray) -> int:
       for variable in decision_variables:
           if domains[variable, MIN] < domains[variable, MAX]:
               return variable
       return -1

:code:`domains` is the current domains as a :code:`(domain_nb, 2)` array indexed by variable then by
:code:`MIN` / :code:`MAX`.


Domain heuristics
#################

A domain heuristic says **where** to split the chosen variable's domain; it does not split it. It writes
the split value into :code:`decision` and returns the kind of split:

============================ ======================= ==========================================
kind                         explored branch         parked alternatives (resumed in this order)
============================ ======================= ==========================================
:code:`DECISION_LE`          :code:`[min, value]`    :code:`[value + 1, max]`
:code:`DECISION_GT`          :code:`[value + 1, max]` :code:`[min, value]`
:code:`DECISION_EQ`          :code:`[value, value]`  :code:`[min, value - 1]` then :code:`[value + 1, max]`
============================ ======================= ==========================================

.. code-block:: python
   :linenos:

   @njit(cache=True)
   def my_dom_heuristic(domains: NDArray, variable: int, params: NDArray, decision: NDArray) -> int:
       decision[DECISION_VALUE] = (domains[variable, MIN] + domains[variable, MAX]) >> 1
       return DECISION_LE

The heuristic mutates nothing: the solver applies the decision, maintains the unbound-variable count and
schedules the propagators the split wakes. A :code:`DECISION_EQ` value outside the domain is clamped into
it, so a split is always a partition of the domain and the enumeration stays complete.

.. warning::

   Both heuristic signatures changed in NuCS 15. A variable heuristic used to receive the whole stack of
   domains and the index of its top; it now receives the current domains directly. A domain heuristic used
   to receive the stacks and write both branches itself; it is now a pure function of the domains. A
   heuristic written against the old signatures will fail to compile against the new ones.
