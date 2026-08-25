##########
Statistics
##########

During the computation, NuCS aggregates some :ref:`statistics <statistics>`.

These statistics can then be accessed by calling the solver's :code:`get_statistics_as_dictionary` method which returns
a dictionary of statistics:

.. code-block:: python
   :linenos:

   print(solver.get_statistics_as_dictionary())

The dictionary contains the global counters, followed by a per-algorithm breakdown of the two filtering
counters, restricted to the propagator algorithms that actually ran:

.. code-block:: bash

   {
       'ALG_BC_NB': 22230,
       'PROPAGATOR_ENTAILMENT_NB': 203454,
       'PROPAGATOR_FILTER_NB': 1400872,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 586754,
       'PROPAGATOR_INCONSISTENCY_NB': 11078,
       'SOLVER_BACKTRACK_NB': 11077,
       'SOLVER_CHOICE_NB': 11142,
       'SOLVER_CHOICE_DEPTH': 9,
       'SOLUTION_NB': 10,
       'SOLVER_ELAPSED_TIME_MS': 197,
       'PROPAGATOR_FILTER_NB_ALLDIFFERENT': 44683,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_ALLDIFFERENT': 6484,
       'PROPAGATOR_FILTER_NB_LEQ_C': 360228,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_LEQ_C': 347920,
       'PROPAGATOR_FILTER_NB_SUM_EQ': 995961,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_SUM_EQ': 232350
   }

