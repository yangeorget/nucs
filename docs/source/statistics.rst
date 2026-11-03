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
       'ALG_BC_NB': 21413,
       'PROPAGATOR_ENTAILMENT_NB': 198429,
       'PROPAGATOR_FILTER_NB': 1392451,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 578465,
       'PROPAGATOR_INCONSISTENCY_NB': 10162,
       'SOLVER_BACKTRACK_NB': 11176,
       'SOLVER_CHOICE_NB': 11241,
       'SOLVER_CHOICE_DEPTH': 9,
       'SOLVER_RESTART_NB': 0,
       'SOLUTION_NB': 10,
       'SOLVER_ELAPSED_TIME_MS': 197,
       'PROPAGATOR_FILTER_NB_ALLDIFFERENT': 44850,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_ALLDIFFERENT': 6567,
       'PROPAGATOR_FILTER_NB_LEQ_C': 361051,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_LEQ_C': 348864,
       'PROPAGATOR_FILTER_NB_SUM_EQ': 986550,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_SUM_EQ': 223034
   }

