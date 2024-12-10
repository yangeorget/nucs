##########
Statistics
##########

During the computation, NuCS aggregates some :ref:`statistics <statistics>`.

These statistics can then be accessed by calling the solver's :code:`get_statistics` method which returns a dictionary of statistics:

.. code-block:: python
   :linenos:

   print(solver.get_statistics())

