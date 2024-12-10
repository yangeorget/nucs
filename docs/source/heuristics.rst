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


