#####################
Variables and domains
#####################

*********
Variables
*********

In NuCS, variables are not entities in their own right, but integers that index domains.
The number of variables is an unsigned 16-bits integer.

Let's consider three variables with integer domains :math:`[1,10]`.

NuCS could represent these domains as a :code:`numpy.ndarray` of shape :code:`(3,2)`:

- one row for each domain
- the first column corresponds to the minimal values
- the second column corresponds to the maximal values

.. list-table::
   :header-rows: 1

   * - Variable index
     - Domain min
     - Domain max
   * - 0
     - 1
     - 10
   * - 1
     - 1
     - 10
   * - 2
     - 1
     - 10

.. note::
   The reality is a bit more complex, as explained below.

*******
Domains
*******

Shared domains
##############

Let's consider two variables with initial domains :math:`[1, 10]` and the constraint :math:`v_0 = v_1 + 4`.
Because of bound consistency,
the domain of :math:`v_0` is :math:`[5, 10]` and the domain of :math:`v_1` is :math:`[1, 6]`.

There is a lot of overhead here:

- each time we update the domain of :math:`v_0`, we also need to update the domain of :math:`v_1` (and vice versa)
- from the point of view of the variable choice heuristic,
  the problem is made arbitrarily large and it makes no difference whether :math:`v_0` or :math:`v_1` is selected
- from the point of view of the backtracking mechanism, there are two variables to store and restore

We can efficiently replace both domains by a single shared domain and two offsets :math:`4` and :math:`0`:

.. list-table::
   :header-rows: 1

   * - Variable index
     - Offset
     - Shared domain index
   * - 0
     - 4
     - 0
   * - 1
     - 0
     - 0

With:

.. list-table::
   :header-rows: 1

   * - Shared domain index
     - Shared domain min
     - Shared domain max
   * - 0
     - 1
     - 6

.. note::
   In NuCS, actual domains are computed on demand from shared domains and offsets.

The constructor of the :code:`Problem` class
********************************************
Because of shared domains and offsets, the constructor of :code:`Problem` accepts 3 arguments:

- the shared domains as a list of pairs of integers
  (if the minimal and maximal values of the pair are equal, the pair can be replaced by the value)
- a list of integers representing, for each variable, the index of its shared domain
- a list of integers representing, for each variable, the offset of its shared domain

Python, with the help of lists and ranges, makes the construction of complex problems an easy task.
Internally, for greater efficiency, shared domains, domain indices and offsets are stored using :code:`numpy.ndarray`.

A concrete example: the 4-queens problem
****************************************

.. figure:: ../../assets/queens.png
   :alt: 4-queens image

   4 non attacking queens


The 4-queens problem can be modelled as follows:

- for :math:`i` in :math:`[0, 3]`, :math:`v_i` is the vertical position of the queen in the :math:`i` th column
- :math:`v_0, v_1, v_2, v_3` are all different
- :math:`v_0, v_1 + 1, v_2 + 2, v_3 + 3` are all different
- :math:`v_0, v_1 - 1, v_2 - 2, v_3 - 3` are all different

This corresponds to the 12 variables :math:`v_i` with the following relations, for :math:`i` in :math:`[0, 3]`:

- :math:`v_{i+4} = v_i + i`
- :math:`v_{i+8} = v_i - i`
- all the :math:`v_i` are different
- all the :math:`v_{i+4}` are different
- all the :math:`v_{i+8}` are different

The domains of these variables can thus be represented as follows:

.. list-table::
   :header-rows: 1

   * - Variable index
     - Offset
     - Shared domain index
   * - 0
     - 0
     - 0
   * - 1
     - 0
     - 1
   * - 2
     - 0
     - 2
   * - 3
     - 0
     - 3
   * - 4
     - 0
     - 0
   * - 5
     - 1
     - 1
   * - 6
     - 2
     - 2
   * - 7
     - 3
     - 3
   * - 8
     - 0
     - 0
   * - 9
     - -1
     - 1
   * - 10
     - -2
     - 2
   * - 11
     - -3
     - 3

With:

.. list-table::
   :header-rows: 1

   * - Shared domain index
     - Shared domain min
     - Shared domain max
   * - 0
     - 0
     - 3
   * - 1
     - 0
     - 3
   * - 2
     - 0
     - 3
   * - 3
     - 0
     - 3

In NuCS, the n-queens problem is indeed constructed as follows:

.. code-block:: python

   def __init__(self, n: int):
      super().__init__(
         [(0, n - 1)] * n,
         list(range(n)) * 3,
         [0] * n + list(range(n)) + list(range(0, -n, -1)),
      )
      self.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
      self.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
      self.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))

Integer domains
###############

NuCS only support integer domains.
Domains limits are 32-bits integers.

Boolean domains
###############

Boolean domains are simply integer domains of the form :math:`[0, 1]`.

