.. _problems:

********
Problems
********

NUCS comes with a few pre-defined problems:


.. py:module:: nucs.problems.circuit_problem
.. py:function:: nucs.problems.circuit_problem.CircuitProblem.__init__(self, n)

   A circuit of size :math:`n` is a permutation of :math:`[0, n-1]` with no sub-cycle.

   :param n: the number of nodes in the circuit
   :type n: int


.. py:module:: nucs.problems.latin_square_problem
.. py:function:: nucs.problems.latin_square_problem.LatinSquareProblem.__init__(self, colors, givens)

   A latin square of size :math:`n` is a :math:`n` by :math:`n` square where all the values in each column or row are different.

   :param colors: the possible values for the cells, usually :math:`[0, ..., n-1]` except in some cases (eg Sudokus) where :math:`[1, ..., n]` is preferred; the number of colors is also the size of the square
   :type colors: List[int]
   :param givens: initial values for the cells, any value different from the possible colors is used as a wildcard
   :type givens: Optional[List[List[int]]]


.. py:function:: nucs.problems.latin_square_problem.LatinSquareRCProblem.__init__(self, n)

   A latin square of size :math:`n` is a :math:`n x n` square where all the values in each column or row are different.
   This class has implements a complete (row, column, color) model.

   :param n: the size of the latin square
   :type n: int


.. py:module:: nucs.problems.permutation_problem
.. py:function:: nucs.problems.permutation_problem.PermutationProblem.__init__(self, n)

   A permutation of size :math:`n` is a bijection from :math:`[0, n-1]` to :math:`[0, n-1]`

   :param n: the number of values
   :type n: int