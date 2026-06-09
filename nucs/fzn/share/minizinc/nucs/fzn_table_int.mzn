% Keep table native so NuCS uses its global RELATION (extensional) propagator
% instead of MiniZinc's array_int_element decomposition.
% FlatZinc builtin arguments must be one-dimensional, so the 2D table is flattened
% (row-major) with array1d before being passed to the native nucs_table_int predicate.
predicate nucs_table_int(array[int] of var int: x, array[int] of int: t);

predicate fzn_table_int(array[int] of var int: x, array[int, int] of int: t) =
  nucs_table_int(x, array1d(t));
