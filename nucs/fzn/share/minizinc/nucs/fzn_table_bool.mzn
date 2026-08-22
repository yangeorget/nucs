% Keep the boolean table native so NuCS uses its global RELATION (extensional) propagator, as it already
% does for the integer table. MiniZinc's decomposition introduces a row-index variable and one
% array_bool_element per column; the propagator replaces all of it with a single extensional constraint.
% Booleans are 0/1 integer variables in NuCS, so the tuples and the variables are both mapped through
% bool2int -- which the adapter absorbs into the boolean's own variable rather than channelling it.
include "fzn_table_int.mzn";

predicate fzn_table_bool(array [int] of var bool: x, array [int, int] of bool: t) =
  nucs_table_int([bool2int(x[i]) | i in index_set(x)], [bool2int(v) | v in array1d(t)]);
