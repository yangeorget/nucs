predicate nucs_member_int_reif(array [int] of int: x, var int: y, var bool: b);

% an array of constants: the reified membership propagator
predicate fzn_member_int_reif(array [int] of int: x, var int: y, var bool: b) = nucs_member_int_reif(x, y, b);

% an array of variables: MiniZinc's standard decomposition, restated since this file shadows the standard one
predicate fzn_member_int_reif(array [int] of var int: x, var int: y, var bool: b) =
  b <-> exists (i in index_set(x)) (x[i] == y);
