predicate nucs_member_int_imp(array [int] of int: x, var int: y, var bool: b);

% an array of constants: the half-reified membership propagator
predicate fzn_member_int_imp(array [int] of int: x, var int: y, var bool: b) = nucs_member_int_imp(x, y, b);

% an array of variables: the implication of MiniZinc's standard decomposition
predicate fzn_member_int_imp(array [int] of var int: x, var int: y, var bool: b) =
  b -> exists (i in index_set(x)) (x[i] == y);
