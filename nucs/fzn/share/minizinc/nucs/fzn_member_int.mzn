predicate nucs_member_int(array [int] of int: x, var int: y);

% an array of constants: the membership propagator
predicate fzn_member_int(array [int] of int: x, var int: y) = nucs_member_int(x, y);

% an array of variables: MiniZinc's standard decomposition, restated since this file shadows the standard one
predicate fzn_member_int(array [int] of var int: x, var int: y) = exists (i in index_set(x)) (x[i] == y);
