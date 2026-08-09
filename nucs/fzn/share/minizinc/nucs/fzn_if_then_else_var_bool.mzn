% Keep if_then_else_var_bool native so NuCS uses its dedicated IF_THEN_ELSE propagator
% (y = x[k] for the first true condition c[k]) instead of MiniZinc's standard decomposition
% into a prefix "no earlier condition" chain plus a per-branch reified equality, which for a
% var-bool result flattens into a chain of raw clauses.
predicate nucs_if_then_else_var_bool(array [int] of var bool: c, array [int] of var bool: x, var bool: y);

predicate fzn_if_then_else_var_bool(
  array [int] of var bool: c,
  array [int] of var bool: x,
  var bool: y,
) = nucs_if_then_else_var_bool(c, x, y);
