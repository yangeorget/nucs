predicate nucs_if_then_else_var_int(array [int] of var bool: c, array [int] of var int: x, var int: y);

predicate fzn_if_then_else_var_int(
  array [int] of var bool: c,
  array [int] of var int: x,
  var int: y,
) = nucs_if_then_else_var_int(c, x, y);
