% Keep circuit propagation native so NuCS uses its global alldifferent + no_sub_cycle propagators instead
% of MiniZinc's auxiliary position-variable decomposition.
%
% A FlatZinc circuit's successor values are node labels drawn from the array's index set, which may be
% non-1-based and whose declared variable domains may be wider than the node set (e.g. var -100..100 over
% an index set 2..5). Rebase to a contiguous 1..n numbering -- which also tightens the successor domains to
% the node set -- before handing the array to the native predicate.
predicate nucs_circuit(array [int] of var int: x);

predicate fzn_circuit(array [int] of var int: x) =
  if length(x) = 0 then
    true
  else
    let {
      int: l = min(index_set(x)),
      int: n = length(x),
      array [1..n] of var 1..n: y,
    } in
    forall (i in index_set(x)) (y[i - l + 1] = x[i] - l + 1) /\
    nucs_circuit(y)
  endif;
