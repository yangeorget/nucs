% Keep subcircuit propagation native so NuCS uses its alldifferent + subcircuit propagators instead of
% MiniZinc's order-variable decomposition.
%
% Like circuit, a subcircuit's successor values are node labels drawn from the array's index set (here a
% node may also point to itself to opt out of the circuit), which may be non-1-based and whose declared
% domains may be wider than the node set. Rebase to a contiguous 1..n numbering -- self-loops i -> i stay
% self-loops -- before handing the array to the native predicate.
predicate nucs_subcircuit(array [int] of var int: x);

predicate fzn_subcircuit(array [int] of var int: x) =
  if length(x) = 0 then
    true
  else
    let {
      int: l = min(index_set(x)),
      int: n = length(x),
      array [1..n] of var 1..n: y,
    } in
    forall (i in index_set(x)) (y[i - l + 1] = x[i] - l + 1) /\
    nucs_subcircuit(y)
  endif;
