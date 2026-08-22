% Keep subcircuit propagation native so NuCS uses its alldifferent + subcircuit propagators instead of
% MiniZinc's order-variable decomposition.
%
% Like circuit, a subcircuit's successor values are node labels drawn from the array's index set (here a
% node may also point to itself to opt out of the circuit), which may be non-1-based and whose declared
% domains may be wider than the node set. The propagator takes that numbering as an offset parameter, so
% the array is handed over as it is -- self-loops i -> i stay self-loops under any numbering -- instead of
% being shifted into a rebased copy.
predicate nucs_subcircuit(array [int] of var int: x, int: offset);

predicate fzn_subcircuit(array [int] of var int: x) =
  if length(x) = 0 then
    true
  else
    forall (i in index_set(x)) (x[i] in index_set(x)) /\
    nucs_subcircuit(x, min(index_set(x)))
  endif;
