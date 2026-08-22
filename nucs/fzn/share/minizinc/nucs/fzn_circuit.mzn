% Keep circuit propagation native so NuCS uses its global alldifferent + no_sub_cycle propagators instead
% of MiniZinc's auxiliary position-variable decomposition.
%
% A FlatZinc circuit's successor values are node labels drawn from the array's index set, which may be
% non-1-based, and whose declared variable domains may be wider than the node set (e.g. var -100..100 over
% an index set 2..5). The propagator takes that numbering as an offset parameter, so the array is handed
% over as it is rather than shifted into a rebased copy -- which used to cost one auxiliary variable and one
% offset propagator per node. Restating the index set as a domain constraint costs nothing (MiniZinc folds
% it into the declarations) and gives the rest of the model the tightened successor domains.
predicate nucs_circuit(array [int] of var int: x, int: offset);

predicate fzn_circuit(array [int] of var int: x) =
  if length(x) = 0 then
    true
  else
    forall (i in index_set(x)) (x[i] in index_set(x)) /\
    nucs_circuit(x, min(index_set(x)))
  endif;
