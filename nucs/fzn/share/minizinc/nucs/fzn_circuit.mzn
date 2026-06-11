% Keep circuit native so NuCS uses its global alldifferent + no_sub_cycle propagators
% instead of MiniZinc's auxiliary position-variable decomposition.
predicate fzn_circuit(array[int] of var int: x);
