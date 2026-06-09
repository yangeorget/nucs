% Keep all_different native so NuCS uses its global alldifferent propagator
% instead of MiniZinc's int_lin_ne decomposition.
predicate fzn_all_different_int(array[int] of var int: x);
