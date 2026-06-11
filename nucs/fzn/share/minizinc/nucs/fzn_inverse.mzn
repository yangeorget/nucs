% Keep inverse native so NuCS uses its global inverse (channeling) propagator
% instead of MiniZinc's element-based decomposition.
predicate fzn_inverse(array[int] of var int: f, array[int] of var int: invf);
