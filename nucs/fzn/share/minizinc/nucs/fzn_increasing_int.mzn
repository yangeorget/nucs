% Keep increasing native so NuCS uses its increasing propagator
% instead of MiniZinc's chain of int_lin_le decomposition.
predicate fzn_increasing_int(array[int] of var int: x);
