% Keep strictly_increasing native so NuCS uses its strictly_increasing propagator
% instead of MiniZinc's chain of int_lin_le decomposition.
predicate fzn_strictly_increasing_int(array[int] of var int: x);
