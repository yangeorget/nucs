% Keep value_precede native so NuCS uses its value_precede propagator
% instead of MiniZinc's reified bool-chain decomposition.
predicate fzn_value_precede_int(int: s, int: t, array[int] of var int: x);
