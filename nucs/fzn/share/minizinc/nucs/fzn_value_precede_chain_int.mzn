% Keep value_precede_chain native so NuCS decomposes it into pairwise value_precede propagators
% instead of MiniZinc's seq_precede_chain decomposition.
predicate fzn_value_precede_chain_int(array[int] of int: c, array[int] of var int: x);
