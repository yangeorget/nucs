% Keep value_precede_chain native so NuCS decomposes it into pairwise value_precede propagators
% instead of MiniZinc's seq_precede_chain decomposition.
%
% The parameter names match the standard library's declaration (T, X): MiniZinc is rolling out named
% arguments for native predicates, so a divergent name is a future error rather than a cosmetic difference.
predicate fzn_value_precede_chain_int(array [int] of int: T, array [int] of var int: X);
