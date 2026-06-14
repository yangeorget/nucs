% Keep lex_less native so NuCS reduces it to its lexleq propagator over sentinel-extended vectors
% instead of MiniZinc's reified bool-chain decomposition.
predicate fzn_lex_less_int(array[int] of var int: x, array[int] of var int: y);
