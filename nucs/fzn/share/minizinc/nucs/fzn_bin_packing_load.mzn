% Keep bin_packing_load native so NuCS uses its global propagator instead of
% MiniZinc's decomposition into one reified equality per (item, bin) pair.
predicate fzn_bin_packing_load(array[int] of var int: load, array[int] of var int: bin, array[int] of int: w);
