% Keep bin_packing_load native so NuCS uses its global propagator instead of
% MiniZinc's decomposition into one reified equality per (item, bin) pair.
%
% The load array is reindexed to 1..n when the model is flattened, but the *values* of the bin variables
% keep the model's own bin labels, which are the elements of load's index set and may not start at 1
% (e.g. `array [0..2] of var int: load`). That offset is a compile-time property of the index set and
% cannot be recovered from the FlatZinc, so it is passed explicitly to the NuCS predicate -- which is what
% the propagator's bin-offset parameter expects.
predicate nucs_bin_packing_load(
  array [int] of var int: load,
  array [int] of var int: bin,
  array [int] of int: w,
  int: offset,
);

predicate fzn_bin_packing_load(
  array [int] of var int: load,
  array [int] of var int: bin,
  array [int] of int: w,
) = nucs_bin_packing_load(load, bin, w, min(index_set(load)));
