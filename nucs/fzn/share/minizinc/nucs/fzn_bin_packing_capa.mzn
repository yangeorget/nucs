% Keep bin_packing_capa native so NuCS uses its global bin_packing_load propagator. MiniZinc's
% decomposition posts, for every (item, bin) pair, a reified equality feeding a per-bin linear -- O(n * m)
% constraints and reification variables. Introducing one load variable per bin instead turns the whole
% constraint into a single propagator: the capacity is then just the load variable's upper bound, which
% MiniZinc folds into its declared domain.
include "fzn_bin_packing_load.mzn";

predicate fzn_bin_packing_capa(
  array [int] of int: c,
  array [int] of var int: bin,
  array [int] of int: w,
) =
  if length(bin) = 0 then
    true
  else
    let {
      array [index_set(c)] of var 0..max(c): load,
    } in
    forall (b in index_set(c)) (load[b] <= c[b]) /\
    forall (i in index_set(bin)) (bin[i] >= min(index_set(c)) /\ bin[i] <= max(index_set(c))) /\
    nucs_bin_packing_load(load, bin, w, min(index_set(c)))
  endif;
