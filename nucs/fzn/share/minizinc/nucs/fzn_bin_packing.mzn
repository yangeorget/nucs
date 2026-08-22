% Keep bin_packing native by reducing it to the capacitated form, which NuCS maps onto its global
% bin_packing_load propagator: the single capacity c applies to every bin the bin variables can take.
include "fzn_bin_packing_capa.mzn";

predicate fzn_bin_packing(int: c, array [int] of var int: bin, array [int] of int: w) =
  if length(bin) = 0 then
    true
  else
    let {
      set of int: bins = lb_array(bin)..ub_array(bin),
    } in
    fzn_bin_packing_capa(array1d(bins, [c | b in bins]), bin, w)
  endif;
