% Keep inverse native so NuCS uses its global inverse (channeling) propagator instead of MiniZinc's
% element-based decomposition.
%
% In inverse(f, invf), f maps its index set onto the index set of invf and vice versa, so f's values are
% the node labels of invf's index set (and conversely). Either index set may be non-1-based and the declared
% variable domains may be wider than the node sets. The propagator takes both numberings as offset
% parameters -- each array's offset being the base of the *other* array's index set, since that is where its
% values live -- so the two arrays are handed over as they are instead of being shifted into rebased copies,
% which used to cost an auxiliary variable and an offset propagator per node of each.
predicate nucs_inverse(
  array [int] of var int: f,
  array [int] of var int: invf,
  int: f_value_offset,
  int: invf_value_offset,
);

predicate fzn_inverse(array [int] of var int: f, array [int] of var int: invf) =
  if length(f) = 0 then
    true
  else
    forall (i in index_set(f)) (f[i] in index_set(invf)) /\
    forall (j in index_set(invf)) (invf[j] in index_set(f)) /\
    nucs_inverse(f, invf, min(index_set(invf)), min(index_set(f)))
  endif;
