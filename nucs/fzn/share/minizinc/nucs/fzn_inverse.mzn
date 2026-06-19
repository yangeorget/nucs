% Keep inverse native so NuCS uses its global inverse (channeling) propagator instead of MiniZinc's
% element-based decomposition.
%
% In inverse(f, invf), f maps its index set onto the index set of invf and vice versa, so f's values are
% the node labels of invf's index set (and conversely). Either index set may be non-1-based and the declared
% variable domains may be wider than the node sets. Rebase both arrays to a contiguous 1..n numbering -- f's
% values by invf's base, invf's values by f's base -- which also tightens the domains, before handing them
% to the native predicate.
predicate nucs_inverse(array [int] of var int: f, array [int] of var int: invf);

predicate fzn_inverse(array [int] of var int: f, array [int] of var int: invf) =
  if length(f) = 0 then
    true
  else
    let {
      int: lf = min(index_set(f)),
      int: lg = min(index_set(invf)),
      int: n = length(f),
      array [1..n] of var 1..n: f2,
      array [1..n] of var 1..n: invf2,
    } in
    forall (i in index_set(f)) (f2[i - lf + 1] = f[i] - lg + 1) /\
    forall (j in index_set(invf)) (invf2[j - lg + 1] = invf[j] - lf + 1) /\
    nucs_inverse(f2, invf2)
  endif;
