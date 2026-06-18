% Keep disjunctive native so NuCS uses its global disjunctive (unary resource, edge-finding) propagator
% instead of the pairwise O(n^2) reified decomposition. NuCS's propagator takes constant durations, so the
% global is kept native only when every duration is fixed; otherwise it falls back to the standard
% decomposition. (MiniZinc rewrites disjunctive with positive durations to disjunctive_strict, so this file
% covers the common case.)
predicate nucs_disjunctive(array [int] of var int: s, array [int] of int: d);

predicate fzn_disjunctive_strict(array [int] of var int: s, array [int] of var int: d) =
  if forall (i in index_set(d)) (is_fixed(d[i])) then
    nucs_disjunctive(s, [fix(d[i]) | i in index_set(d)])
  else
    forall (i in index_set(d)) (d[i] >= 0) /\
    forall (i, j in index_set(d) where i < j) (s[i] + d[i] <= s[j] \/ s[j] + d[j] <= s[i])
  endif;
