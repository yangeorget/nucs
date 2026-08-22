% Keep the non-strict disjunctive native so NuCS uses its global disjunctive propagator instead of the
% pairwise O(n^2) reified decomposition. It differs from disjunctive_strict in that a zero-duration task may
% be scheduled anywhere, even inside another task -- so such a task is simply dropped from the resource,
% which leaves exactly the strict problem over the remaining tasks. As for disjunctive_strict, this needs
% fixed durations; otherwise it falls back to the standard decomposition.
include "fzn_disjunctive_strict.mzn";

predicate fzn_disjunctive(array [int] of var int: s, array [int] of var int: d) =
  if forall (i in index_set(d)) (is_fixed(d[i])) then
    forall (i in index_set(d)) (d[i] >= 0) /\
    let {
      set of int: busy = {i | i in index_set(d) where fix(d[i]) > 0},
    } in
    if card(busy) > 1 then
      nucs_disjunctive([s[i] | i in busy], [fix(d[i]) | i in busy])
    else
      true
    endif
  else
    forall (i in index_set(d)) (d[i] >= 0) /\
    forall (i, j in index_set(d) where i < j) (
      d[i] = 0 \/ d[j] = 0 \/ s[i] + d[i] <= s[j] \/ s[j] + d[j] <= s[i]
    )
  endif;
