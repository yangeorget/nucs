% Keep cumulative native so NuCS uses its global cumulative (timetabling + energetic reasoning) propagator
% instead of the time-indexed decomposition. NuCS's propagator takes constant durations, demands and capacity,
% so the global is kept native only when all of them are fixed; otherwise it falls back to the standard
% time-resource decomposition.
predicate nucs_cumulative(
  array [int] of var int: s,
  array [int] of int: d,
  array [int] of int: r,
  int: b,
);

predicate fzn_cumulative(
  array [int] of var int: s,
  array [int] of var int: d,
  array [int] of var int: r,
  var int: b,
) =
  if forall (i in index_set(d)) (is_fixed(d[i]))
    /\ forall (i in index_set(r)) (is_fixed(r[i]))
    /\ is_fixed(b) then
    nucs_cumulative(
      s,
      [fix(d[i]) | i in index_set(d)],
      [fix(r[i]) | i in index_set(r)],
      fix(b),
    )
  else
    let {
      set of int: tasks = {i | i in index_set(s) where ub(r[i]) > 0 /\ ub(d[i]) > 0},
    } in
    if card(tasks) == 0 then
      true
    else
      forall (t in min([lb(s[i]) | i in tasks])..max([ub(s[i]) + ub(d[i]) | i in tasks]) - 1) (
        b >= sum (i in tasks) (r[i] * bool2int(s[i] <= t /\ t < s[i] + d[i]))
      )
    endif
  endif;
