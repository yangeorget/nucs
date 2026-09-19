% Keep cumulative native so NuCS uses its global cumulative (timetabling + energetic reasoning) propagator
% instead of the time-indexed decomposition. NuCS's propagator takes constant durations, demands and capacity,
% so the global is kept native only when all of them are fixed; otherwise it falls back to the standard
% decompositions -- time-indexed over a short horizon and task-indexed beyond 5000 time units, as the standard
% library chooses. Without that guard, test-scheduling (0/1 demands over a long horizon) did not flatten in
% 600 s, where Gecode and Choco take 0.3 s.
predicate nucs_cumulative(
  array [int] of var int: s,
  array [int] of int: d,
  array [int] of int: r,
  int: b,
);

% Variable durations, but fixed demands and capacity: NuCS's variable-duration cumulative propagator
% (compulsory parts use each task's minimum duration).
predicate nucs_cumulative_var(
  array [int] of var int: s,
  array [int] of var int: d,
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
  elseif forall (i in index_set(r)) (is_fixed(r[i])) /\ is_fixed(b) then
    nucs_cumulative_var(
      s,
      d,
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
      let {
        int: early = min([lb(s[i]) | i in tasks]),
        int: late = max([ub(s[i]) + ub(d[i]) | i in tasks]),
      } in
      if late - early > 5000 then
        % one resource check per task start: a task j's start meets the tasks running then
        forall (j in tasks) (
          b >= sum (i in tasks) (r[i] * bool2int(s[i] <= s[j] /\ s[j] < s[i] + d[i]))
        )
      else
        forall (t in early..late - 1) (
          b >= sum (i in tasks) (r[i] * bool2int(s[i] <= t /\ t < s[i] + d[i]))
        )
      endif
    endif
  endif;
