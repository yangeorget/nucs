% nucs-target: COUNT_EQ_C
% nucs-solve: all
% count_shifts with exact quotas: each shift gets exactly `quota` days, stated as count_eq over the WHOLE horizon
% with a constant value and a constant count. That is what NuCS posts as count_eq_c, at arity n, which no Python
% example builds: sports_tournament_scheduling posts count_eq_c at arity 4 to 6 only.
%
% The rest of the model is count_shifts': the same horizon, the same spacing constraints, and most of the horizon
% pre-assigned, so the arity is set independently of how much search there is. See count_shifts.mzn for why the
% pre-assignment cuts both ways: a change measured here should be checked with `free` set high enough that little
% is pre-assigned before it is believed in general.
%
% Measured with it (2026-11-03): the change report of count_eq_c, 1.087x on 2.8M calls at 83% no-change.
include "count_eq.mzn";
int: n = 120;
int: k = 5;
int: quota = n div k;
int: free = 20;
array[1..n] of var 1..k: shift;
constraint forall(s in 1..k) (count_eq(shift, s, quota));
constraint forall(i in 1..n - 1) (shift[i] != shift[i + 1]);
constraint forall(i in 1..n - 2) (shift[i] != shift[i + 2]);
constraint forall(i in 1..n - 3) (shift[i] != shift[i + 3]);
constraint forall(i in 1..n - free) (shift[i] = 1 + (i - 1) mod k);
solve satisfy;
