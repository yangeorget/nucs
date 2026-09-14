% nucs-target: COUNT_LEQ_C
% nucs-solve: all
% A long shift roster with a per-shift quota, expressed as at_most/at_least rather than as a cardinality
% global. That is what FlatZinc's count_leq/count_geq lower to, each over the WHOLE horizon, so this is
% the wide count no Python example builds -- and the one a live set inside count_leq_c would live or die on.
%
% The horizon is long but most of it is pre-assigned, the way a real roster carries fixed allocations. That
% is deliberate: it sets the propagators' arity independently of how much search there is, so the model can
% be wide (which is what makes it representative) without the enumeration exploding. Widening n alone
% leaves the node count untouched; only `free` moves it.
%
% Worth stating plainly, because it cuts both ways: pre-assigning most of the horizon also means most of
% each count's variables are ground, which is exactly the property an incremental count propagator would
% exploit. That is a real roster's shape rather than a contrivance -- fixed allocations are why
% on-call-rostering models carry them -- but a change measured here should be checked against `free` set
% high enough that little is pre-assigned before it is believed in general.
include "at_most.mzn";
include "at_least.mzn";
int: n = 120;
int: k = 5;
int: quota = n div k;
int: free = 20;
array[1..n] of var 1..k: shift;
constraint forall(s in 1..k) (at_most(quota, shift, s));
constraint forall(s in 1..k) (at_least(quota, shift, s));
constraint forall(i in 1..n - 1) (shift[i] != shift[i + 1]);
constraint forall(i in 1..n - 2) (shift[i] != shift[i + 2]);
constraint forall(i in 1..n - 3) (shift[i] != shift[i + 3]);
constraint forall(i in 1..n - free) (shift[i] = 1 + (i - 1) mod k);
solve satisfy;
