% nucs-target: COUNT_LEQ_C
% A long shift sequence with per-value quotas expressed as at_most/at_least rather than as a cardinality
% global. That is what FlatZinc's count_leq/count_geq lower to, each over the WHOLE array, which is the
% wide count the Python examples never build and the one a live set inside count_leq_c would need.
include "at_most.mzn";
include "at_least.mzn";
int: n = 72;
int: k = 6;
array[1..n] of var 1..k: shift;
var 0..n * n: spread;
constraint forall(s in 1..k) (at_most(14, shift, s));
constraint forall(s in 1..k) (at_least(10, shift, s));
constraint forall(i in 1..n - 1) (shift[i] != shift[i + 1]);
constraint forall(i in 1..n - 2) (shift[i] != shift[i + 2]);
constraint spread = 0;
solve satisfy;
