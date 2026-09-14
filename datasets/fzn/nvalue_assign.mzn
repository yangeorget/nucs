% nucs-target: NVALUE
% Use as few distinct values as possible over a constrained assignment: nvalue over the whole array.
include "nvalue.mzn";
int: n = 30;
array[1..n] of var 1..10: x;
var 1..10: distinct;
constraint nvalue(distinct, x);
constraint forall(i in 1..n - 1) (x[i] != x[i + 1]);
constraint forall(i in 1..n - 3) (x[i] != x[i + 3]);
constraint sum(i in 1..n) (x[i]) = 169;
solve minimize distinct;
