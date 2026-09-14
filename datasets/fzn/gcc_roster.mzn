% nucs-target: GCC
% A shift roster over a long horizon: each shift is used a fixed number of times and no shift repeats
% within three days. global_cardinality_low_up posts count_geq_c and count_leq_c over the whole horizon,
% which is the wide count the Python examples never build.
include "global_cardinality_low_up.mzn";
int: n = 27;
int: k = 5;
array[1..n] of var 1..k: shift;
var 0..n * k: weighted;
constraint global_cardinality_low_up(shift, [s | s in 1..k], [4 | s in 1..k], [8 | s in 1..k]);
constraint forall(i in 1..n - 1) (shift[i] != shift[i + 1]);
constraint forall(i in 1..n - 2) (shift[i] != shift[i + 2]);
constraint weighted = sum(i in 1..n) (bool2int(shift[i] = 1) * i);
solve minimize weighted;
