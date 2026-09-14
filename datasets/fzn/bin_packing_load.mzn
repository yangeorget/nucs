% nucs-target: BIN_PACKING_LOAD
% Partition weighted items into bins and minimise the heaviest load. bin_packing_load runs over the item
% assignments and the load variables together, so its arity is items + bins and its cost is quadratic in
% that -- which is why this model is sized by item count rather than by search depth: at 54 items the
% propagator is most of the runtime, which is what makes a change to it measurable here.
include "bin_packing_load.mzn";
int: n = 54;
int: m = 7;
array[1..n] of int: w = [23, 19, 17, 13, 13, 11, 11, 9, 9, 8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3,
                         3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 23, 19, 17, 13, 13, 11, 11, 9, 9, 8, 8, 7, 7, 7,
                         6, 6, 5, 5, 5, 4];
int: cap = sum(i in 1..n) (w[i]);
array[1..n] of var 1..m: bin;
array[1..m] of var 0..cap: load;
var 0..cap: heaviest;
constraint bin_packing_load(load, bin, w);
constraint forall(b in 1..m) (load[b] <= heaviest);
constraint bin[1] = 1;  % break part of the bin symmetry
solve minimize heaviest;
