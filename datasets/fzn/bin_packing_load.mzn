% nucs-target: BIN_PACKING_LOAD
% Partition weighted items into bins, minimising the heaviest load: the classic bin_packing_load shape,
% with the load variables as the objective's handle.
include "bin_packing_load.mzn";
int: n = 34;
int: m = 8;
array[1..n] of int: w = [23, 19, 17, 13, 13, 11, 11, 9, 9, 8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1];
array[1..n] of var 1..m: bin;
array[1..m] of var 0..50: load;
var 0..50: heaviest;
constraint bin_packing_load(load, bin, w);
constraint forall(b in 1..m) (load[b] <= heaviest);
constraint bin[1] = 1;  % break part of the bin symmetry
solve minimize heaviest;
