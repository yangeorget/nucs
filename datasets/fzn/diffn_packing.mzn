% nucs-target: DIFFN
% Strip packing: place squares in a strip of fixed width and minimise the height. diffn runs over every
% pair, with the two cumulative projections alongside it -- what Choco's diffN posts.
include "diffn.mzn";
include "cumulative.mzn";
int: n = 14;
array[1..n] of int: s = [7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1];
int: w = 12;
int: hmax = 18;
array[1..n] of var 0..w: x;
array[1..n] of var 0..hmax: y;
var 1..hmax: height;
constraint forall(i in 1..n) (x[i] + s[i] <= w /\ y[i] + s[i] <= height);
constraint diffn(x, y, s, s);
constraint cumulative(x, s, s, hmax);
solve minimize height;
