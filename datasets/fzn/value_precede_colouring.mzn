% nucs-target: VALUE_PRECEDE
% Graph colouring on a circulant graph, minimising the number of colours, with the value symmetry broken
% by value_precede_chain. value_precede_chain_int posts one value_precede propagator per consecutive pair
% of colours, each over the whole array -- the shape no hand-written example here produces.
include "value_precede_chain.mzn";
int: n = 50;
int: k = 8;
set of int: NODE = 1..n;
array[NODE] of var 1..k: colour;
var 1..k: used;
constraint forall(i in NODE, d in [1, 2, 3, 4, 5, 6]) (colour[i] != colour[1 + (i - 1 + d) mod n]);
constraint value_precede_chain([c | c in 1..k], colour);
constraint forall(i in NODE) (colour[i] <= used);
solve minimize used;
