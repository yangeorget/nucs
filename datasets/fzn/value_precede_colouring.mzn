% nucs-target: VALUE_PRECEDE
% COVERAGE ONLY -- this model posts value_precede wide but cannot make it hot, and neither can any other.
% value_precede(s, t) returns entailment as soon as the earliest position that can hold s is ground to s,
% and entailment is trailed, so a search that grounds variables settles it at shallow depth and it stays
% settled for the whole subtree. Measured across five shapes -- fixed prefix, fixed suffix, reversed search
% order, all-solutions enumeration to 945,937 nodes -- the propagator is called at most 12 times in any of
% them. Keeping it live needs s excluded from a long prefix without ever being ground there, and every
% attempt to arrange that made the model infeasible: barring s early makes value_precede bar t there too,
% and the cascade runs out of values. So this stays as coverage, and there is nothing here to optimise.
%
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
