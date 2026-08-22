% Keep at_least native so NuCS uses its count_geq_c propagator directly. MiniZinc's decomposition goes
% through the count function, which introduces a variable for the number of occurrences and then bounds it;
% the dedicated propagator takes the bound as a parameter and needs no variable at all.
include "fzn_count_geq.mzn";

predicate fzn_at_least_int(int: n, array [int] of var int: x, int: v) = fzn_count_geq(x, v, n);
