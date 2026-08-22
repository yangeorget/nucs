% Keep exactly native so NuCS uses its count_eq_c propagator directly, for the same reason as at_least:
% no introduced occurrence-count variable.
include "fzn_count_eq.mzn";

predicate fzn_exactly_int(int: n, array [int] of var int: x, int: v) = fzn_count_eq(x, v, n);
