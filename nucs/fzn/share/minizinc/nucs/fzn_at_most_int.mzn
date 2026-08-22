% Keep at_most native so NuCS uses its count_leq_c propagator directly, for the same reason as at_least:
% no introduced occurrence-count variable.
include "fzn_count_leq.mzn";

predicate fzn_at_most_int(int: n, array [int] of var int: x, int: v) = fzn_count_leq(x, v, n);
