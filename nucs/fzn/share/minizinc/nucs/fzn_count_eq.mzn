% Keep count_eq native so NuCS uses its count_eq / count_eq_c propagator
% instead of MiniZinc's int_eq_reif + int_lin_eq decomposition.
predicate fzn_count_eq(array[int] of var int: x, var int: y, var int: c);
