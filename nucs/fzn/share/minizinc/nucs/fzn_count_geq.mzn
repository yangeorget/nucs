% Keep count_geq native so NuCS uses its count_geq_c propagator
% instead of MiniZinc's int_eq_reif + int_lin_le decomposition.
predicate fzn_count_geq(array[int] of var int: x, var int: y, var int: c);
