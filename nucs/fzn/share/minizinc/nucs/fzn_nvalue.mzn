% Keep nvalue native so NuCS uses its nvalue propagator
% instead of MiniZinc's int_eq_reif + linear-sum decomposition.
predicate fzn_nvalue(var int: n, array[int] of var int: x);
