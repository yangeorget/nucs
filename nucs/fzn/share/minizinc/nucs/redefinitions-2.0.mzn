% NuCS overrides of the standard redefinitions-2.0.mzn.
%
% A solver-provided redefinitions-2.0.mzn shadows the standard one *by filename*, so this file must
% restate every entry NuCS still relies on -- it is not merged with the standard version.
%
% The only deviation from the standard file is that array_int_maximum / array_int_minimum are declared
% body-less (native): NuCS has dedicated MAX_EQ / MIN_EQ propagators (see array_int_maximum / array_int_minimum
% in builtins.py), so keeping the n-ary global avoids MiniZinc's default decomposition into a chain of
% binary int_max / int_min constraints with one introduced variable per element.
%
% bool_clause_reif and the float maxima/minima keep their standard decomposition bodies: NuCS has no native
% propagator for them, so they must still flatten through the standard library.

predicate bool_clause_reif(array [int] of var bool: as, array [int] of var bool: bs, var bool: b) =
  clause(as, bs ++ [b]) /\
    forall (i in index_set(as)) (as[i] -> b) /\
    forall (i in index_set(bs)) (bs[i] \/ b);

predicate array_int_maximum(var int: m, array [int] of var int: x);

predicate array_int_minimum(var int: m, array [int] of var int: x);

predicate array_float_maximum(var float: m, array [int] of var float: x) =
  let {
    int: l = min(index_set(x));
    int: u = max(index_set(x));
    float: ly = lb_array(x);
    float: uy = ub_array(x);
    array [l..u] of var ly..uy: y;
  } in y[l] = x[l] /\ m = y[u] /\ forall (i in l + 1..u) (y[i] == max(x[i], y[i - 1]));

predicate array_float_minimum(var float: m, array [int] of var float: x) =
  let {
    int: l = min(index_set(x));
    int: u = max(index_set(x));
    float: ly = lb_array(x);
    float: uy = ub_array(x);
    array [l..u] of var ly..uy: y;
  } in y[l] = x[l] /\ m = y[u] /\ forall (i in l + 1..u) (y[i] == min(x[i], y[i - 1]));
