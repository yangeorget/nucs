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
% bool_clause_reif is reified as the same linear inequality NuCS already uses for the non-reified
% bool_clause -- b <-> (sum(as) - sum(bs) >= 1 - |bs|) -- which flattens to a single int_lin_ge_reif that
% NuCS handles natively (one auxiliary sum + a reified comparison), instead of the standard library's
% expansion into one clause plus a per-literal implication each (a ~5-6x blow-up in bool_clause count).
%
% The standard file's array_float_maximum / array_float_minimum bodies are deliberately not restated here.
% They are unreachable for NuCS: any model that could trigger them declares var float, which the FlatZinc
% parser rejects outright (NuCS handles integer variables only), so no float decomposition can ever help.

predicate bool_clause_reif(array [int] of var bool: as, array [int] of var bool: bs, var bool: b) =
  b <-> (sum (i in index_set(as)) (bool2int(as[i])) - sum (i in index_set(bs)) (bool2int(bs[i]))
    >= 1 - length(bs));

predicate array_int_maximum(var int: m, array [int] of var int: x);

predicate array_int_minimum(var int: m, array [int] of var int: x);
