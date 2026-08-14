% Keep regular native so NuCS uses its DFA (layered-graph) propagator instead of MiniZinc's decomposition
% into a state variable per position plus element/linear transition constraints. The standard fzn_regular
% takes the transition table as a 2D array [state, symbol], which a FlatZinc builtin cannot; so it is flattened
% row-major into nucs_regular, the 1D-argument predicate NuCS handles.
predicate nucs_regular(
  array[int] of var int: x,
  int: Q,
  int: S,
  array[int] of int: d,
  int: q0,
  set of int: F,
);

predicate fzn_regular(
  array[int] of var int: x,
  int: Q,
  int: S,
  array[int, int] of int: d,
  int: q0,
  set of int: F,
) = nucs_regular(x, Q, S, array1d(d), q0, F);
