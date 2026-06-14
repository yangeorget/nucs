% Keep subcircuit native so NuCS uses its alldifferent + subcircuit propagators
% instead of MiniZinc's order-variable decomposition.
predicate fzn_subcircuit(array[int] of var int: x);
