% Keep lex_less native so NuCS reduces it to its lexleq propagator over sentinel-extended vectors
% instead of MiniZinc's reified bool-chain decomposition.
%
% The context promises are the standard library's: in a monotone context x may only be weakened and
% y strengthened, which is what lets MiniZinc half-reify a use of this constraint.
predicate fzn_lex_less_int(
  array [int] of var int: x :: promise_ctx_antitone,
  array [int] of var int: y :: promise_ctx_monotone,
);
