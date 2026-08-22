% Keep lex_lesseq native so NuCS uses its lexicographic propagator.
%
% The context promises are the standard library's: in a monotone context x may only be weakened and
% y strengthened, which is what lets MiniZinc half-reify a use of this constraint.
predicate fzn_lex_lesseq_int(
  array [int] of var int: x :: promise_ctx_antitone,
  array [int] of var int: y :: promise_ctx_monotone,
);
