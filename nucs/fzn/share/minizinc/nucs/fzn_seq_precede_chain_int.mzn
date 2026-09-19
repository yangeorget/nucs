% seq_precede_chain(X): each value v >= 2 in X first occurs after v - 1 does, values below 1 being free. That is
% value_precede_chain over 1..ub(X), which NuCS keeps native as consecutive value_precede propagators, instead of
% the standard decomposition's running maximum H and its 2n element-wise constraints.
%
% The parameter name matches the standard library's declaration (X): MiniZinc is rolling out named arguments for
% native predicates, so a divergent name is a future error rather than a cosmetic difference.
include "fzn_value_precede_chain_int.mzn";

predicate fzn_seq_precede_chain_int(array [int] of var int: X) =
  if length(X) = 0 then
    true
  elseif ub_array(X) < 2 then
    true  % no value has a predecessor to wait for
  else
    fzn_value_precede_chain_int([v | v in 1..ub_array(X)], X)
  endif;
