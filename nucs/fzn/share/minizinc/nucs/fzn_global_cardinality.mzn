% global_cardinality(x, cover, counts) with fixed counts -- the usual case, as in elitserien's
% global_cardinality_closed(break, [0, 9, 11, ...], [2, 2, ...]) -- is global_cardinality_low_up with
% lbound = ubound = counts, which NuCS keeps native; the standard decomposition posts one count per value, with no
% reasoning across values. global_cardinality_closed reaches this predicate too, after restricting x to the cover.
% Variable counts keep the standard decomposition.
%
% The parameter names match the standard library's declaration (x, cover, counts).
include "count.mzn";
include "fzn_global_cardinality_low_up.mzn";

predicate fzn_global_cardinality(
  array [int] of var int: x,
  array [int] of int: cover,
  array [int] of var int: counts,
) =
  if is_fixed(counts) then
    fzn_global_cardinality_low_up(x, cover, fix(counts), fix(counts))
  else
    forall (i in index_set(cover)) (count(x, cover[i], counts[i])) /\
      % Implied constraint
      length(x) >= sum(counts)
  endif;
