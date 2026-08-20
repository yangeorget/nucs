% Keep diffn native so NuCS uses its global diffn (non-overlapping rectangles) propagator instead of the
% pairwise 4-way reified decomposition. NuCS's propagator takes constant rectangle sizes, so the global is
% kept native only when every dx and dy is fixed; otherwise it falls back to the standard decomposition.
include "fzn_cumulative.mzn";

predicate nucs_diffn(
  array [int] of var int: x,
  array [int] of var int: y,
  array [int] of int: dx,
  array [int] of int: dy,
);

predicate fzn_diffn(
  array [int] of var int: x,
  array [int] of var int: y,
  array [int] of var int: dx,
  array [int] of var int: dy,
) =
  if forall (i in index_set(dx)) (is_fixed(dx[i])) /\ forall (i in index_set(dy)) (is_fixed(dy[i])) then
    let {
      array [int] of int: fdx = [fix(dx[i]) | i in index_set(dx)],
      array [int] of int: fdy = [fix(dy[i]) | i in index_set(dy)],
      % Bounding-box spans: every non-overlapping placement fits inside them, so projecting the rectangles
      % on each axis gives two implied cumulative constraints (heights on one axis, capacity on the other).
      % They add the energetic reasoning that the pairwise diffn cannot derive.
      int: spanx = max (i in index_set(x)) (ub(x[i]) + fdx[i]) - min (i in index_set(x)) (lb(x[i])),
      int: spany = max (i in index_set(y)) (ub(y[i]) + fdy[i]) - min (i in index_set(y)) (lb(y[i])),
    } in
      nucs_diffn(x, y, fdx, fdy)
      /\ nucs_cumulative(x, fdx, fdy, spany)
      /\ nucs_cumulative(y, fdy, fdx, spanx)
  else
    forall (i, j in index_set(x) where i < j) (
      x[i] + dx[i] <= x[j] \/ y[i] + dy[i] <= y[j] \/ x[j] + dx[j] <= x[i] \/ y[j] + dy[j] <= y[i]
    )
  endif;
