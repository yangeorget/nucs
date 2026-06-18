% Keep diffn native so NuCS uses its global diffn (non-overlapping rectangles) propagator instead of the
% pairwise 4-way reified decomposition. NuCS's propagator takes constant rectangle sizes, so the global is
% kept native only when every dx and dy is fixed; otherwise it falls back to the standard decomposition.
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
    nucs_diffn(x, y, [fix(dx[i]) | i in index_set(dx)], [fix(dy[i]) | i in index_set(dy)])
  else
    forall (i, j in index_set(x) where i < j) (
      x[i] + dx[i] <= x[j] \/ y[i] + dy[i] <= y[j] \/ x[j] + dx[j] <= x[i] \/ y[j] + dy[j] <= y[i]
    )
  endif;
