% NuCS-specific redefinitions added to the standard library (this shadows the empty std
% redefinitions.mzn; the versioned redefinitions-*.mzn files are unaffected).
%
% MiniZinc's strictly_decreasing has no non-optional int overload, so a `var int` array routes through the
% optional path (fzn_strictly_increasing_int_opt), which decomposes into a pairwise int_lin_le chain instead
% of reaching NuCS's native strictly_increasing propagator. Supplying the more specific non-optional overload
% routes it straight to fzn_strictly_increasing_int(reverse(x)) -- one propagator instead of n-1, matching
% how decreasing already maps to fzn_increasing_int.
predicate strictly_decreasing(array [$X] of var int: x) = fzn_strictly_increasing_int(reverse(array1d(x)));
