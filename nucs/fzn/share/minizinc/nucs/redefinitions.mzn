% NuCS-specific redefinitions added to the standard library (this shadows the empty std
% redefinitions.mzn; the versioned redefinitions-*.mzn files are unaffected).
%
% MiniZinc's strictly_decreasing has no non-optional int overload, so a `var int` array routes through the
% optional path (fzn_strictly_increasing_int_opt), which decomposes into a pairwise int_lin_le chain instead
% of reaching NuCS's native strictly_increasing propagator. Supplying the more specific non-optional overload
% routes it straight to fzn_strictly_increasing_int(reverse(x)) -- one propagator instead of n-1, matching
% how decreasing already maps to fzn_increasing_int.
%
% redefinitions.mzn is auto-included for every model, so it must declare fzn_strictly_increasing_int itself
% (it is not pulled in unless the model includes globals.mzn) -- otherwise every compilation fails with
% "no function or predicate with name 'fzn_strictly_increasing_int' found".
include "fzn_strictly_increasing_int.mzn";

% NuCS has no set variables (its domains are integer intervals). Including the standard nosets.mzn rewrites
% every `var set` into an array of `var bool` plus int/bool constraints NuCS already supports, so set-based
% models flatten instead of failing with "domain 'set' is not supported". This is what the other non-set
% solvers (cp-sat, chuffed, the linear/MIP backend) do. Caveat: a set comprehension over a large value
% universe can blow up into a huge bool encoding -- such models compile but may be impractical for a
% single-threaded bounds solver.
include "nosets.mzn";

predicate strictly_decreasing(array [$X] of var int: x) = fzn_strictly_increasing_int(reverse(array1d(x)));
