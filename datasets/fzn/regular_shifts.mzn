% nucs-target: REGULAR
% A rostering pattern automaton over a long horizon: at most three consecutive work days and a rest after
% a night, with the number of night shifts maximised. regular is posted once over the whole sequence.
include "regular.mzn";
int: n = 24;
array[1..n] of var 1..3: d;   % 1 = day, 2 = night, 3 = rest
int: S = 6;
array[1..S, 1..3] of 0..S: T =
  array2d(1..S, 1..3, [2, 5, 1,
                       3, 5, 1,
                       4, 5, 1,
                       0, 0, 1,
                       0, 0, 6,
                       2, 5, 1]);
var 0..n: nights;
constraint regular(d, S, 3, T, 1, 1..S);
constraint nights = sum(i in 1..n) (bool2int(d[i] = 2));
constraint sum(i in 1..n) (bool2int(d[i] = 3)) >= 6;
solve maximize nights;
