# Benchmarking NuCS

How to measure a change to the speed or to the pruning of NuCS, and how to know that the number is correct. Each
rule below comes from a measurement that gave a wrong number at least once.

## The tools

| script | what it does |
|---|---|
| `scripts/benchmark.py` | runs the Python examples and gives a report; `--only <name>` selects some, `--json` writes the results |
| `scripts/fzn_benchmark.py` | runs the FlatZinc models of `datasets/fzn` and tells which propagators each one uses; `--coverage` |
| `scripts/diagnose_fzn.py` | for one slow MiniZinc model: the flattening time, which globals stay native, the propagator mix |
| `scripts/warm_cache.py` | compiles all the jitted code into the Numba cache, so that no compilation occurs in a timed run |

Run them with `NUMBA_CACHE_DIR=.numba/cache`, which is also what the nix shell sets.

## Two metrics, two causes

Measure the solve time (`SOLVER_ELAPSED_TIME_MS`) and the wall time of the process separately. They change for
different reasons:

- The solve time is the search: propagation, heuristics, the trail.
- The wall time also has the start of the process. Numba loads each cached function at the start, so a new
  `@njit(cache=True)` helper that a propagator calls can add about 0.2 s of wall time and nothing to the solve time.
  Give such helpers `inline="always"`.

A difference in wall time that stays the same at a very small problem size (for example `-n 8`) is a start cost,
not a solver cost. Profile it with `python -m cProfile`: a `compile_extra` frame is a real compilation, so a cache
miss.

## The protocol for an A/B comparison

1. **Commit before the A/B.** `git checkout <branch>` takes the changes that are not committed to the other branch.
   Then the two sides run the same code and give the same numbers.
2. **Run each side in its own `git worktree`, with `PYTHONPATH=<worktree>`, and print `nucs.__file__`.** Python puts
   the directory of the script on `sys.path`, not the current directory. Without `PYTHONPATH`, the two sides import
   the installed copy, and `git bisect run` measures the same code at each step.
3. **Warm the Numba cache with one run that you discard, for each build.** Numba compiles on the first call, which
   is in the timed part of `solve()`. Cold, two builds gave 1,731 / 4,120 / 3,561 / 4,059 ms; warm, the same builds
   gave 1,739 / 2,052 / 1,735 / 2,009 ms.
4. **Clear the cache when you change a constant that another module uses** (`rm -rf .numba/cache`). Numba compares
   only the date of the file of the function. A change to `STATS_MAX` in `statistics.py` left the cached
   `bc_algorithm` with the old value.
5. **Always run a control model that the change cannot affect**, in the same command. `queens` posts only
   `alldifferent`, so a change to another propagator must not move it. `nvalue_assign` and `regular_shifts` are
   also good controls. For a change to the engine, no model is safe: use two subjects of different shapes, and
   predict how the ratio between them moves.
6. **Nothing under 5% is a result.** On the reference machine, the time changes by 1–4% with the order of the runs:
   the build that runs second reads slower, and the opposite order reverses each sign. Run both orders, and
   discard the comparison when the control moves as much as the subjects.
7. **Compare all the statistics, not only the answer.** Use each counter of all the FlatZinc and Python models,
   and the number of calls of each propagator. A change to the engine that changes the order of the propagation
   shows only there. When the trees are the same (`SOLVER_CHOICE_NB`), a time difference is a cost per node.
8. **Change the memory layout of the process at each run.** The environment and the arguments are at the top of the
   stack, so their size moves the stack of the solver, and some stack positions make the same code much slower.
   v12.4.9 solved `bibd(10,15,6,4,2)` in 67 ms or in 98 ms, with the same code, cache and data: one empty
   environment variable more or less changed it. Of 32 padding lengths from 0 to 248 bytes, 2 were slow. In the slow
   process, a few memory instructions of the jitted loop cost 5 to 8 times more. A path of a different length (a
   worktree, a cache directory, the current directory) is enough, so the two sides of an A/B have different layouts
   even when nothing else differs. Repetitions with the same environment repeat the same layout and cannot show
   this. Give each run an environment variable of random length (for example 0 to 4,088 bytes, in steps of 8), and
   use the median of at least five runs. This is the setup randomization of Mytkowicz et al., *Producing Wrong Data
   Without Doing Anything Obviously Wrong!*, ASPLOS 2009.

## Probes by duplication

To price one part of a propagator, run it two times in each call and measure the increase.

- **The node count must not change.** If the duplicated part is not idempotent, the search changes and the time
  has no meaning. Duplicating the full body of `sum_eq` changed 20,584 nodes to 375,343.
- **Duplication gives a price that is too high for a part that is made of calls**, because it also doubles the
  cost of the calls. It priced the pruning loop of `regular` at 64% of the model; the removal of the loop gave 8%.
  Inline the callee first, then measure again.

## A change to the pruning

The gate is the node count, not the time. A change that makes the propagation stronger must decrease the number of
nodes on its target model. If the node count does not change, the change does not do what it claims. Check the node
count on the other models too: stronger propagation that costs more for each call can lose on all of them.
