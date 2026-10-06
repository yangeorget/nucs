# Plan: the tdtsp consistency gap

> **Status:** proposed, 2026-09-15. Not started.

## The prize, stated first

`tdtsp/inst_10_1_20` is the one Challenge-2026 problem where NuCS explores a materially different search
tree from Gecode: **158,699 nodes against 7,666, a factor of 20.7**. It is also the problem where NuCS is
**3.2× faster per node** than Gecode (52,032 n/s against 16,311).

So the arithmetic is unusually attractive. Gecode's tree at NuCS's node rate would finish in about
**0.15 s against Gecode's 0.47 s** — closing this gap does not make NuCS competitive here, it makes NuCS
the fastest solver on the problem. NuCS already beats Choco and CP-SAT on it, which both fail to prove
optimality inside 30 s.

## What is already established

- **It is not search.** The model carries `solve :: seq_search([int_search(atposition, input_order,
  indomain_min, complete), int_search(tvar, smallest, indomain_min, complete)])`, and NuCS honours it.
  Both solvers branch the same way; a 20.7× tree difference under identical branching is pruning.
- **It is not a missing global.** NuCS posts `nucs_inverse` natively (2, arity 22). The full list:
  `element_l_eq` ×54 (arity ≤14), `linear_eq_c` ×38, `element_eq` ×30, `linear_leq_c` ×22, `div_c_eq` ×19,
  `member` ×17, `alldifferent` ×4 (arity 11), `element_l_eq_c` ×4, `inverse` ×2. 190 propagators,
  156 variables.
- **Element does leave holes, but modestly.** Instrumented over the full solve: 10,340,662 `element_l_eq`
  calls scanning 43,538,941 index positions, of which **6,746,114 (15.5%) are dead** — an index inside
  `i`'s interval whose `l[idx]` cannot intersect `v`, and which interval domains cannot remove. That is
  0.65 dead values per call on an average interval 4.21 wide.

## The gap has two candidates and the plan must pick one first

**Candidate A — `element`.** NuCS prunes `i` to `[first live index, last live index]`; Gecode removes the
interior dead ones. Measured hole rate 15.5%.

**Candidate B — `alldifferent`.** NuCS's is Puget bound consistency over Hall intervals. Gecode's
`distinct` can run Régin's matching, which is domain-consistent. Four of them at arity 11, on a model whose
`inverse` makes the whole thing a permutation problem — which is exactly where Régin's filtering earns its
keep. **Nothing has been measured about this one yet.**

There is a real argument that A is *weaker* than its 15.5% suggests: under `indomain_min` the search only
ever touches `i`'s bounds, and `element` already advances a bound off a dead value, so interior holes are
never branched on directly. They cost indirectly — every other constraint reasoning about `i` sees a hull
that is too wide — but that is a second-order effect, and 15.5% of a 4.21-wide interval is not obviously
worth 20×. This is a reason to suspect B and a reason not to start building anything before step 0.

### Step 0 — attribute it, before designing anything

Build a MiniZinc globals directory that weakens one global at a time, and run **Gecode** against it. Gecode
is the instrument here precisely because it is the solver with the small tree: if weakening a global blows
its tree up toward 158,699, that global is where its advantage lives.

- `-G` dir 1: redefine `fzn_all_different_int` as pairwise `!=`.
- `-G` dir 2: redefine `fzn_array_var_int_element` as its standard `i = k -> v = l[k]` decomposition.

**Predicts**: exactly one of them should move Gecode's node count by an order of magnitude. If neither
does, the gap is somewhere neither candidate covers — the linear/`div` chain, or `inverse` itself — and the
plan restarts from the attribution rather than from a fix.

**Cost**: two globals dirs, four Gecode runs, under ten minutes.

## If it is `alldifferent` (candidate B)

Régin's filtering removes *values* from domains. NuCS has interval domains and cannot represent the result,
so there is no version of this that fits the current representation. The honest options:

1. **Do nothing, and record it.** A bound-consistent `alldifferent` is a documented, deliberate consequence
   of interval domains. This would be one more measured instance of that cost, which is worth having
   written down next to the others.
2. **Hall intervals are not the ceiling of what bounds can express.** Before concluding, check whether the
   current propagator is actually achieving bound consistency or something weaker — the implementation is
   Puget's, and the cheap thing is to verify it against a brute-force bound-consistent oracle on small
   random instances, the way `bin_packing_load`'s fuzz test already works. A bug here would be worth far
   more than a redesign.

## If it is `element` (candidate A)

Interior holes cannot be stored, but they can be *used*. Ranked by cost:

1. **Consume holes at the bounds, permanently.** The propagator already advances `i.min` past a dead prefix
   each call. What it does not do is *remember* it: the same dead prefix is re-derived on every one of the
   10.3M calls. A trailed "first live / last live" pair costs two cells and makes the advance monotone —
   this is `lexleq`'s resume, applied to element. **It is a throughput win, not a pruning win**, and should
   be measured on that basis, not counted against the 20.7×.
2. **A hole mask for narrow domains.** Every index variable here has at most 14 values. An optional
   per-variable 32-bit mask, written only by propagators that can produce holes and read by `tighten_at`
   when a bound lands on one, would convert interior holes into real bound movement. This is a change to
   the domain representation — the trail, `update_domains`, and the heuristics all see it — and it should
   not be attempted on the strength of one problem. Cost it properly before starting.
3. **Nothing.** If step 0 says element is worth a factor of two rather than twenty, this is the answer.

## The measurement protocol still applies

The rules of [design/benchmarking.md](../benchmarking.md) apply. The one that matters most here: **the gate
is the node count, not the time.** A change intended to strengthen propagation must reduce
`ALG_BC_NB` on tdtsp; if the node count does not move, the change did not do what it claims, whatever the
clock says. And the node count must be checked on the other 17 Challenge problems too — stronger
propagation that costs more per call can lose everywhere else, which is the trade `alldifferent`'s Hall
intervals already represent.

## Stop rule

If step 0 attributes the gap to a global whose fix needs hole representation — which is the likely outcome
for both candidates — **stop and write it up**. The deliverable is then a measured statement of what
interval domains cost on a permutation problem, not a redesign. NuCS still wins this problem against two of
the three other solvers; the gap is against Gecode alone, and it is the only one of eighteen problems where
the tree differs at all.
