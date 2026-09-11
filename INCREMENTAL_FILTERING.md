# Incremental filtering: how Choco, Gecode and OR-Tools do it, and what NuCS should do

## 0. Executive summary

Three solvers, three answers to the same question — *how does a propagator keep a computed invariant
alive across a backtracking search?*

| | Choco | Gecode | OR-Tools (`constraint_solver`) |
|---|---|---|---|
| Mechanism | **trailing** (`IEnvironment`) | **copying + recomputation** (`Space::clone`) | **trailing** (`ReversibleEngine`) |
| Unit of state | `IStateInt`, `IStateBool`, `IStateBitSet`, `IStateIntVector`, … | any propagator member field | `Rev<T>`, `NumericalRev<T>`, `RevArray<T>`, `RevIntSet`, `RevBitSet`, … |
| Write barrier | per-object world stamp | none needed | per-object search stamp |
| Delta information | `propagate(idxVarInProp, mask)` + `IIntDeltaMonitor` | `advise(home, a, Delta&)` via `Council<A>` | demons (`WhenBound`/`WhenRange`/`WhenDomain`) |
| Cost model | one trail entry per cell per node | one space copy per `c_d` nodes | one trail entry per cell per node |

Two findings matter more than the taxonomy:

1. **Backtrackable state is necessary but not sufficient.** All three solvers need a *second* mechanism —
   fine-grained modification events — before an O(n) rescan becomes an O(#changed) update. Backtrackable
   state alone tells you the invariant is still valid; it does not tell you what to add to it.
2. **The cheapest and most profitable form of incrementality needs no deltas at all.** All three
   independently implement the same trick: *remove ground variables from the constraint's own scan,
   permanently for the subtree*. Grounding is monotone along a branch, so the update is idempotent and
   needs no old value. This is where I would start in NuCS.

---

## 1. Choco — a general-purpose trailing environment

### 1.1 The environment is a first-class subsystem

Choco factors backtrackable memory out of the solver entirely, into `org.chocosolver.memory`. `IEnvironment`
is a factory for backtrackable primitives:

```java
IStateBool        makeBool(boolean initialValue);
IStateInt         makeInt();            IStateInt  makeInt(int initialValue);
IStateLong        makeLong(long init);  IStateDouble makeFloat(double initialValue);
IStateBitSet      makeBitSet(int size); IStateBitSet makeSparseBitset(int blocksize);
IStateIntVector   makeIntVector(int size, int initialValue);
IStateDoubleVector makeDoubleVector(int size, double initialValue);
void save(IOperation operation);        // arbitrary undo closure
void saveAt(IOperation operation, int worldIndex);
```

plus the search-tree API: `worldPush()` ("starts a new branch in the search tree"), `worldPop()`
("backtracks to the previous choice point"), `worldPopUntil(int w)`, `worldCommit()`, `getWorldIndex()`,
and `getTimeStamp()` — "the current time stamp. It differs from world index since it never decrements".

The default implementation is `EnvironmentTrailing`. Nothing in it knows about constraints: a propagator
that wants an incremental counter just asks the model's environment for an `IStateInt`.

### 1.2 The write barrier: one trail entry per object per world

`StoredInt.set()` is the whole idea in six lines:

```java
if (y != currentValue) {
    final int wi = environment.getWorldIndex();
    if (this.timeStamp < wi) {
        myTrail.savePreviousState(this, currentValue, timeStamp);
        timeStamp = wi;
    }
    currentValue = y;
}
```

Each stored object carries the index of the world in which it was last written. If that is older than the
current world, the value must be saved before overwriting; otherwise the previous value in this world is
already on the trail and the write is free. Repeated narrowing inside one fixpoint therefore costs one
trail entry, not one per write.

### 1.3 The trail

`StoredIntTrail` (the "flatten" variant; there is also a chunked one) is three parallel arrays plus a
per-world index:

```java
private StoredInt[] variableStack;   // which object
private int[]       valueStack;      // its former value
private int[]       stampStack;      // the world that value belonged to
private int[]       worldStartLevels;// trail position at each worldPush
```

`worldPop` walks back to `worldStartLevels[worldIndex]` restoring **both the value and the stamp**:

```java
final StoredInt v = variableStack[currentLevel];
v._set(valueStack[currentLevel], stampStack[currentLevel]);
```

Restoring the stamp is what keeps the barrier correct after a pop — without it, an object would look
"already saved in this world" when it is not. There is one trail per primitive type (int, bool, long,
double, plus an operation trail), so an entry carries no discriminator. `worldCommit` merges a world into
its parent and compacts the trail in place, dropping entries made redundant by the merge.

### 1.4 Fine-grained propagation is the other half

Choco propagators have two entry points:

- **coarse**: `propagate(int evtmask)` — must reach the propagator's own fixpoint from the current domains;
- **fine**: `propagate(int idxVarInProp, int mask)` — "an incremental filtering algorithm called whenever
  the variable of index `idxVarInProp` has changed".

Opting into fine propagation is a constructor flag (`reactToFineEvent = true`). For value-level (AC)
incrementality, `IIntDeltaMonitor` additionally replays the values removed from a domain since the last
call.

### 1.5 Exhibit A — the same constraint, twice

Choco ships **both** a stateless and an incremental version of the boolean sum and picks between them when
the constraint is posted:

`PropSumBool.prepare()` rescans everything:

```java
sumLB = sumUB = 0;
for (; i < pos; i++) { lb = vars[i].getLB(); ub = vars[i].getUB(); sumLB += lb; sumUB += ub; ... }
```

`PropSumBoolIncr` keeps the two sums as `IStateInt` — the class javadoc says so outright, *"Sum of lower
bounds maintained incrementally. Main reason this version exists."*:

```java
private final IStateInt bLB, bUB;      // == environment.makeInt()

public void propagate(int idxVarInProp, int mask) {          // fine: O(1)
    if (idxVarInProp < pos) {
        if (vars[idxVarInProp].getLB() == 1) { bLB.add(1);  doFilter |= o != Operator.GE; }
        else                                 { bUB.add(-1); doFilter |= o != Operator.LE; }
    } ...
    if (doFilter) forcePropagate(PropagatorEventType.CUSTOM_PROPAGATION);
}

protected void prepare() { sumLB = bLB.get() - sum.getUB(); sumUB = bUB.get() - sum.getLB(); }
```

and keeps a **resync path** for the case where the state cannot be trusted:

```java
public void propagate(int evtmask) {
    if (PropagatorEventType.isFullPropagation(evtmask)) { /* rebuild bLB, bUB by a full scan */ }
    doFilter = false; filter();
}
```

Note what this exhibit does *not* say: Choco's general integer `PropSum` (arbitrary coefficients) is **not**
incremental — it recomputes `sumLB`/`sumUB` from scratch on every call. Incrementality is reserved for the
boolean case, where the delta is known to be ±1 without reading an old bound. That is a deliberate
engineering judgement, not an oversight.

### 1.6 Exhibit B — state that is a data structure, not a number

`PropNoSubtour` (the no-sub-cycle constraint) keeps the path decomposition itself backtrackable:

```java
private final IStateInt[] origin, end, size;
...
public void propagate(int idxVarInProp, int mask) {
    varInstantiated(idxVarInProp, vars[idxVarInProp].getValue() - offset);
}
// inside varInstantiated:
size[start].add(size[val].get());
origin[last].set(start);
end[start].set(last);
```

Each instantiation merges two chains in O(1). Nothing is ever rebuilt; the environment undoes the merges on
backtrack. This is the pattern with the largest payoff, because the *stateless* version of this constraint
is quadratic.

---

## 2. Gecode — state is free because the search copies it

### 2.1 No trail at all

Gecode has no backtrackable-value machinery. Search is **copying with recomputation**: a `Space` is cloned,
and every actor in it implements

```cpp
virtual Actor* copy(Space& home) = 0;   // Gecode::Actor
```

A propagator's copy constructor copies its member fields. Therefore *any* member field of *any* propagator
is automatically backtrackable, with no barrier, no stamp and no declaration.

The price is paid by the search engine, and it is tuned by two numbers documented in `gecode/search.hh`:

> - `c_d` as minimal recomputation distance: this guarantees that a path between two nodes in the search
>   tree for which copies are stored has at least length `c_d`. […] it stores `c_d` times less nodes than
>   full copying.
> - `a_d` as adaptive recomputation distance: when a node needs to be recomputed and the path is longer
>   than `a_d`, an intermediate copy is created (approximately in the middle of the path) to speed up
>   future recomputation.
>
> Full copying corresponds to a maximal recomputation distance `c_d` of 1. All recomputation performed is
> based on batch recomputation: batch recomputation performs propagation only once for an entire path used
> in recomputation.

Defaults are `c_d = 8`, `a_d = 2`. So Gecode does not so much solve the state-persistence problem as
*dissolve* it: state is persisted by the same mechanism that persists domains, and the memory cost is
amortised by replaying propagation instead of storing every node.

### 2.2 Exhibit — permanent elimination of assigned views

`Gecode::Int::Linear::Lin<Val,P,N,PC>` holds

```cpp
ViewArray<P> x;   ViewArray<N> y;   Val c;
```

and permanently folds ground views into `c`, shrinking the arrays:

```cpp
for (int i=x.size(); i--; )
  if (x[i].assigned()) { c -= x[i].val();  x.move_lst(i); }
for (int i=y.size(); i--; )
  if (y[i].assigned()) { c += y[i].val();  y.move_lst(i); }
```

These mutations persist across `propagate()` calls and down the subtree. A long linear constraint therefore
becomes *cheaper the deeper the search goes*, and is restored to its full width automatically when the
search backtracks past the clone. No delta information is needed: only the fact that a view is assigned,
which is monotone along a branch.

### 2.3 Advisors — the O(#changed) escape hatch

When copying is not enough and the propagator must react to *what* changed, Gecode uses **advisors**
subscribed through a `Council<A>`. An advisor's `advise` runs at the moment a view is modified and receives
the `Delta`:

```cpp
virtual ExecStatus advise(Space& home, Advisor& a, const Delta& d);
```

with return values `ES_FIX` (the propagator does not even need to run), `ES_NOFIX`, `ES_NOFIX_DISPOSE`,
`ES_NOFIX_FORCE_DISPOSE`. The manual notes the delta "describes how the variable has been changed by an
operation on the advisor's variable".

`ReLinBoolInt` in `gecode/int/linear/bool-int.hpp` is the canonical use: two counters, `n_s` (number of
still-subscribed views) and `c` (running count of ones), maintained entirely inside `advise`:

```cpp
if (VX::one(d)) c--;
n_s--;
if ((n_s < c) || (c <= 0)) return ES_NOFIX;
```

and the copy constructor carries the counters over verbatim:

```cpp
ReLinBoolInt<VX,VB>::ReLinBoolInt(Space& home, ReLinBoolInt<VX,VB>& p)
  : Propagator(home,p), n_s(p.n_s), c(p.c) {
  p.normalize();  co.update(home,p.co);  x.update(home,p.x);  b.update(home,p.b);
}
```

The propagator body never rescans the array.

---

## 3. OR-Tools `constraint_solver` — a trail with typed stacks and object ownership

### 3.1 `ReversibleEngine`

The reversibility layer (recently split out of `Solver` into `reversible_engine.h`) exposes:

```cpp
template <class T> void SaveValue(T* o);                 // unconditional; call before modifying
template <class T> void SaveAndSetValue(T* adr, T val);  // guarded by  *adr != val
template <class T> void SaveAndAdd(T* adr, T val);       // guarded by  val != 0
template <typename T> T* RevAlloc(T* object);            // ownership -> deleted on backtrack
template <typename T> T* RevAllocArray(T* object);
uint64_t stamp() const;                                  // "how many moves in the search tree"
void BacktrackTo(StateMarker* m);
```

Internally `struct Trail` keeps one stack per primitive type (`int`, `int64_t`, `uint64_t`, `double`,
`bool`, `void*`) — the same "no discriminator" idea as Choco. `RevAlloc`/`RevAllocArray` have no analogue
in the other two solvers: they make *allocation itself* reversible, so a propagator can build a temporary
structure at a node and let the engine destroy it on backtrack.

### 3.2 `Rev<T>` — the same stamped barrier as Choco

```cpp
template <class T> class Rev {
  void SetValue(ReversibleEngine* engine, const T& val) {
    if (val != value_) {
      if (stamp_ < engine->stamp()) { engine->SaveValue(&value_); stamp_ = engine->stamp(); }
      value_ = val;
    }
  }
  uint64_t stamp_;  T value_;
};

template <class T> class NumericalRev : public Rev<T> {   // Add / Incr / Decr
  void Add(ReversibleEngine* engine, const T& to_add) { this->SetValue(engine, this->Value() + to_add); }
};
```

`RevArray<T>` is the vectorised form, and its comment states the invariant explicitly:

> It contains the stamp optimization. I.e., the SaveValue call is done only once per node of the search
> tree. Please note that actual stamp always starts at 1, thus an initial value of 0 always triggers the
> first SaveValue.

On top of these primitives sits a much larger library of reversible containers than either competitor
offers: `RevIntSet` (sparse set with reversible size — the classic swap-to-the-end structure),
`RevBitSet`/`RevBitMatrix`, `RevGrowingArray`, `RevPartialSequence`, `RevSwitch`, `SimpleRevFIFO`,
`RevImmutableMultiMap`.

### 3.3 Demons

Propagation is demon-based: a constraint attaches `Demon`s to variables (`WhenBound`, `WhenRange`,
`WhenDomain`), each demon carrying a priority and its own `stamp_` used to avoid re-enqueueing a demon
already scheduled at the current node. The demon is the unit of *fine* propagation, playing the role of
Choco's `propagate(idxVarInProp, mask)`.

### 3.4 Exhibit — `pack.cc`

The bin-packing constraint is built almost entirely out of reversible counters:

```cpp
RevArray<int>     first_unbound_backward_vector_;
RevArray<int64_t> sum_of_bound_variables_vector_;
RevArray<int64_t> sum_of_all_variables_vector_;
Rev<int64_t>      sum_of_assigned_items_;
Rev<int>          assigned_count_, unassigned_count_;
Rev<int>          card_min_;
```

`first_unbound_backward_` is OR-Tools' version of Gecode's `move_lst`: a reversible cursor that
monotonically shrinks the range each dimension has to scan as items get assigned.

### 3.5 Footnote: CP-SAT's newer take

`ortools/util/rev.h` (used by CP-SAT rather than the classic CP solver) generalises the idea to a
*level-indexed* interface rather than a mark-based one:

```cpp
class ReversibleInterface { virtual void SetLevel(int level) = 0; };
template <class T> class RevRepository : public ReversibleInterface {
  void SaveState(T* object);
  void SaveStateWithStamp(T* object, int64_t* stamp);   // same one-save-per-level trick
};
```

Same barrier, different plumbing: the search declares its level and every registered reversible object
syncs itself, instead of the objects pushing onto a shared trail.

---

## 4. What the three have in common

1. **Two ways to persist, not three.** Trail it (Choco, OR-Tools) or copy it (Gecode). NuCS already chose
   trailing, and the `trailing` branch note in `ARCHITECTURE.md` records why: 2–80× less memory for ~8%
   throughput. That decision also decides this one.
2. **The barrier is always "one save per cell per node."** Choco compares a per-object *world index*,
   OR-Tools a per-object *search stamp*, CP-SAT a per-object *level stamp*. NuCS's `trail_set` already
   implements the same rule *positionally* (`mark <= trail_indices[cell] < trail_size`), and the docstring
   in `state.py` argues — correctly — that stating it positionally removes the class of bug where a site
   forgets to bump a counter. **NuCS's memory layer is already at parity with all three.**
3. **State without deltas buys less than you think.** Choco's fine `propagate(idx, mask)`, Gecode's
   `advise(…, Delta&)` and OR-Tools' demons exist because knowing the invariant is *valid* is useless
   unless you also know *what to add to it*. This is the piece NuCS does not have.
4. **Except for the one shape that needs no deltas.** Gecode `move_lst` + absorb into `c`; OR-Tools
   `first_unbound_backward_`; Choco `PropNoSubtour` chain merging. All three are "fold ground variables in,
   shrink the active set", all three are monotone along a branch, and all three are idempotent under
   re-entry. That last property matters a lot for NuCS, where non-idempotent propagators are re-scheduled
   into their own fixpoint.
5. **Incrementality is opt-in, per constraint, and measured.** Choco literally ships `PropSumBool` and
   `PropSumBoolIncr` side by side and does not bother making the general integer sum incremental.

---

## 5. Proposal for NuCS

### 5.1 The good news: three of the four pieces already exist

| piece | Choco | OR-Tools | NuCS today |
|---|---|---|---|
| a flat backtrackable store | typed trails | typed trails | `state` (`int32`), one flat array |
| an undo log | `variableStack/valueStack/stampStack` | `struct Trail` | `trail_log`, `trail_undo` |
| a one-save-per-node barrier | world stamp | `Rev<T>::stamp_` | `trail_set` + `trail_indices` |
| a place for propagator state | `IStateInt` | `Rev<T>` | **missing** |

So the proposal is not "build a persistence mechanism". It is "**give propagators an address in the
mechanism that already exists**".

### 5.2 The layout change

Extend `state` with a fourth region, keeping the unbound count last so `unbound_index()` stays
`len(state) - 1`:

```
 0            2n          2n+P              2n+P+S     2n+P+S+1
 [ domains  | entailed  | propagator state | unbound ]
```

Address it CSR-style, exactly like `propagator_variables` and `propagator_parameters`: a third column
`OFFSETS_STATE` in `offsets`, so propagator `p` owns
`state[offsets[p, OFFSETS_STATE] : offsets[p+1, OFFSETS_STATE]]`. Propagators that want nothing get a
zero-width block and pay nothing — no branch, no cache line, no trail entry.

Registration gains one function alongside `get_triggers_*` / `get_complexity_*`:

```python
def get_state_linear_leq_c(n: int, parameters: NDArray) -> tuple[int, int]:
    """Returns (trailed_nb, hint_nb): the number of backtrackable and of untrailed int32 cells."""
    return 2, 0
```

defaulting to `(0, 0)` so nothing else has to change.

### 5.3 The signature change, and why to pay it once

```python
SIGN_COMPUTE_DOMAINS = int64(int32[:, ::1], int32[::1], int32[::1])   # domains, parameters, prop_state
```

`prop_state` is a **direct view into `state`**, typed C-contiguous. `ARCHITECTURE.md` already records the
measured reason for `::1`: the same experiment run with a `int32[:]` (any-layout) third argument came out
5% *slower* than baseline despite allocating nothing.

This is the API break that shelved both the scratch-buffer experiment (~4%) and the warm-permutation
experiment (~8% on queens, ~5.5% on langford). `ARCHITECTURE.md` says the break should only be paid once
and names the condition — *"revisit if the signature changes anyway for another reason"*. **This is that
reason.** Land all three at once: the hint suffix of `prop_state` *is* the scratch buffer, and warm
permutations *are* a hint block. One break, three payoffs.

### 5.4 Two tiers, two contracts

**Tier A — hints (untrailed).** Contract: *the filtering result must be identical whatever the block
contains.* Staleness costs time, never correctness. Nothing is trailed, nothing is restored, and the
non-JIT path is untouched. This tier is already validated: the warm-permutation experiment verified it
across 73,712 solutions of queens 13 and millions of backtracks, with a bit-identical propagation checksum.

**Tier B — semantic state (trailed).** Contract: *the cells are a function of the current domains,
maintained incrementally.* The engine trails the block; the propagator must keep it in sync.

Both live in one contiguous block — trailed prefix, hint suffix — so there is still exactly one extra
argument.

### 5.5 The barrier: trail on entry

In `bc_algorithm`, immediately before the `compute_domains_fcts[algorithm](...)` call, trail the block's
trailed prefix with the existing `trail_set`, which no-ops when this choice point already holds a live entry
for the cell:

```
for cell in prefix:  trail_size = trail_set(state, trail_log, trail_indices, mark, trail_size, cell,
                                            state[cell], state[cell])
```

Cost: `trailed_nb` L1 loads per call (zero for the ~45 propagators with no state), and `trailed_nb` trail
pushes *at most once per propagator per choice point*. For `linear_*` with `trailed_nb == 2` this is noise.

This placement buys a property worth stating explicitly in the docs: **a propagator that returns
`PROP_INCONSISTENCY` halfway through may already have written its state block, and that is safe** — the
block was trailed on entry, so `trail_undo` restores it. The "propagators cannot corrupt global state"
invariant survives, in a slightly weaker form.

*Alternative, if a block ever gets wide:* copy-in / diff-out through a `state_buffer`, mirroring exactly
what `domain_buffer` + `update_domains` already do for domains — pay a diff instead of a barrier scan. I
would not start there; the blocks that matter are 1–3 cells wide.

### 5.6 The delta problem — be honest about it

NuCS's engine tells a propagator *that* it was triggered, never *what changed*. Three ways out, in
increasing order of cost:

**(1) Don't need deltas — depend only on groundness.** State that is a function of *which variables are
ground and to what value*: Gecode's eliminated views, OR-Tools' `first_unbound_backward_`, Choco's
`PropNoSubtour` chains. The propagator scans, folds ground variables into its state, and removes them from
its own active set. Because grounding is monotone along a branch, the update needs no old value and — the
key property for NuCS — **is idempotent under re-entry**, since a variable removed from the active set
cannot be absorbed twice. That matters because NuCS's non-idempotent propagators are re-scheduled into
their own fixpoint; a naive `sum += delta` would double-count on the second call. *This category covers the
highest-value candidates and is where I would spend the first effort.*

**(2) Self-recorded deltas.** The block caches the bounds the propagator last saw, so it computes its own
deltas on entry. This is O(n) per call — no asymptotic win over a rescan for `linear_*`, but it turns an
O(n log n) or O(n²) *rebuild* into an O(n) *update*, which is the whole game for `alldifferent`, `scc`,
`no_sub_cycle` and `cumulative`.

**(3) Engine-recorded deltas.** `update_domains` already holds `(variable, old_min, old_max, new_min,
new_max)` at the moment of the write and is already walking the trigger list; it could append into a
per-propagator ring buffer. This is Choco's fine-propagation model, and it moves cost onto the *writer*,
proportional to the number of watchers. Given the measured ~1193 propagator-calls per node (against
Gecode's 63), the writer side is already the hot side. **Recommend against as a first step**; revisit only
if (1) and (2) land and a specific constraint still demands it.

### 5.7 Ranked candidates

| propagator | today | with state | tier | delta need |
|---|---|---|---|---|
| `no_sub_cycle`, `subcircuit` | O(n²) rebuild + a fresh `np.zeros((n,3))` **per call** | trailed chain arrays, O(1) per grounding — literally Choco's `PropNoSubtour` | B | (1) none |
| `linear_*`, `sum_*` | O(n) rescan, every call, every node | active-prefix compaction + absorbed constant; the constraint gets cheaper with depth — Gecode's `Lin` | B | (1) none |
| `alldifferent`, `gcc` | O(n log n) with an identity-seeded argsort (O(n²) cliff when keys decorrelate from index) | warm permutations — **already measured at ~8% end-to-end on queens, 49× per call at n=2048** | A | none |
| `scc` | O(n²) | component/chain state | B | (1) none |
| `cumulative`, `disjunctive` | O(n³) | ground-task elimination, cached profile | B | (1), then (2) |
| `count_*`, `bin_packing_load` | O(n) / O(item² · bin) | reversible counters — Choco's `PropSumBoolIncr`, OR-Tools' `pack.cc` | B | (1) |
| `regular`, `relation`, `element_*` | table/automaton scans | untrailed lookup caches | A | none |

`no_sub_cycle` is the single best first exhibit: it is quadratic, it allocates on every call, its incremental
form is textbook and independently implemented by Choco, and it needs no delta plumbing at all.

### 5.8 Staging

- **Stage 0 — decide before breaking the API.** Instrument `no_sub_cycle` and `linear_leq_c` to measure how
  much of each call is redundant with the previous call at the same node. Cheap, no API change, and it puts
  a number on the ceiling before spending the break.
- **Stage 1 — the break, once.** Add `prop_state` to `SIGN_COMPUTE_DOMAINS` and the 52 propagators
  (mechanical), add `get_state_*` with a `(0, 0)` default, add `OFFSETS_STATE`, add the entry barrier. Land
  the shelved scratch-buffer and warm-permutation work in the same commit — the ~4% and ~8% are already
  measured and are pure profit once the argument exists.
- **Stage 2 — two exhibits.** `no_sub_cycle` (Tier B, data-structure state) and `linear_leq_c` (Tier B,
  range compaction). One of each shape, so the pattern is documented by example.
- **Stage 3 — roll out by measured win**, one propagator at a time, keeping the stateless implementation
  as the differential oracle exactly as Choco keeps `PropSumBool` next to `PropSumBoolIncr`.

### 5.9 Invariants to protect

- **Idempotence.** A Tier-B update must be idempotent with respect to re-entry inside one fixpoint. This is
  the single most likely source of a silent wrong-answer bug, and it is why "eliminate from the active set"
  is the right shape and "`sum += delta`" is not.
- **Entailment.** An entailed propagator stops being called, so its state goes stale; on backtrack both the
  entailment flag and the state block are restored from the same trail, so this stays consistent — but it
  deserves an explicit sentence in `ARCHITECTURE.md`.
- **`OPTIM_RESET`.** `choice_point_init` runs at solve time on every reset and clears `trail_top` and
  `trail_indices`; Tier-B blocks must be re-seeded there, or seeded to a value valid at the root.
- **Trail sizing.** `trail_headroom` is currently `len(state) + STEP_TIGHTENING_NB * TIGHTENING_TRAIL_ENTRY_NB`,
  justified by "a choice point can trail each cell of state at most once". That argument still holds with
  the fourth region — `len(state)` simply grows by `S` — so the bound stays correct by construction. Worth
  re-checking the *measured* 2–12× band on the benchmark set after the change.
- **`NUMBA_DISABLE_JIT=1` parity.** `prop_state` is a plain NumPy slice; no typed lists, no fallback needed.
- **Multiprocessing.** State blocks live inside `state`, which is already per-process.
- **Testing.** Every Tier-B propagator needs (a) a differential test against its stateless version on random
  domains, and (b) a full-search test asserting identical solution counts *and* identical
  `SOLVER_CHOICE_NB` / `PROPAGATOR_FILTER_NB`, so that a state bug shows up as a search difference rather
  than as a rare wrong answer.

### 5.10 What I would *not* copy

- **Gecode's copying.** NuCS already made the opposite call, with numbers.
- **A `save(IOperation)`-style arbitrary undo closure** (Choco) or `RevAlloc` (OR-Tools). Both require heap
  objects and virtual dispatch on the backtrack path — the exact thing NuCS's flat-array design exists to
  avoid. Fixed-width int32 blocks cover every candidate in §5.7.
- **A full delta/advisor subsystem.** It is the largest piece of machinery in both Choco and Gecode, it puts
  cost on the writer side which is already NuCS's hot side, and §5.7 shows the good candidates do not need
  it.

---

## Sources

- Choco: [`IEnvironment`](https://github.com/chocoteam/choco-solver/blob/master/solver/src/main/java/org/chocosolver/memory/IEnvironment.java),
  [`StoredInt`](https://github.com/chocoteam/choco-solver/blob/master/solver/src/main/java/org/chocosolver/memory/trailing/StoredInt.java),
  [`StoredIntTrail`](https://github.com/chocoteam/choco-solver/blob/master/solver/src/main/java/org/chocosolver/memory/trailing/trail/flatten/StoredIntTrail.java),
  [`PropSum`](https://github.com/chocoteam/choco-solver/blob/master/solver/src/main/java/org/chocosolver/solver/constraints/nary/sum/PropSum.java),
  [`PropSumBoolIncr`](https://github.com/chocoteam/choco-solver/blob/master/solver/src/main/java/org/chocosolver/solver/constraints/nary/sum/PropSumBoolIncr.java),
  [`PropNoSubtour`](https://github.com/chocoteam/choco-solver/blob/master/solver/src/main/java/org/chocosolver/solver/constraints/nary/circuit/PropNoSubtour.java),
  [`Propagator` javadoc](https://javadoc.io/static/org.choco-solver/choco-solver/4.0.2/org/chocosolver/solver/constraints/Propagator.html)
- Gecode: [`kernel/core.hpp`](https://github.com/Gecode/gecode/blob/master/gecode/kernel/core.hpp),
  [`search.hh`](https://github.com/Gecode/gecode/blob/master/gecode/search.hh),
  [`int/linear/int-nary.hpp`](https://github.com/Gecode/gecode/blob/master/gecode/int/linear/int-nary.hpp),
  [`int/linear/bool-int.hpp`](https://github.com/Gecode/gecode/blob/master/gecode/int/linear/bool-int.hpp)
- OR-Tools: [`reversible_engine.h`](https://github.com/google/or-tools/blob/main/ortools/constraint_solver/reversible_engine.h),
  [`reversible_data.h`](https://github.com/google/or-tools/blob/main/ortools/constraint_solver/reversible_data.h),
  [`constraint_solver.h`](https://github.com/google/or-tools/blob/main/ortools/constraint_solver/constraint_solver.h),
  [`pack.cc`](https://github.com/google/or-tools/blob/main/ortools/constraint_solver/pack.cc),
  [`util/rev.h`](https://github.com/google/or-tools/blob/main/ortools/util/rev.h)
