# Design documents

This directory has the documents for the contributors of NuCS. The user documentation is in `docs/`, which Read the
Docs builds. `ARCHITECTURE.md`, at the root, gives the shape of the code and is the place to start.

## The current code

These documents describe `main`. When the code changes, they change with it.

| document | subject |
|---|---|
| [choice-points.md](choice-points.md) | how the solver saves and restores the search state: the trail, the write barrier, the stack of choice points |
| [benchmarking.md](benchmarking.md) | how to measure a change to the speed or to the pruning, and how to know that the number is correct |

## Proposals

A proposal records a decision. It starts with a status line: *proposed*, *implemented*, *closed* (a part is
implemented and the rest is rejected) or *rejected*. When the status is not *proposed*, the document is frozen: what
stays true about the code goes into `ARCHITECTURE.md` or into a document of the section above.

| document | status | subject |
|---|---|---|
| [proposals/lcg.md](proposals/lcg.md) | proposed, 2026-10 | lazy clause generation: explanations, conflict analysis, learned clauses |
| [proposals/tdtsp-element-gap.md](proposals/tdtsp-element-gap.md) | proposed, 2026-09 | why Gecode explores 20 times fewer nodes than NuCS on tdtsp: `element` or `alldifferent` |
| [proposals/engine-per-call-overhead.md](proposals/engine-per-call-overhead.md) | closed, 2026-09 | the cost of the engine around each propagator call: dispatch, statistics, write-back |
| [proposals/incremental-filtering.md](proposals/incremental-filtering.md) | closed, 2026-09 | how Choco, Gecode and OR-Tools keep propagator state, and the state block of NuCS |
| [proposals/sequential-search.md](proposals/sequential-search.md) | implemented, 2026-06 | a list of searches, for the sequential search of MiniZinc |

Working drafts go in `specs/`, which git ignores.
