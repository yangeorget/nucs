# Developing a constraint solver with an AI agent — plan of the article

> **Status:** plan, 2026-10-06. Not written.

## The idea

NuCS is a constraint solver in Python and Numba. Since mid-2026, most of its development is done with Claude Code,
an AI agent that reads the code, runs the tests, measures and commits. This article tells what this changed: what
became faster and better, what went wrong, and which methods made the agent useful.

The main message: **the agent is fast at making claims and slow to doubt them.** The gains are real, but they come
only with guardrails: measurements with a protocol, tests with oracles, checks before actions that cannot be undone,
and a human who decides. An article that shows only the gains is not credible; an article that shows the gains *and*
the guardrails that the speed made necessary is.

**Readers:** developers who use or think about AI agents, and the constraint programming community. The reader knows
software development, but not necessarily constraint solvers. Explain each solver term once, briefly.

**Length:** about 3,000 to 4,000 words, with 4 or 5 charts.

## Outline

### 1. Introduction

- What NuCS is, in three sentences: a solver that runs Python code compiled by Numba, with propagators, search
  heuristics and a FlatZinc adapter for MiniZinc.
- Why a solver is a good test of an agent: the code is small but subtle, a bug can give a wrong answer that looks
  correct, and speed is measurable on a fixed set of problems.
- The period: from the first commit with the agent (May 2026) to 18.0.0 (October 2026).

### 2. The way of working

- **Skills:** instructions that the agent loads when it needs them (`.claude/skills/`): style checks, tests,
  docstrings, the addition of a propagator, the version number, the release.
- **Architecture documents:** `ARCHITECTURE.md` as the entry point; `design/` for the current mechanisms and for the
  proposals, each with a status line.
- **Memory:** notes that the agent keeps between sessions: results of benchmarks, traps, decisions.
- **Simplified Technical English** (ASD-STE100) for all the text that the agent writes: answers, commits, changelog,
  docs. What it changed: more uniform, easier for a non-native reader, but sometimes longer.
- **Where each piece of knowledge goes**, and why: the repository for what is true about the code, the memory for the
  history and the traps, the skills for the procedures.

### 3. Speed of development

- The number of commits per month before and after the agent (chart 1).
- The new features in the period: dom/wdeg, restarts and last-conflict, native globals (`bin_packing_load`,
  `cumulative`, `regular`, `circuit_chains`, `disjunctive` with Theta-trees), full half-reification, sequential
  search, the per-propagator state block.
- Breaking changes became cheaper: 18.0.0 removed ten parameters, with migration examples in the changelog.
- What the agent does not make faster: the decisions, the reviews, the waits for CI.

### 4. Quality of the code

- Documentation: lines of docs over time, the docstrings, the changelog with a migration section for each major.
- Commit messages: before and after (examples side by side).
- Tests: number of tests and coverage over time (chart 2). The tests run with and without the JIT, on two Python
  versions.
- Style: ruff and mypy at each commit, now enforced by a git pre-commit hook.

### 5. Correctness: the agent writes the oracles, not only the tests

- The brute-force probe of all 60 propagators against a solver that enumerates all the values: it found a wrong
  answer in `count_eq` and hangs in `element_eq` and `gcc`.
- Differential testing against Gecode: it found a class of bugs with index sets that do not start at 1 in the
  FlatZinc adapter, which made NuCS answer "unsatisfiable" for problems that have a solution.
- The `diffn` soundness bug in optimization, and how it made idempotence an invariant of the whole solver.
- The point: exhaustive testing became cheap, so it is now done for each propagator, not only for the difficult ones.

### 6. Speed of the solver

- A fixed set of problems: the Python examples (`scripts/benchmark.py`) and the FlatZinc models (`datasets/fzn`).
- The solve time of each release tag on this set (chart 3).
- The comparison with other solvers on the 16 medium problems of the MiniZinc Challenge 2026: at parity with Choco
  and Gecode, behind the learning solvers (Chuffed, CP-SAT). This gap is why lazy clause generation is next.
- Which gains came from the engine (no-change reporting: 3× on magic_sequence) and which came from search
  (last-conflict: 23× to 44× on magic_sequence).

### 7. The agent makes hypotheses, the measurement filters them

- The list of negative results: array merges (no gain), delta recording (0.75× to 0.80×), half-reification (about
  no gain), incremental filtering stage 2 (rejected), the per-call overhead of the engine (the stop rule fired).
- The protocol that came from wrong measurements (`design/benchmarking.md`): commit before an A/B, one worktree for
  each side, a warm cache, a control model, nothing under 5%, compare all the statistics.
- A stop rule set before the measurement, so that a result cannot be explained after the fact.

### 8. Where the agent was wrong, and what found the error

- `ARCHITECTURE.md` said that no propagator used trailed state; the agent repeated it and designed with it. Six
  propagators use it. Only a read of the code found it: three sources agreed, but they were one source.
- Measurements that were noise: run-order drift, a stale Numba cache, an A/B that measured the installed copy on
  both sides.
- A hook that fixed the code between two partial edits, and deleted an import that the next edit needed.
- The lesson: check the claims of the agent against the code and against measurements, not against other text.

### 9. Guardrails for actions that cannot be undone

- The release footgun: the tag of 16.0.0 was put 111 commits after its release commit. The lesson is now a skill: the
  tag first, then the release with `--verify-tag`.
- The release of 18.0.0: published only after green CI on the release code, with a go-ahead for each push.
- Permissions: the agent asks before a push; a classifier blocks some commands.
- Checks, not fixes, at the moment of a commit.

### 10. The role of the human

- The human decides: for example, no thread for the FlatZinc output, after a gain of 3× to 6× was already there.
- The agent does work that was not asked for. Sometimes it helps (a broken download URL in CI found before the
  push), sometimes it is scope creep. Where is the line?
- The periodic full code review: why it is necessary, what it finds that the review of each change does not.
- Review is the new limit: the agent writes more than a human can read with care.

### 11. Conclusion

- What I would do again, and what I would do differently.
- What is next: lazy clause generation, designed with the agent (`design/proposals/lcg.md`).

## Data to collect

| data | how | for |
|---|---|---|
| commits per month, with and without the agent | `git log` and the `Co-Authored-By: Claude` trailer | chart 1 |
| tests and coverage over time | the CI runs, or `pytest --collect-only` and `coverage` on one commit per month, in worktrees | chart 2 |
| lines of documentation over time | `docs/`, `ARCHITECTURE.md`, `design/`, `CHANGELOG.md`, one commit per month | section 4 |
| solve time per release tag | `scripts/benchmark.py` on each tag, with the protocol of `design/benchmarking.md` | chart 3 |
| nodes and time against other solvers | the MiniZinc Challenge 2026 results in `mzn-challenge/2026/` | chart 4 |
| bugs found by probes and differential tests | the commits of the probe sweep and the Gecode tests | section 5 |
| examples of commit messages | one commit before and one after the agent, for the same kind of change | section 4 |

**A limit of the trailer data:** the commits of May and June 2026 have almost no trailer, but the agent was already
in use. The trailer started in July. Thus it is a lower bound of the work of the agent, not a measure of it. Say this
in the article.

**First numbers** (2026-10-06): 1,240 commits since 2024-03-28; 262 with the trailer. By month (all / with the
trailer): 2026-05 71/1, 2026-06 92/1, 2026-07 38/18, 2026-08 89/81, 2026-09 152/141.

## Questions to decide before writing

1. One article, or two: one on the methods (sections 2, 7, 8, 9, 10), one on the results (sections 3 to 6)?
2. Name the agent and the product (Claude Code), or keep it general?
3. Show the exact prompts and skills, or only describe them?
4. Where to publish: the same place as the earlier articles?
5. How long was the period before the agent that the comparison uses? The history starts in March 2024, but
   2025 has few commits.
