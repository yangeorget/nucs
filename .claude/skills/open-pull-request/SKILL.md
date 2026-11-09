---
name: open-pull-request
description: Takes a complex change of NuCS from a branch to a pull request ready for review — the branch and its base, the checks, the push, the PR body, the CI, a code review and the hand-over. Use when a change is a complex feature or a step of a plan, when asked to open a PR, or when deciding whether a change needs a branch and a PR or only a commit.
---

# Open pull request

This skill says **how** to open a pull request. It does not give the permission to push: follow the rules of your
user about when to push and when to ask.

## 1. Decide if the change needs a PR

- **A PR:** a complex feature, a change with more than one commit, or a step of a plan (for example a step of
  `design/proposals/lcg.md`, 7.0).
- **A commit only:** a small change, such as a rename, a fix of a few lines, or a move of code in one module. Commit it
  when your user asks.

## 2. Make the branch from the right base

- A change for all of NuCS: from an up-to-date `main` (`git fetch origin`, then
  `git checkout -b <branch> origin/main`).
- A step of a feature that has its own branch (stacked PRs): from that feature branch, with a name that shows the
  step, for example `lcg-step7` from `lcg`. The PR goes into the feature branch, and the feature branch has its own
  draft PR into `main`.

Before the first push, check that the base runs CI on pull requests: `.github/workflows/test.yml` must trigger on
`pull_request` for that base. If it does not, the PR gets no CI until it reaches `main`.

## 3. Check before the push

- `check-style` passes. The pre-commit hook also runs it.
- `run-tests`: the whole suite passes with the JIT. Let CI run the suite without the JIT, except for the paths that the
  JIT hides.
- A new test fails when the change is broken: break the rule that it guards, run it, and restore the code.
- When the change can change the search (a solver, a propagator, the heuristics), compare the counters on the examples
  with those of the base. When it touches the hot path, measure the speed against the base, as
  `design/benchmarking.md` says.

## 4. Push and open the PR

```bash
git push -u origin <branch>
gh pr create --base <base> --head <branch> --title "<title>" --body-file -
```

Write the title and the body in Simplified Technical English. The body has these sections, and leaves out a section
that has nothing to say:

- **Change**: what changes, and why.
- **Behavior**: what a user sees, or "no change", and why.
- **Tests**: what was run, and the results.
- **Speed**: the ratios against the base, with the number of runs.
- **Departures from the design**: each place where the code is different from the design document, and why.

End the body with the attribution line that your environment gives for pull requests.

## 5. Wait for the CI

```bash
gh pr checks <number> --watch
```

If a check fails, read its log (`gh run view <run-id> --log-failed`), fix the cause with a new commit, and push again.

## 6. Review the code

For each step of a complex feature, when the CI passes, run `/code-review medium --comment`. Use `high` for code that
keeps a solver invariant: soundness, idempotence, or a trail that must stay in step with the state. The findings go on
the lines of the PR. Fix each confirmed finding with a new commit. Skip this step for a small PR or a PR of
documentation only.

## 7. Hand the PR over

Tell your user the number and the link of the PR, the CI result, and the findings of the review with what was done for
each.

## Rules

- Never push to `main`, and never merge a PR: the maintainer merges.
- Never force-push a branch under review. After a review comment, push new commits to the same branch: the comments
  then stay on the right lines.
- Do not push to a feature branch while its CI runs, unless your user asks: the push starts the run again.
- When a PR is merged into a feature branch, pull that branch before you make the next step from it.
