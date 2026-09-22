---
name: bump-version
description: Raises the version in pyproject.toml under NuCS's own semantic-versioning rule and rolls the changelog's Unreleased section into it. Use when cutting a release, when asked to bump the version, or when deciding whether the pending changes are a major, a minor or a patch.
---

# Bump version

`pyproject.toml` holds the only version string: `docs/source/conf.py` reads it out of there at build time, so the
docs never need a bump of their own.

## 1. Decide the level

NuCS's rule is stricter than semver.org's and is stated in the `CHANGELOG.md` preamble:

> a major bump means the extension points documented in [the docs](https://nucs.readthedocs.io/) changed shape.

The `## Unreleased` section is the evidence — it is written as the changes land, so read it rather than diffing
against the last tag. Take the highest level that applies to any one entry:

- **major** — a documented extension point changed shape or went away: the signature of a consistency algorithm, of a
  heuristic or of `Problem`'s public methods; the arguments a propagator module must export. *Documented* means it
  appears under `docs/source/`; a propagator's internals do not. 16.0.0 was a major because a consistency algorithm
  and `update_propagators` both lost their `propagator_nb` parameter and `Problem.split` was removed.
- **minor** — something new that leaves every existing program working: a new propagator and its `ALG_*`, a new
  FlatZinc builtin, a new heuristic or solver option, a new optional parameter.
- **patch** — nothing new to call and nothing to relearn: bug fixes, stronger pruning, speed, dependency pins.

Two calls that come up often:

- Pruning more, or reporting entailment where it did not, is a **patch**: it changes how long a solve takes, not
  what it answers. A change to *which* solutions come back is a bug fix, so also a patch.
- A new `ALG_*` is a **minor** even when it only replaces a decomposition that MiniZinc used to post, because the
  constant is now part of the public API.

When two levels are arguable, take the higher one: an unneeded major costs a version number, a missed one lets a
user's program break silently. Say which level and which single entry forced it before editing anything.

## 2. Edit

- `pyproject.toml`: `version = "X.Y.Z"`, resetting the lower fields — 16.3.1 becomes 17.0.0 for a major, 16.4.0 for a
  minor.
- `CHANGELOG.md`: rename `## Unreleased` to `## X.Y.Z` and open a fresh, empty `## Unreleased` above it. The
  headings carry no date; the tag records when.
- `README.md`: the line pointing at the changelog lists the releases that carry a migration note. Add this one only
  if its section has a `### Migrating from ...` heading.
- Confirm nothing else pinned the old number: `grep -rn '<old version>' --exclude-dir={.git,build,dist,docs/output} .`

## 3. Leave the diff

Stop there, unstaged, and report the level with its reason. Committing, tagging and releasing are a separate ask,
and `publish-release` handles them: `publish.yml` fires on GitHub **release creation**, not on the tag, and pushes
to PyPI and Docker Hub, so that step is not undoable.
