---
name: publish-release
description: Ships a NuCS release — the release commit, the vX.Y.Z tag on that commit, and the GitHub release whose notes are the version's changelog section. Use when cutting a release, tagging a version, or publishing to PyPI and Docker Hub.
---

# Publish release

Start from a tree that `bump-version` has already prepared: `pyproject.toml` and `CHANGELOG.md` carry the new
version and the diff is unstaged.

`publish.yml` fires on GitHub **release creation**, not on a tag push. Creating the release publishes to PyPI and
Docker Hub, and PyPI refuses a second upload of a version, so that one step cannot be taken back. Everything before
it can be. Do steps 1–3 when asked to release; stop before step 4 and ask.

## 1. Commit

```bash
git add pyproject.toml CHANGELOG.md README.md
git commit   # subject "release X.Y.Z"; body argues the level, as bump-version decided it
```

Keep unrelated work out of it: the tag names this commit, and a revert of the release should not take anything else
with it.

## 2. Tag the release commit, and push the tag *before* the release exists

The order is the whole point of this step. `v16.0.0` points at `6f1bd93`, **111 commits past** its release commit
`07bc5ac`, because the release was created from the web UI with the target left as `main`: GitHub tags the branch
tip at the moment of the click, not the commit that set the version. PyPI's 16.0.0 therefore shipped 111 commits
that the changelog credits to 16.1.0. A tag that already exists cannot be minted at the wrong place.

```bash
git push origin main
git tag vX.Y.Z <release commit>   # lightweight, as every NuCS tag is; name the commit, don't rely on HEAD
git push origin vX.Y.Z
```

Check both before going on — the tag is still free to move at this point, and will not be after step 4:

```bash
git ls-remote --tags origin | grep vX.Y.Z      # points at the release commit
git show vX.Y.Z:pyproject.toml | sed -n '3p'   # reads version = "X.Y.Z"
```

## 3. Take the notes from the changelog

The release notes are the version's section, verbatim — never a fresh summary, which would say something subtly
different from what the repository says:

```bash
{ awk -v v="X.Y.Z" '$0 == "## " v {f=1; next} f && /^## / {exit} f' CHANGELOG.md
  echo '**Full Changelog**: https://github.com/yangeorget/nucs/compare/vPREV...vX.Y.Z'
} > <scratch>/notes.md
```

Read it back: it must start at the first `###` subsection and stop before the previous version's heading. The
compare link is the line GitHub generates for itself, and `--notes-file` replaces the body whole, so put it back or
it is lost.

## 4. Create the release — the step that publishes

Give the notes at creation, whichever way it is done. A release published with an empty body tends to keep one, and
on 16.1.0 — cut only so that a published version's notes would match its contents — it did.

- **Here**, with `gh` (installed 2026-09-22; `gh auth status` should show `yangeorget`, and `! gh auth login` from
  the user restores it):

  ```bash
  gh release create vX.Y.Z --title "X.Y.Z" --notes-file <scratch>/notes.md --verify-tag
  ```

  `--verify-tag` makes it fail rather than invent a tag — the command-line form of the same footgun.

- **User, in the UI**: *Draft a new release* → **choose the existing tag `vX.Y.Z` from the dropdown** → title
  `X.Y.Z` → paste `notes.md`. The *target* field can be left on `main`: once the tag exists, GitHub attaches the
  release to it and does not move it, which is what step 2 buys.

Ask for a go-ahead before running it, even mid-release: this is where PyPI and Docker Hub become involved.

A body that did go out empty is repairable, and this is the one part of a release that stays mutable —
`publish.yml` listens for `release: [created]`, not `edited`, so nothing re-runs and PyPI is never asked to take
the version twice:

```bash
gh release edit vX.Y.Z --notes-file <scratch>/notes.md
```

## 5. Verify what went out

```bash
gh run list --workflow=publish.yml --limit 3          # or the Actions tab
curl -s https://pypi.org/pypi/nucs/json | python -c "import json,sys; print(sorted(json.load(sys.stdin)['releases'])[-3:])"
```

Both jobs must pass: PyPI, and the Docker image pushed as `minizinc-nucs:X.Y.Z` and `:latest`. If the release was
cut from the wrong commit, the tag is already public — say so plainly rather than moving it, and fix it forward with
the next version.
