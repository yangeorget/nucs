---
name: check-style
description: Runs the NuCS style checks — ruff lint with autofix, ruff format and mypy — over nucs/, tests/ and scripts/. Use after editing any Python file in those directories, before committing, or when asked to lint, format or type-check.
---

# Check style

Run the whole check with:

```bash
./scripts/bash/style.sh
```

It runs these in order, over `nucs`, `tests` and `scripts`, and stops at the first that fails:

```bash
ruff check --fix nucs tests scripts  # lint, auto-fixing what it can
ruff format nucs tests scripts       # format
mypy nucs tests scripts              # type-check
```

- The first two rewrite files: review the diff they leave.
- Fix by hand what ruff cannot fix and every mypy error, then rerun until the script passes. It must pass before
  committing.
- A hook (`.claude/hooks/ruff-edited-file.sh`) already runs the first two on each Python file Claude edits. The script
  is still what must pass before committing: only it runs mypy.
