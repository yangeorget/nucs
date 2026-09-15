#!/bin/bash

# PostToolUse hook: lint and format the Python file Claude just wrote, when it lives under nucs/, tests/ or scripts/.
# Exit code 2 sends ruff's unfixable findings back to Claude. mypy stays in scripts/bash/style.sh: it checks the whole
# tree, too slow to run on every edit.
#
# F401 is reported but never fixed here: edits arrive one at a time, so an import added a step before the code that
# uses it is briefly unused, and fixing it would delete it. scripts/bash/style.sh still removes a truly unused one.

file=$(jq -r '.tool_input.file_path // empty')
case "$file" in
  "$CLAUDE_PROJECT_DIR"/nucs/*.py | "$CLAUDE_PROJECT_DIR"/tests/*.py | "$CLAUDE_PROJECT_DIR"/scripts/*.py) ;;
  *) exit 0 ;;
esac

if ! command -v ruff > /dev/null; then
  echo "ruff is not on PATH: $file was not checked" >&2
  exit 1
fi

cd "$CLAUDE_PROJECT_DIR" || exit 1
lint=$(ruff check --fix --unfixable F401 "$file" 2>&1)
lint_status=$?
ruff format --quiet "$file"
if [ $lint_status -ne 0 ]; then
  echo "$lint" >&2
  exit 2
fi
