---
name: add-header
description: Stamps the NuCS ASCII-art copyright banner from header.txt onto Python files. Use when creating a Python file under nucs/, tests/ or scripts/, when a file is missing its banner, or when re-stamping banners in bulk.
---

# Add header

Every Python file under `nucs/`, `tests/` and `scripts/` starts with the ASCII-art copyright banner in `header.txt`.

- **A new file**: make the banner its first lines, copied from `header.txt`.
- **Existing files, in bulk**: re-stamp a whole tree.

  ```bash
  addheader nucs -t header.txt
  addheader tests -t header.txt
  addheader scripts -t header.txt
  ```

  `addheader` touches only `*.py` files and skips `__init__.py` by default. Add `-n` to list the files it would change
  without changing them.
