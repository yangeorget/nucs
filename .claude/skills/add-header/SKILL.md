---
name: add-header
description: Skill to add the NuCS header to each Python file. Use it whenever you create a new Python file under `nucs/` or `tests/`.
---

# Add header

Every Python file under `nucs/` and `tests/` starts with the ASCII-art copyright banner in `header.txt`.
New files need it. To re-stamp existing files:

```bash
addheader nucs -t header.txt
addheader tests -t header.txt
```