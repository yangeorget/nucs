# Add header

Every Python file under `nucs/` and `tests/` starts with the ASCII-art copyright banner in `header.txt`.
New files need it. To re-stamp existing files:

```bash
addheader nucs -t header.txt
addheader tests -t header.txt
```