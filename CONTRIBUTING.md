# CONTRIBUTING.md

Guidance for human developers.

## Common commands

### Build the pip package

```bash
python -m build
```

### Publish the package

```bash
python -m twine upload --verbose dist/*
```

### Fix source file headers

```bash
addheader nucs -t header.txt
addheader tests -t header.txt
```

### Generate documentation

```bash
sphinx-build -M html docs/source docs/output
```
