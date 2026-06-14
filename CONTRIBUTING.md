# CONTRIBUTING.md

Guidance for human developers.

## Common commands

### Build the pip package

```bash
./scripts/bash/build.sh
```

This cleans `dist/` and `build/`, builds the dist and wheel with `python -m build`, and validates the
result with `twine check`.

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

### Minizinc Docker image

```bash
docker build -t minizinc-nucs .
```

```bash
docker tag minizinc-nucs:latest yangeorget/minizinc-nucs:version
```

```bash
docker push yangeorget/minizinc-nucs:version
```

```bash
docker pull yangeorget/minizinc-nucs:version
```

```bash
docker run --rm -v "$PWD:/work" -w /work yangeorget/minizinc-nucs:12.2.0 minizinc tmp/mzn/latinbool.mzn
```