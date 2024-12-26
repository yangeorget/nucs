# Useful scripts and commands to contribute to NUCS

## Code
### How to fix the header
```bash
addheader nucs -t header.txt 
addheader tests -t header.txt 
```

### How to check the coding style
```bash
./scripts/bash/style.sh    
```

### How to launch the tests
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/
```

### How to compute the code coverage
```bash
NUMBA_DISABLE_JIT=1 PYTHONPATH=. coverage run --source=nucs,tests -m pytest tests
coverage html
```

## Doc
### How to generate the doc
```bash
sphinx-build -M html docs/source docs/output
```

## Performance
### How to profile the code
```bash
NUMBA_DISABLE_JIT=1 python -m "cProfile" -s time -m nucs.examples.queens | more
```
```bash
NUMBA_DISABLE_JIT=1 viztracer --open -m nucs.examples.queens 
```

### How to measure the performance
```bash
time NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python -m nucs.examples.queens -n 12 
```

## Pip package
### How to build the package
```bash
python -m build
```

### How to publish the package
```bash
python -m twine upload --verbose dist/*
```

## Docker image
### How to create an image
```bash  
docker build -t nucs .  
```

### How to tag an image
```bash  
docker tag nucs yangeorget/nucs:<version>   
```

### How to publish an image
```bash  
docker push yangeorget/nucs:<version>  
```

### How to run a container
```bash  
docker run -it nucs bash  
```