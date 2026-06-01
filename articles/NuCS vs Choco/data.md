NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.all_interval_series
-n 500 --symmetry-breaking --no-display-solutions --log-level=ERROR --cp-max-height 10000 --var-heuristic 3
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 9
--symmetry-breaking --no-display-solutions --consistency-algorithm 0
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 9
--symmetry-breaking --no-display-solutions
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.latin_square
-n 20 --no-find-all --no-display-solutions --cp-max-height 10000
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence --no-display-solutions -n 100
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma --cp 10000 --no-find-all --log-level ERROR
--no-display-solutions --no-symmetry-breaking -n 100

java -cp classes:../choco-solver-6.0.1-light.jar:lib/pf4cs-1.0.5.jar:lib/args4j-2.33.jar
org.chocosolver.samples.integer.AllIntervalSeries -o 500
java -cp classes:../choco-solver-6.0.1-light.jar:lib/pf4cs-1.0.5.jar:lib/args4j-2.33.jar
org.chocosolver.samples.integer.GolombRuler -m 10
java -cp classes:../choco-solver-6.0.1-light.jar:lib/pf4cs-1.0.5.jar:lib/args4j-2.33.jar
org.chocosolver.samples.integer.LatinSquare -n 80

| problem type      | problem name        | problem size | solver with parameters                      | time in ms |
|-------------------|---------------------|--------------|---------------------------------------------|------------|
| find 1 solution   | all interval series | 500          | choco(BC, ff)                               | 240        |
| find 1 solution   | all interval series | 1000         | choco(BC, ff)                               | 392        |
| find 1 solution   | all interval series | 2000         | choco(BC, ff)                               | 950        |
| find 1 solution   | all interval series | 4000         | choco(BC, ff)                               | 3261       |
| find 1 solution   | all interval series | 8000         | choco(BC, ff)                               | 16595      |
| find 1 solution   | all interval series | 500          | nucs(BC, ff)                                | 225        |
| find 1 solution   | all interval series | 1000         | nucs(BC, ff)                                | 398        |
| find 1 solution   | all interval series | 2000         | nucs(BC, ff)                                | 972        |
| find 1 solution   | all interval series | 4000         | nucs(BC, ff)                                | 3352       |
| find 1 solution   | all interval series | 8000         | nucs(BC, ff)                                | 15153      |
| minimize          | golomb ruler        | 9            | choco(BC, enumerated domains)               | 202        |
| minimize          | golomb ruler        | 10           | choco(BC, enumerated domains)               | 481        |
| minimize          | golomb ruler        | 11           | choco(BC, enumerated domains)               | 6800       |
| minimize          | golomb ruler        | 12           | choco(BC, enumerated domains)               | 65705      |
| minimize          | golomb ruler        | 9            | nucs(BC)                                    | 438        |
| minimize          | golomb ruler        | 10           | nucs(BC)                                    | 883        |
| minimize          | golomb ruler        | 11           | nucs(BC)                                    | 12428      |
| minimize          | golomb ruler        | 12           | nucs(BC)                                    | 128040     |
| minimize          | golomb ruler        | 9            | nucs(custom consistency)                    | 414        |
| minimize          | golomb ruler        | 10           | nucs(custom consistency)                    | 641        |
| minimize          | golomb ruler        | 11           | nucs(custom consistency)                    | 6791       |
| minimize          | golomb ruler        | 12           | nucs(custom consistency)                    | 67319      |
| find 1 solution   | latin square        | 20           | choco(AC)                                   | 121        |
| find 1 solution   | latin square        | 40           | choco(AC)                                   | 315        |
| find 1 solution   | latin square        | 60           | choco(AC)                                   | 1415       |
| find 1 solution   | latin square        | 80           | choco(AC)                                   | 8247       |
| find 1 solution   | latin square        | 20           | nucs(BC, redundant constraints)             | 376        |
| find 1 solution   | latin square        | 40           | nucs(BC, redundant constraints)             | 398        |
| find 1 solution   | latin square        | 60           | nucs(BC, redundant constraints)             | 473        |
| find 1 solution   | latin square        | 80           | nucs(BC, redundant constraints)             | 638        |
| find 1 solution   | magic sequence      | 100          | choco(AC, GCC mostly)                       | 587        |
| find 1 solution   | magic sequence      | 150          | choco(AC, GCC mostly)                       | 2444       |
| find 1 solution   | magic sequence      | 200          | choco(AC, GCC mostly)                       | 11181      |
| find 1 solution   | magic sequence      | 250          | choco(AC, GCC mostly)                       | 27346      |
| find 1 solution   | magic sequence      | 100          | nucs(BC, no GCC but 1 redundant constraint) | 481        |
| find 1 solution   | magic sequence      | 150          | nucs(BC, no GCC but 1 redundant constraint) | 532        |
| find 1 solution   | magic sequence      | 200          | nucs(BC, no GCC but 1 redundant constraint) | 552        |
| find 1 solution   | magic sequence      | 250          | nucs(BC, no GCC but 1 redundant constraint) | 709        |
| prove no solution | schur lemma         | 100          | choco(BC)                                   | 197        |
| prove no solution | schur lemma         | 200          | choco(BC)                                   | 333        |
| prove no solution | schur lemma         | 400          | choco(BC)                                   | 702        |
| prove no solution | schur lemma         | 800          | choco(BC)                                   | 2701       |
| prove no solution | schur lemma         | 100          | nucs(BC)                                    | 418        |
| prove no solution | schur lemma         | 200          | nucs(BC)                                    | 542        |
| prove no solution | schur lemma         | 400          | nucs(BC)                                    | 1050       |
| prove no solution | schur lemma         | 800          | nucs(BC)                                    | 3118       |
