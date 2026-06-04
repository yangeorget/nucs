NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma \
-n 100 --no-symmetry-breaking \
--no-display-solutions --no-display-stats --log-level ERROR

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.all_interval_series \
-n 500 --var-heuristic 3 --symmetry-breaking --cp-max-height 10000 \
--no-display-solutions --no-display-stats --log-level=ERROR

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb \
-n 9 --symmetry-breaking --consistency-algorithm 0 \
--no-display-solutions --no-display-stats --log-level ERROR

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.latin_square \
-n 20 --cp-max-height 10000 \
--no-display-solutions --no-display-stats --log-level ERROR
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.latin_square \
-n 20 --model-rc --cp-max-height 10000 \
--no-display-solutions --no-display-stats --log-level ERROR

NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence \
-n 100 --var-heuristic 3 --model-r1\
--no-display-solutions --no-display-stats --log-level ERROR
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence \
-n 100 --var-heuristic 3 --model-r1 --model-r2 \
--no-display-solutions --no-display-stats --log-level ERROR

java -cp classes:../choco-solver-6.0.1-light.jar:lib/pf4cs-1.0.5.jar:lib/args4j-2.33.jar
org.chocosolver.samples.integer.AllIntervalSeries -o 500
java -cp classes:../choco-solver-6.0.1-light.jar:lib/pf4cs-1.0.5.jar:lib/args4j-2.33.jar
org.chocosolver.samples.integer.GolombRuler -m 10
java -cp classes:../choco-solver-6.0.1-light.jar:lib/pf4cs-1.0.5.jar:lib/args4j-2.33.jar
org.chocosolver.samples.integer.LatinSquare -n 80

| problem type      | problem name        | problem size | solver with parameters                        | time in ms     |
|-------------------|---------------------|--------------|-----------------------------------------------|----------------|
| prove no solution | schur lemma         | 100          | choco(BC)                                     | 197            |
| prove no solution | schur lemma         | 200          | choco(BC)                                     | 333            |
| prove no solution | schur lemma         | 400          | choco(BC)                                     | 702            |
| prove no solution | schur lemma         | 800          | choco(BC)                                     | 2701           |
| prove no solution | schur lemma         | 1600         | choco(BC)                                     | 17864          |
| prove no solution | schur lemma         | 100          | nucs(BC)                                      | 418            |
| prove no solution | schur lemma         | 200          | nucs(BC)                                      | 542            |
| prove no solution | schur lemma         | 400          | nucs(BC)                                      | 1050           |
| prove no solution | schur lemma         | 800          | nucs(BC)                                      | 3118           |
| prove no solution | schur lemma         | 1600         | nucs(BC)                                      | 14592          |
| find 1 solution   | all interval series | 500          | choco(BC, ff)                                 | 240            |
| find 1 solution   | all interval series | 1000         | choco(BC, ff)                                 | 392            |
| find 1 solution   | all interval series | 2000         | choco(BC, ff)                                 | 950            |
| find 1 solution   | all interval series | 4000         | choco(BC, ff)                                 | 3261           |
| find 1 solution   | all interval series | 8000         | choco(BC, ff)                                 | 16595          |
| find 1 solution   | all interval series | 16000        | choco(BC, ff)                                 | 120340         |
| find 1 solution   | all interval series | 500          | nucs(BC, ff)                                  | 225            |
| find 1 solution   | all interval series | 1000         | nucs(BC, ff)                                  | 398            |
| find 1 solution   | all interval series | 2000         | nucs(BC, ff)                                  | 972            |
| find 1 solution   | all interval series | 4000         | nucs(BC, ff)                                  | 3352           |
| find 1 solution   | all interval series | 8000         | nucs(BC, ff)                                  | 15153          |
| find 1 solution   | all interval series | 16000        | nucs(BC, ff)                                  | 85236          |
| minimize          | golomb ruler        | 9            | choco(BC, enumerated domains)                 | 202            |
| minimize          | golomb ruler        | 10           | choco(BC, enumerated domains)                 | 481            |
| minimize          | golomb ruler        | 11           | choco(BC, enumerated domains)                 | 6800           |
| minimize          | golomb ruler        | 12           | choco(BC, enumerated domains)                 | 65705          |
| minimize          | golomb ruler        | 9            | nucs(BC)                                      | 438            |
| minimize          | golomb ruler        | 10           | nucs(BC)                                      | 883            |
| minimize          | golomb ruler        | 11           | nucs(BC)                                      | 12428          |
| minimize          | golomb ruler        | 12           | nucs(BC)                                      | 128040         |
| minimize          | golomb ruler        | 9            | nucs(custom consistency)                      | 414            |
| minimize          | golomb ruler        | 10           | nucs(custom consistency)                      | 641            |
| minimize          | golomb ruler        | 11           | nucs(custom consistency)                      | 6791           |
| minimize          | golomb ruler        | 12           | nucs(custom consistency)                      | 67319          |
| find 1 solution   | latin square        | 20           | choco(BC)                                     | 105            |
| find 1 solution   | latin square        | 30           | choco(BC)                                     | 126            |
| find 1 solution   | latin square        | 40           | choco(BC)                                     | 180            |
| find 1 solution   | latin square        | 50           | choco(BC)                                     | did not finish |
| find 1 solution   | latin square        | 20           | nucs(BC)                                      | 376            |
| find 1 solution   | latin square        | 30           | nucs(BC)                                      | 380            |
| find 1 solution   | latin square        | 40           | nucs(BC)                                      | 397            |
| find 1 solution   | latin square        | 50           | nucs(BC)                                      | did not finish |
| find 1 solution   | latin square        | 20           | nucs(BC, redundant constraints)               | 412            |
| find 1 solution   | latin square        | 30           | nucs(BC, redundant constraints)               | 453            |
| find 1 solution   | latin square        | 40           | nucs(BC, redundant constraints)               | 547            |
| find 1 solution   | latin square        | 50           | nucs(BC, redundant constraints)               | 728            |
| find 1 solution   | magic sequence      | 100          | choco(AC, ff)                                 | 149            |
| find 1 solution   | magic sequence      | 200          | choco(AC, ff)                                 | 307            |
| find 1 solution   | magic sequence      | 300          | choco(AC, ff)                                 | 779            |
| find 1 solution   | magic sequence      | 400          | choco(AC, ff)                                 | 1783           |
| find 1 solution   | magic sequence      | 100          | nucs(BC, ff)                                  | 416            |
| find 1 solution   | magic sequence      | 200          | nucs(BC, ff)                                  | 710            |
| find 1 solution   | magic sequence      | 300          | nucs(BC, ff)                                  | 1451           |
| find 1 solution   | magic sequence      | 400          | nucs(BC, ff)                                  | 2913           |
| find 1 solution   | magic sequence      | 100          | nucs(BC, ff, additional redundant constraint) | 387            |
| find 1 solution   | magic sequence      | 200          | nucs(BC, ff, additional redundant constraint) | 455            |
| find 1 solution   | magic sequence      | 300          | nucs(BC, ff, additional redundant constraint) | 624            |
| find 1 solution   | magic sequence      | 400          | nucs(BC, ff, additional redundant constraint) | 942            |

