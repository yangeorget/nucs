import pstats

pstats.Stats("profile.pstats").sort_stats(pstats.SortKey.TIME).print_stats()
