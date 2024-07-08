from ncs.problems.problem import Problem


class AlphaProblem(Problem):
    """
    This problem comes from the newsgroup rec.puzzle.
    The numbers from 1 to 26 are assigned to the letters of the alphabet.
    The numbers beside each word are the total of the values assigned to the letters in the word
    (e.g for LYRE: L,Y,R,E might be to equal 5,9,20 and 13 or any other combination that add up to 47).

    Find the value of each letter under the equations:
    BALLET  45     GLEE  66     POLKA      59     SONG     61
    CELLO   43     JAZZ  58     QUARTET    50     SOPRANO  82
    CONCERT 74     LYRE  47     SAXOPHONE 134     THEME    72
    FLUTE   30     OBOE  53     SCALE      51     VIOLIN  100
    FUGUE   50     OPERA 65     SOLO       37     WALTZ    34
    """

    def __init__(self, n: int):
        super().__init__(
            shared_domains=list(range(1, 27)),
            domain_indices=list(range(26)),
            domain_offsets=[0] * 26,
        )
        self.set_propagators(
            [
                # TODO
            ]
        )
