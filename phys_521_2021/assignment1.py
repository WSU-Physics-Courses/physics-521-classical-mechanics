"""Assignment 1: Monty Hall Problem
"""


def play_monty_hall():
    """Return `(stick_wins, switch_wins)` playing one round of Monty Hall.

    Results
    -------
    stick_wins : bool
        `True` if sticking with the original door wins the prize.
    switch_wins : bool
        `True` if switching doors wins the prize.
    """


def sample_monty_hall(N=1000, seed=3):
    """Return a list of the results of `N` cames after the number of wins `(stick_wins, switch_wins)` for sticking or switching
    with the traditional Monty Hall problem.

    The game goes as follows:

    1. There is a single prize behind one of

    Arguments
    ---------
    Nsamples : int
       Number of samples to generate.
    seed : int
       Seed to pass to `np.random.seed(seed)` before generating samples.
    """


def _samples_monty_hall(
    prize_probabilities=(1, 1, 1), monty_probabilities=(1, 1), Nsamples=1000, seed=3
):
    """Return

    Assume (without loss of generality) that the contestant chooses the first door.  If
    the prize is behind the first door,
    Monty Hall chooses

    Arguments
    ---------
    prize_probabilities : list
       List of relative probabilities for the prize being behind the corresponding
       door.  These need not be normalized.
    monty_probabilities : list
       List of relative probabilities for Monty choosing from the available doors.
       These need not be normalized.
    Nsamples : int
       Number of samples to generate.
    seed : int
       Seed to pass to `np.random.seed(seed)` before generating samples.
    """
