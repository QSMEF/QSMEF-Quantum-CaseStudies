import itertools


def generate_coalitions(players):
    """
    Generate all coalitions of a set of functional components.

    Parameters
    ----------
    players : iterable
        Functional components considered as players.

    Returns
    -------
    list of frozenset
        All possible coalitions, including the empty coalition.
    """

    players = list(players)
    coalitions = []

    for size in range(len(players) + 1):
        for combination in itertools.combinations(players, size):
            coalitions.append(frozenset(combination))

    return coalitions
