import itertools
import math


def compute_shapley_values(players, characteristic_function):
    """
    Compute Shapley values for a cooperative game.

    Parameters
    ----------
    players : iterable
        Functional components considered as players.

    characteristic_function : dict
        Mapping from coalitions (frozenset) to their characteristic
        values v(C).

    Returns
    -------
    dict
        Shapley value associated with each player.
    """

    players = list(players)

    phi = {player: 0.0 for player in players}

    for permutation in itertools.permutations(players):

        coalition = frozenset()

        for player in permutation:

            next_coalition = coalition | {player}

            phi[player] += (
                characteristic_function[next_coalition]
                - characteristic_function[coalition]
            )

            coalition = next_coalition

    normalization = math.factorial(len(players))

    for player in players:
        phi[player] /= normalization

    return phi
