"""
Energy-based observable used in the SKW case study.

The functional property evaluated in this case study is defined by

    H_ener = I_C ⊗ (-gamma A_P - |w><w|)

where:

    I_C   : identity operator on the coin space
    A_P   : adjacency operator of the hypercube
    |w>   : marked position state
    gamma : coupling parameter

The expectation value of this observable is used as the
functional metric for the QSMEF evaluation of the SKW case study.
"""

import numpy as np

from .skw import (
    position_dim,
    reshape_cp,
    flatten_cp,
)


def energy_expectation(
    state,
    n,
    target,
    gamma=1.0
):
    """
    Evaluate the expectation value of the energy observable.

    The observable is

        H_ener = I_C ⊗ (-gamma A_P - |w><w|)

    and the returned value corresponds to

        <H_ener>

    for the supplied quantum state.

    Parameters
    ----------
    state : numpy.ndarray
        Quantum state of the SKW walk.

    n : int
        Dimension of the hypercube.

    target : int
        Marked vertex |w>.

    gamma : float
        Coupling parameter of the energy observable.

    Returns
    -------
    float
        Expectation value of H_ener.
    """

    n = int(n)
    target = int(target)

    N = position_dim(n)

    psi_cp = (
        reshape_cp(state, n)
        if np.asarray(state).ndim == 1
        else np.asarray(state, dtype=complex)
    )

    norm = np.linalg.norm(
        flatten_cp(psi_cp)
    )

    if norm == 0:
        raise ValueError(
            "The quantum state has zero norm."
        )

    if abs(norm - 1.0) > 1e-10:
        psi_cp = psi_cp / norm

    # Contribution of the marked-position projector |w><w|
    prob_target = np.sum(
        np.abs(psi_cp[:, target]) ** 2
    )

    # Expectation value of the hypercube adjacency operator A_P
    adjacency_expectation = 0.0 + 0.0j

    for a in range(n):

        for x in range(N):

            y = x ^ (1 << a)

            adjacency_expectation += np.vdot(
                psi_cp[:, x],
                psi_cp[:, y]
            )

    return float(
        np.real(
            -gamma * adjacency_expectation
            - prob_target
        )
    )


def energy_metric(
    state,
    n,
    target,
    gamma=1.0
):
    """
    Functional metric used by QSMEF in the SKW case study.

    M_H(state) = <H_ener>
    """

    return energy_expectation(
        state,
        n=n,
        target=target,
        gamma=gamma
    )
