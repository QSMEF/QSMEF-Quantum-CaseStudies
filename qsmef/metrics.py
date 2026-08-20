import numpy as np


def functional_metric(state, observable):
    """
    Evaluate the functional metric associated with a Hermitian observable.

    The metric is defined as:

        M_H(rho) = Tr(H rho)

    For a pure state |psi>, this is equivalent to:

        M_H(|psi>) = <psi| H |psi>

    Parameters
    ----------
    state : numpy.ndarray
        Quantum state represented either as a state vector or
        as a density matrix.

    observable : numpy.ndarray
        Hermitian operator representing the functional property
        being evaluated.

    Returns
    -------
    float
        Expected value of the observable for the given state.

    Raises
    ------
    ValueError
        If the observable is not Hermitian or if its dimension
        is incompatible with the quantum state.
    """

    state = np.asarray(state, dtype=complex)
    observable = np.asarray(observable, dtype=complex)

    if observable.ndim != 2 or observable.shape[0] != observable.shape[1]:
        raise ValueError("The observable must be a square matrix.")

    if not np.allclose(observable, observable.conj().T):
        raise ValueError("The observable must be Hermitian.")

    # Pure state |psi>
    if state.ndim == 1:

        if observable.shape[0] != state.shape[0]:
            raise ValueError(
                "The observable dimension is incompatible with the state."
            )

        value = np.vdot(state, observable @ state)

    # Density matrix rho
    elif state.ndim == 2:

        if state.shape[0] != state.shape[1]:
            raise ValueError("The density matrix must be square.")

        if observable.shape != state.shape:
            raise ValueError(
                "The observable dimension is incompatible with the state."
            )

        value = np.trace(observable @ state)

    else:
        raise ValueError(
            "The quantum state must be a state vector or a density matrix."
        )

    return float(np.real_if_close(value))
