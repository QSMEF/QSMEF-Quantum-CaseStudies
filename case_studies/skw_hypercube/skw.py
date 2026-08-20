"""
SKW quantum search on the hypercube.

Quantum implementation used as a case study for QSMEF.

Functional components:
    O : Oracle
    G : Grover coin
    S : Flip-flop shift

Operational order:
    O -> G -> S
"""

import numpy as np


# ============================================================
# Hilbert-space dimensions and state representation
# ============================================================

def position_dim(n):
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << n


def coin_dim(n):
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    return n


def uniform_coin_state(n):
    return np.ones(int(n), dtype=complex) / np.sqrt(int(n))


def reshape_cp(psi_vec, n):
    return np.asarray(
        psi_vec,
        dtype=complex
    ).reshape(
        coin_dim(n),
        position_dim(n)
    )


def flatten_cp(psi_cp):
    return np.asarray(
        psi_cp,
        dtype=complex
    ).reshape(-1)


# ============================================================
# Grover coin
# ============================================================

def grover_coin_matrix(n):
    D = uniform_coin_state(n)

    return (
        2.0 * np.outer(D, D.conj())
        - np.eye(int(n), dtype=complex)
    )


# ============================================================
# Functional component O: Oracle
# ============================================================

def apply_oracle_cp(
    psi_cp,
    n,
    target,
    oracle_phase=-1.0
):
    """
    Apply the oracle phase to the marked vertex.

    The correct SKW oracle uses:

        oracle_phase = -1 = exp(i*pi)

    A different phase can be used to construct a modified
    implementation for comparison.
    """

    target = int(target)

    psi_out = psi_cp.copy()

    psi_out[:, target] *= oracle_phase

    return psi_out


# ============================================================
# Functional component G: Grover coin
# ============================================================

def apply_grover_coin_cp(psi_cp, n):
    G = grover_coin_matrix(n)

    return G @ psi_cp


# ============================================================
# Functional component S: Flip-flop shift
# ============================================================

def apply_shift_flip_flop_cp(psi_cp, n):
    n = int(n)

    dP = position_dim(n)

    out = np.zeros_like(
        psi_cp,
        dtype=complex
    )

    for a in range(n):

        for x in range(dP):

            y = x ^ (1 << a)

            out[a, y] += psi_cp[a, x]

    return out


# ============================================================
# Vector-state wrappers
# ============================================================

def apply_oracle_Rprime(
    psi,
    target,
    n,
    oracle_phase=-1.0
):
    return flatten_cp(
        apply_oracle_cp(
            reshape_cp(psi, n),
            n=n,
            target=target,
            oracle_phase=oracle_phase
        )
    )


def apply_grover_coin_locally(psi, n):
    return flatten_cp(
        apply_grover_coin_cp(
            reshape_cp(psi, n),
            n
        )
    )


def apply_shift_flip_flop(psi, n):
    return flatten_cp(
        apply_shift_flip_flop_cp(
            reshape_cp(psi, n),
            n
        )
    )


# ============================================================
# Complete SKW step
# ============================================================

def apply_step_skw(
    psi,
    n,
    target,
    order=("O", "G", "S"),
    oracle_phase=-1.0
):
    """
    Apply one complete SKW step while preserving
    the operational order of the functional components.
    """

    out = psi.copy()

    for op in order:

        if op == "O":

            out = apply_oracle_Rprime(
                out,
                target=target,
                n=n,
                oracle_phase=oracle_phase
            )

        elif op == "G":

            out = apply_grover_coin_locally(
                out,
                n
            )

        elif op == "S":

            out = apply_shift_flip_flop(
                out,
                n
            )

        else:

            raise ValueError(
                f"Unknown operator: {op}"
            )

    return out


# ============================================================
# Coalition-induced configuration
# ============================================================

def apply_coalition(
    psi,
    coalition,
    n,
    target,
    order=("O", "G", "S"),
    oracle_phase=-1.0
):
    """
    Apply the functional components belonging to coalition C.

    Components absent from C are neutralized by the identity
    operation. The original operational order is preserved.
    """

    out = psi.copy()

    for op in order:

        if op in coalition:

            if op == "O":

                out = apply_oracle_Rprime(
                    out,
                    target=target,
                    n=n,
                    oracle_phase=oracle_phase
                )

            elif op == "G":

                out = apply_grover_coin_locally(
                    out,
                    n
                )

            elif op == "S":

                out = apply_shift_flip_flop(
                    out,
                    n
                )

    return out


# ============================================================
# Initial state
# ============================================================

def initial_state(
    n,
    pos0=0,
    uniform_pos=True
):
    n = int(n)

    N = position_dim(n)

    coin_state = uniform_coin_state(n)

    if uniform_pos:

        pos_state = (
            np.ones(N, dtype=complex)
            / np.sqrt(N)
        )

    else:

        pos_state = np.zeros(
            N,
            dtype=complex
        )

        pos_state[int(pos0)] = 1.0

    return np.kron(
        coin_state,
        pos_state
    )


# ============================================================
# Success probability
# ============================================================

def success_probability(
    psi,
    target,
    n
):
    psi_cp = (
        reshape_cp(psi, n)
        if np.asarray(psi).ndim == 1
        else np.asarray(psi)
    )

    return float(
        np.real(
            np.sum(
                np.abs(
                    psi_cp[:, int(target)]
                ) ** 2
            )
        )
    )
