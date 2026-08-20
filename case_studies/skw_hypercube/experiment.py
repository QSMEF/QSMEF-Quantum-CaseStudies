"""
QSMEF experiment for the SKW quantum search on the hypercube.

This module connects:

    - the SKW implementation,
    - the energy-based functional metric,
    - coalition generation,
    - characteristic-function construction,
    - and Shapley-value computation.

The functional decomposition considered is:

    B = {O, G, S}

with operational order:

    O -> G -> S
"""

import numpy as np

from QSMEF.coalitions import generate_coalitions
from QSMEF.game import build_characteristic_function
from QSMEF.shapley import compute_shapley_values

from .skw import (
    initial_state,
    apply_step_skw,
    apply_coalition,
    success_probability,
)

from .observables import energy_metric


PLAYERS = ("O", "G", "S")


def evaluate_step(
    state,
    n,
    target,
    gamma=1.0,
    order=("O", "G", "S"),
    oracle_phase=-1.0
):
    """
    Apply QSMEF to one SKW step.

    For the current input state, this function:

        1. generates all coalitions,
        2. constructs the induced state for each coalition,
        3. evaluates the energy-based functional metric,
        4. builds the characteristic function v(C),
        5. computes the Shapley value of each functional component.

    Parameters
    ----------
    state : numpy.ndarray
        Input state of the analyzed SKW step.

    n : int
        Dimension of the hypercube.

    target : int
        Marked vertex.

    gamma : float
        Coupling parameter of the energy observable.

    order : tuple
        Operational order of the SKW functional components.

    oracle_phase : complex
        Phase applied by the oracle.

    Returns
    -------
    dict
        QSMEF evaluation for the current SKW step.
    """

    coalitions = generate_coalitions(PLAYERS)

    def coalition_application(reference_state, coalition):
        return apply_coalition(
            reference_state,
            coalition=coalition,
            n=n,
            target=target,
            order=order,
            oracle_phase=oracle_phase
        )

    def metric_function(current_state):
        return energy_metric(
            current_state,
            n=n,
            target=target,
            gamma=gamma
        )

    characteristic_function = build_characteristic_function(
        coalitions=coalitions,
        reference_state=state,
        apply_coalition=coalition_application,
        metric=metric_function
    )

    shapley_values = compute_shapley_values(
        PLAYERS,
        characteristic_function
    )

    full_coalition = frozenset(PLAYERS)

    return {
        "phi": shapley_values,
        "v": characteristic_function,
        "v_full": characteristic_function[full_coalition],
        "metric_before": metric_function(state),
        "metric_after_full": (
            metric_function(state)
            + characteristic_function[full_coalition]
        ),
    }


def run_and_collect(
    n=8,
    target=0,
    T=24,
    order=("O", "G", "S"),
    gamma=1.0,
    oracle_phase=-1.0,
    label="correct implementation",
    t_opt=19
):
    """
    Run the SKW implementation and evaluate QSMEF at every step.

    Parameters
    ----------
    n : int
        Hypercube dimension.

    target : int
        Marked vertex.

    T : int
        Number of SKW steps.

    order : tuple
        Operational order of the functional components.

    gamma : float
        Coupling parameter of the energy observable.

    oracle_phase : complex
        Phase used by the oracle.

    label : str
        Label identifying the evaluated implementation.

    t_opt : int
        Reference step used in the case-study analysis.

    Returns
    -------
    dict
        States, functional metric, success probability,
        Shapley values and total metric variation.
    """

    state = initial_state(
        n=n,
        pos0=0,
        uniform_pos=True
    )

    states = []
    metric_values = []
    success_probabilities = []

    phi_O = []
    phi_G = []
    phi_S = []

    delta_metric = []

    for step in range(T + 1):

        states.append(state.copy())

        metric_values.append(
            energy_metric(
                state,
                n=n,
                target=target,
                gamma=gamma
            )
        )

        success_probabilities.append(
            success_probability(
                state,
                target=target,
                n=n
            )
        )

        if step < T:

            evaluation = evaluate_step(
                state,
                n=n,
                target=target,
                gamma=gamma,
                order=order,
                oracle_phase=oracle_phase
            )

            phi_O.append(
                float(evaluation["phi"]["O"])
            )

            phi_G.append(
                float(evaluation["phi"]["G"])
            )

            phi_S.append(
                float(evaluation["phi"]["S"])
            )

            delta_metric.append(
                float(evaluation["v_full"])
            )

            state = apply_step_skw(
                state,
                n=n,
                target=target,
                order=order,
                oracle_phase=oracle_phase
            )

    return {
        "ts": np.arange(T + 1),
        "steps": np.arange(T),
        "states": states,
        "MH": np.array(metric_values),
        "PS": np.array(success_probabilities),

        "phi": {
            "O": np.array(phi_O),
            "G": np.array(phi_G),
            "S": np.array(phi_S),
        },

        "delta_E": np.array(delta_metric),

        "n": n,
        "target": target,
        "T": T,
        "order": order,
        "gamma": gamma,
        "oracle_phase": oracle_phase,
        "label": label,
        "t_opt": int(t_opt),
    }


if __name__ == "__main__":

    n = 8
    target = 0
    T = 24
    gamma = 1.0

    order = ("O", "G", "S")

    t_opt = 19

    # Correct implementation
    phase_correct = -1.0

    correct_results = run_and_collect(
        n=n,
        target=target,
        T=T,
        order=order,
        gamma=gamma,
        oracle_phase=phase_correct,
        label="correct implementation",
        t_opt=t_opt
    )

    # Modified implementation:
    # oracle phase = 0.70*pi instead of pi
    theta_modified = 0.70 * np.pi

    phase_modified = np.exp(
        1j * theta_modified
    )

    modified_results = run_and_collect(
        n=n,
        target=target,
        T=T,
        order=order,
        gamma=gamma,
        oracle_phase=phase_modified,
        label="modified implementation: incorrect oracle phase",
        t_opt=t_opt
    )

    print("QSMEF - SKW Hypercube Case Study")
    print("--------------------------------")
    print(f"n = {n}")
    print(f"N = {2 ** n}")
    print(f"target = {target}")
    print(f"T = {T}")
    print(f"gamma = {gamma}")
    print(f"order = {order}")

    print()
    print("Correct implementation:")
    print(
        "Maximum efficiency residual:",
        np.max(
            np.abs(
                correct_results["phi"]["O"]
                + correct_results["phi"]["G"]
                + correct_results["phi"]["S"]
                - correct_results["delta_E"]
            )
        )
    )

    print()
    print("Modified implementation:")
    print(
        "Maximum efficiency residual:",
        np.max(
            np.abs(
                modified_results["phi"]["O"]
                + modified_results["phi"]["G"]
                + modified_results["phi"]["S"]
                - modified_results["delta_E"]
            )
        )
    )
