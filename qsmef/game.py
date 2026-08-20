def build_characteristic_function(
    coalitions,
    reference_state,
    apply_coalition,
    metric
):
    """
    Build the characteristic function used by QSMEF.

    For each coalition C:

        v(C) = M_H(rho_C) - M_H(rho_empty)

    where rho_C is the state induced by coalition C and rho_empty
    is the state associated with the empty coalition.

    Parameters
    ----------
    coalitions : iterable
        Coalitions of functional components.

    reference_state :
        Input state of the analyzed functional segment.

    apply_coalition : callable
        Function that receives the reference state and a coalition,
        and returns the state induced by that coalition.

    metric : callable
        Functional metric M_H used to evaluate each induced state.

    Returns
    -------
    dict
        Characteristic function mapping each coalition to v(C).
    """

    empty_coalition = frozenset()

    rho_empty = apply_coalition(
        reference_state,
        empty_coalition
    )

    baseline = metric(rho_empty)

    characteristic_function = {}

    for coalition in coalitions:

        rho_c = apply_coalition(
            reference_state,
            coalition
        )

        characteristic_function[coalition] = (
            metric(rho_c) - baseline
        )

    return characteristic_function
