"""
Shor order-finding circuit used for the QSMEF applicability analysis.

This module constructs the quantum phase-estimation structure used in
Shor's order-finding procedure for N = 15.

The circuit is included as an applicability-boundary case for QSMEF.
It provides the implementation structure required to analyze whether
the functional decomposition satisfies the methodological conditions
of the framework.

No QSMEF contribution or Shapley-value computation is performed here.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import QFTGate


# ---------------------------------------------------------------------
# Modular multiplication
# ---------------------------------------------------------------------

def amod15(a: int) -> Gate:
    """
    Construct the modular multiplication operator

        |y> -> |a * y mod 15>

    on a four-qubit work register.

    Parameters
    ----------
    a : int
        Integer coprime with 15. Supported values are
        {2, 4, 7, 8, 11, 13}.

    Returns
    -------
    Gate
        Qiskit gate implementing modular multiplication by a modulo 15.
    """

    if a not in {2, 4, 7, 8, 11, 13}:
        raise ValueError(
            "'a' must be coprime with 15 and belong to "
            "{2, 4, 7, 8, 11, 13}."
        )

    circuit = QuantumCircuit(4, name=f"{a} mod 15")

    if a in {2, 13}:
        circuit.swap(2, 3)
        circuit.swap(1, 2)
        circuit.swap(0, 1)

    if a in {7, 8}:
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        circuit.swap(2, 3)

    if a in {4, 11}:
        circuit.swap(1, 3)
        circuit.swap(0, 2)

    if a in {7, 11, 13}:
        for qubit in range(4):
            circuit.x(qubit)

    return circuit.to_gate()


def controlled_amod15(a: int) -> Gate:
    """
    Return the controlled version of the modular multiplication operator.

    The resulting gate contains one control qubit and four target qubits.
    """

    return amod15(a).control(1)


# ---------------------------------------------------------------------
# Inverse Quantum Fourier Transform
# ---------------------------------------------------------------------

def inverse_qft(num_qubits: int) -> Gate:
    """
    Construct the inverse Quantum Fourier Transform.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the counting register.

    Returns
    -------
    Gate
        Inverse QFT gate.
    """

    qft_gate = QFTGate(num_qubits)
    qft_dagger = qft_gate.inverse()
    qft_dagger.label = "QFT†"

    return qft_dagger


# ---------------------------------------------------------------------
# Shor order-finding circuit
# ---------------------------------------------------------------------

def build_shor_order_finding_circuit(
    a: int = 2,
    N: int = 15,
    precision: int = 4,
    measure: bool = True,
) -> QuantumCircuit:
    """
    Construct the quantum phase-estimation circuit used in Shor's
    order-finding procedure.

    This implementation is specialized for N = 15.

    The circuit contains:

    1. initialization of the work register in |1>,
    2. uniform superposition of the counting register,
    3. controlled modular exponentiation,
    4. inverse Quantum Fourier Transform,
    5. optional measurement of the counting register.

    Parameters
    ----------
    a : int
        Base used for modular exponentiation.
    N : int
        Integer to factor. This implementation supports N = 15 only.
    precision : int
        Number of qubits in the counting register.
    measure : bool
        If True, measure the counting register.

    Returns
    -------
    QuantumCircuit
        Shor order-finding circuit.
    """

    if N != 15:
        raise NotImplementedError(
            "This implementation is specialized for N = 15."
        )

    count = QuantumRegister(precision, "count")
    work = QuantumRegister(4, "work")

    if measure:
        classical = ClassicalRegister(precision, "c")
        circuit = QuantumCircuit(count, work, classical)
    else:
        circuit = QuantumCircuit(count, work)

    # Prepare the work register in |1>.
    circuit.x(work[0])

    # Prepare the counting register in uniform superposition.
    circuit.h(count)

    # Controlled modular exponentiation.
    controlled_u = controlled_amod15(a)

    for k in range(precision):
        for _ in range(2**k):
            circuit.append(
                controlled_u,
                [count[k]] + list(work),
            )

    # Inverse QFT on the counting register.
    circuit.append(
        inverse_qft(precision),
        count,
    )

    # Optional measurement.
    if measure:
        circuit.measure(count, classical)

    return circuit