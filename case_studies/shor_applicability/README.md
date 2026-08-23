# Shor Applicability Analysis

This directory contains the implementation used to analyze the applicability of QSMEF to the order-finding stage of Shor's algorithm.

Unlike the SKW hypercube case study, this example is **not presented as a valid application of QSMEF**. Its purpose is to document a case in which the applicability conditions of the framework are not satisfied.

## Circuit

The file `shor_circuit.py` implements the quantum order-finding structure for:

- \(N = 15\)
- \(a = 2\)
- a four-qubit counting register
- a four-qubit work register

The implementation includes:

1. preparation of the work register,
2. Hadamard gates on the counting register,
3. controlled modular multiplication operations,
4. the inverse Quantum Fourier Transform (QFT),
5. optional measurement of the counting register.

## Why QSMEF is not applied

QSMEF evaluates functional components by constructing coalition-induced configurations in which absent components are neutralized while the structural and semantic context required for comparison is preserved.

In the order-finding circuit, the preparation stage, controlled modular operations, and inverse QFT are functionally interdependent. Neutralizing these stages changes the conditions under which the remaining stages operate and therefore prevents the resulting configurations from preserving a common functional interpretation.

Consequently, the coalition-induced states cannot be considered directly comparable under the same functional semantics required by QSMEF.

For this reason, this case is included as an **applicability-boundary example** rather than as a QSMEF evaluation case.

## Running the circuit

From the repository root:

```bash
python -c "from case_studies.shor_applicability.shor_circuit import build_shor_order_finding_circuit; qc = build_shor_order_finding_circuit(); print(qc)"