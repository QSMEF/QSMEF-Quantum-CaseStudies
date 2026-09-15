# QPE/Shor Applicability Analysis

This directory contains the implementation used to analyze the
applicability of QSMEF to the quantum order-finding procedure considered
in Shor's algorithm.

The analysis is restricted to the quantum part of the order-finding
procedure and does not constitute an analysis of Shor's algorithm as a
whole.

Unlike the SKW hypercube case study, this example is **not presented as
a valid application of QSMEF**. Its purpose is to document a case in
which the applicability conditions of the framework are not satisfied
for the analyzed functional decomposition.

## Quantum order-finding circuit

The file `shor_circuit.py` implements the quantum structure used for the
order-finding analysis with the following configuration:

- \(N = 21\)
- \(a = 2\)
- a five-qubit counting register (\(m = 5\))
- a work register used to represent the modular states required by the
  order-finding procedure

The analyzed quantum procedure includes:

1. preparation of the registers,
2. Hadamard gates on the counting register,
3. controlled modular operations,
4. the inverse Quantum Fourier Transform (`QFT^{-1}`),
5. optional measurement of the counting register.

The measurement stage is not treated as a player in the QSMEF analysis.

## Analyzed functional decomposition

For the applicability analysis, the quantum procedure is considered as
a sequence of functionally related stages.

The preparation stage and the inverse `QFT^{-1}` are not treated as
players. The candidate players correspond to selected controlled
operations within the analyzed quantum order-finding structure.

QSMEF constructs coalition-induced configurations by neutralizing absent
players through the identity operation while preserving the execution
order of the original implementation.

The purpose of the applicability analysis is to determine whether these
partial configurations remain functionally and semantically comparable
under the declared scope.

## Why QSMEF is not applied

For the decomposition analyzed in this case, the preparation of the
quantum state, the controlled modular operations, and the inverse
`QFT^{-1}` participate in functionally dependent stages of the
order-finding procedure.

When selected controlled operations are neutralized to construct the
coalition-induced configurations, the functional dependencies required
between state preparation, controlled evolution, and the subsequent
decoding through the inverse `QFT^{-1}` are not preserved uniformly.

As a consequence, the resulting partial configurations cannot be
interpreted uniformly as functionally comparable variants of the
original quantum order-finding procedure under the declared scope.

Therefore, QSMEF is **not applicable to the functional decomposition
analyzed in this case**.

This conclusion is specific to the selected decomposition and semantic
scope. It should not be interpreted as establishing that QSMEF is
intrinsically inapplicable to Shor's algorithm or to every possible
decomposition of a quantum order-finding implementation.

For this reason, the case is included as an **applicability-boundary
example** rather than as a QSMEF functional-attribution case.

## Running the circuit

From the repository root:

```bash
python -c "from case_studies.shor_applicability.shor_circuit import build_shor_order_finding_circuit; qc = build_shor_order_finding_circuit(); print(qc)"
