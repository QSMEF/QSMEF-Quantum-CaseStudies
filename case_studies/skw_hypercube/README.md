# SKW Hypercube Case Study

This directory contains the application of QSMEF (Quantum Software Engineering Module Evaluation Framework) to the Shenvi–Kempe–Whaley (SKW) quantum search algorithm on the hypercube.

The purpose of this case study is to evaluate the functional contribution of the main components of the SKW implementation with respect to a selected functional property.

## Case Study

The SKW quantum search algorithm is implemented on an n-dimensional hypercube.

The experimental configuration used in this case study is:

- Hypercube dimension: `n = 8`
- Number of vertices: `N = 256`
- Target vertex: `w = 0`
- Number of steps: `T = 24`
- Analysis window: `t = 15,...,23`
- Reference step: `t = 19`

## Functional Decomposition

The implementation is decomposed into three functional components:

\[
B = \{O, G, S\}
\]

where:

- `O` — Oracle
- `G` — Grover coin
- `S` — Shift operator

The operational order of the components is preserved when evaluating every coalition.

When a component is absent from a coalition, its action is replaced by the identity operation. This preserves the Hilbert space and the structural context of the analyzed implementation.

## Functional Observable

The functional property analyzed in this case study is represented by an energy-based Hermitian observable:

\[
H_{\mathrm{ener}} =
I_C \otimes
\left(
-\gamma A_P - |w\rangle\langle w|
\right)
\]

with:

\[
\gamma = 1
\]

For each coalition \(C \subseteq B\), QSMEF evaluates the induced quantum state \(\rho_C\) and computes the functional metric:

\[
M_H(\rho_C) = \mathrm{Tr}(H\rho_C)
\]

The characteristic function is defined relative to the empty coalition:

\[
v(C) =
M_H(\rho_C) -
M_H(\rho_{\emptyset})
\]

## QSMEF Evaluation

For each step in the analysis window, the framework:

1. Generates all coalitions of the functional components.
2. Constructs the quantum state induced by each coalition.
3. Evaluates the selected functional observable.
4. Builds the characteristic function.
5. Computes the Shapley value of each component.

The resulting values quantify the contribution of each functional component with respect to the selected observable.

The Shapley values are not absolute measures of component importance. Their sign and magnitude must be interpreted with respect to the selected functional observable and the adopted functional decomposition.

## Experimental Scenarios

Two configurations are evaluated.

### Correct implementation

The first experiment evaluates the original SKW implementation using the standard oracle.

### Modified oracle

The second experiment introduces an oracle phase variation:

\[
e^{i0.70\pi}
\]

This modification is used to analyze how the functional contribution profile changes when the behavior of one component is altered.

## Results

The experimental results and generated figures are available in:

[Experimental results and figures](./results/)

The figures include:

- `skw_correct.png` — QSMEF contribution profile for the correct SKW implementation.
- `skw_incorrect_oracle_phase.png` — QSMEF contribution profile when the oracle phase is modified.

At the reference step `t = 19`, the correct implementation exhibits a dominant contribution from the oracle, while the Grover coin and shift components present smaller contributions.

The Shapley efficiency property is also verified numerically:

\[
\sum_{b \in B} \phi_b(t) = \Delta E_t
\]

up to numerical precision.

## Repository Structure


skw_hypercube/
├── __init__.py
├── skw.py
├── observables.py
├── experiment.py
├── visualization.py
├── Resultados/
│   ├── README.md
│   ├── skw_correct.png
│   └── skw_incorrect_oracle_phase.png
└── README.md
