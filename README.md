# QSMEF — Quantum Software Engineering Module Evaluation Framework

QSMEF is a methodological framework for evaluating the functional contribution of components within quantum software implementations.

The framework analyzes how individual functional components contribute to a specific property of interest while preserving the structural and operational context of the analyzed implementation.

## Overview

QSMEF combines:

- Functional decomposition of quantum software implementations.
- Coalition-based evaluation through component neutralization.
- Functional metrics defined from Hermitian observables.
- Cooperative game theory and Shapley values for component-level attribution.

For a set of functional components \(B\), QSMEF constructs partial configurations by replacing absent components with identity operations while preserving the Hilbert space and the operational order of the components that remain.

For each coalition \(C \subseteq B\), the induced quantum state is evaluated through a functional metric derived from a Hermitian observable \(H\).

The characteristic function is defined as:

\[
v(C) = M_H(\rho_C) - M_H(\rho_{\emptyset})
\]

where

\[
M_H(\rho) = \mathrm{Tr}(H\rho).
\]

The resulting characteristic function is used to compute the Shapley value of each component, providing a quantitative measure of its functional contribution with respect to the selected observable.

The Shapley values are not absolute measures of component importance. Their sign and magnitude must be interpreted with respect to the selected functional observable and the adopted functional decomposition.

## Case Studies

### SKW Quantum Search on the Hypercube

The first case study applies QSMEF to the Shenvi–Kempe–Whaley (SKW) quantum search algorithm on a hypercube.

The implementation is decomposed into three functional components:

- **O** — Oracle
- **G** — Grover coin
- **S** — Flip-flop shift

The contribution of these components is evaluated with respect to an energy-based functional property.

Two configurations are analyzed:

1. The correct SKW implementation.
2. A modified implementation in which the oracle phase is changed.

This allows the functional contribution profiles obtained with QSMEF to be compared when the behavior of one component is altered.

Detailed information about the case study, experimental configuration, and results is available in:

[`case_studies/skw_hypercube/`](case_studies/skw_hypercube/)


### Shor Order-Finding Applicability Analysis

A second case analyzes the applicability of QSMEF to the quantum order-finding stage of Shor's algorithm.

Unlike the SKW case study, this example is **not presented as a valid application of QSMEF**. The analysis shows that the functional interdependence among the preparation stage, controlled modular operations, and inverse Quantum Fourier Transform prevents the coalition-induced configurations from preserving the common functional semantics required by QSMEF.

This case is therefore included to document an **applicability boundary of the framework**.

The corresponding circuit implementation and methodological rationale are available in:

[`case_studies/shor_applicability/`](case_studies/shor_applicability/)


## Repository Structure

```text
QSMEF-Quantum-CaseStudies/
│
├── qsmef/
│   ├── __init__.py
│   ├── coalitions.py
│   ├── game.py
│   ├── metrics.py
│   └── shapley.py
│
├── case_studies/
│   ├── skw_hypercube/
│   │   ├── results/
│   │   │   ├── skw_correct.png
│   │   │   ├── skw_correct.pdf
│   │   │   ├── skw_incorrect_oracle_phase.png
│   │   │   └── skw_incorrect_oracle_phase.pdf
│   │   ├── __init__.py
│   │   ├── experiment.py
│   │   ├── observables.py
│   │   ├── skw.py
│   │   ├── visualization.py
│   │   └── README.md
│   │
│   └── shor_applicability/
│       ├── __init__.py
│       ├── shor_circuit.py
│       └── README.md
│
├── literature_review/
│   ├── data/
│   │   ├── retrieved_studies.csv
│   │   ├── excluded_studies.csv
│   │   └── selected_studies_analysis.xlsm
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `qsmef` package contains the reusable components of the framework, including coalition generation, characteristic-function construction, functional metric evaluation, and Shapley-value computation.

The `case_studies` directory contains the SKW application case and the Shor order-finding applicability analysis.

The `literature_review` directory contains the research artifacts used to support the literature review, including the retrieved studies, excluded studies, and the analysis of the selected studies.

## Requirements

The current implementation requires:

- Python 3
- NumPy
- Matplotlib
- Qiskit

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running the SKW Case Study

From the repository root, run:

```bash
python -m case_studies.skw_hypercube.experiment
```

The experiment evaluates the functional contribution of the Oracle, Grover coin, and flip-flop shift components over the selected analysis window.

The generated results can be found in:

```text
case_studies/skw_hypercube/results/
```
## Running the Shor Applicability Example

The Shor order-finding circuit can be constructed from the repository root with:

```bash
python -c "from case_studies.shor_applicability.shor_circuit import build_shor_order_finding_circuit; qc = build_shor_order_finding_circuit(); print(qc)"
```

This example is provided to support the applicability analysis of QSMEF. It does not compute QSMEF contributions or Shapley values.

Further details are available in:

[`case_studies/shor_applicability/`](case_studies/shor_applicability/)


## Interpretation

QSMEF attributes the variation of the selected functional metric among the components of the analyzed implementation.

For each component \(b \in B\), the Shapley value \(\phi_b\) represents its average marginal contribution across all possible coalitions.

By the efficiency property of the Shapley value:

\[
\sum_{b \in B} \phi_b
=
v(B)
=
M_H(\rho_B) - M_H(\rho_{\emptyset}).
\]

This provides a consistency relation between the component-level contributions and the global functional variation measured by QSMEF.

## Scope

QSMEF is intended for quantum software implementations that admit a meaningful functional decomposition and for which the selected observable preserves the same semantic interpretation across the evaluated coalitions.

The framework does not assume that every decomposition of a quantum implementation is admissible. Component neutralization must preserve the structural conditions required for meaningful comparison between configurations.
