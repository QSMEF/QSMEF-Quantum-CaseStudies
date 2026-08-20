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

The resulting characteristic function is used to compute the Shapley value of each component, providing a quantitative measure of its functional contribution.

## Case Studies

### SKW Quantum Search on the Hypercube

The first case study applies QSMEF to the Shenvi–Kempe–Whaley (SKW) quantum search algorithm on a hypercube.

The implementation is decomposed into three functional components:

- **O** — Oracle
- **G** — Grover coin
- **S** — Flip-flop shift

The contribution of these components is evaluated with respect to an energy-based functional property.

## Repository Structure

The repository contains the QSMEF implementation, case-study code, experimental results, and supporting documentation.

