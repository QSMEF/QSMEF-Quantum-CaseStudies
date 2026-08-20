# Experimental Results

This directory contains the main graphical outputs generated from the
QSMEF analysis of the Shenvi–Kempe–Whaley (SKW) quantum search on the hypercube.

## Experimental configuration

- Hypercube dimension: n = 8
- Number of vertices: N = 256
- Target vertex: w = 0
- Number of steps: T = 24
- Analysis window: t = 15,...,23
- Reference step: t = 19
- Functional components: B = {O, G, S}
- Operational order: O → G → S
- Energy parameter: γ = 1

The functional components are:

- **O** — Oracle
- **G** — Grover coin
- **S** — Flip-flop shift

The functional evaluation uses the energy-based observable:

H_ener = I_C ⊗ (-γ A_P - |w><w|)

## Correct implementation

`skw_correct.png` shows the Shapley contribution profile for the
original SKW implementation.

At the reference step t = 19, the oracle presents the dominant
functional contribution:

φ_O(19) = -0.253055

The efficiency property is numerically satisfied up to floating-point
error.

## Modified oracle phase

`skw_incorrect_oracle_phase.png` shows the contribution profile obtained
after replacing the ideal oracle phase π with:

exp(i 0.70π)

At t = 19, the oracle contribution changes from:

φ_O(19) = -0.253055

to:

φ_O_modified(19) = 0.134201

with:

Δφ_O(19) = 0.387256

## Interpretation

The comparison illustrates how QSMEF can quantify changes in the
functional contribution profile when a localized modification is
introduced into a component of the implementation.

The Shapley values are not absolute measures of component importance.
Their sign and magnitude must be interpreted with respect to the
selected functional observable and the adopted functional decomposition.
