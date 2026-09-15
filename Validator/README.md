# QSMEF Universal Validator

This directory contains the computational artifact developed to operationalize
the applicability validation of QSMEF across different quantum software
implementations.

## Main Artifact

The `qsmef_universal_validator.py` file contains the executable implementation
of the computational artifact developed to operationalize the applicability
validation of QSMEF.

The implementation follows a modular architecture composed of:

- case-specific constructors;
- the common `QSMEFCase` structure;
- the Universal Validator;
- functional contribution computation using Shapley values;
- a multi-case orchestrator;
- an optional granularity assistant.

## Architecture

Each quantum implementation is represented by a case-specific constructor
that generates a `QSMEFCase` instance.

The instance contains the information required for the analysis:
initial state, operations, functional blocks, execution schedule,
observable, functional constraints, and semantic contract.

The Validator receives exclusively a `QSMEFCase` instance and applies the
same validation procedure independently of the quantum algorithm being
analyzed. Implementation-specific logic is declared in the corresponding
case constructor rather than encoded in the Universal Validator.

In simplified form, the workflow is:

`Case Constructor → QSMEFCase → Universal Validator → Validation Report`

When a case is classified as applicable, the artifact computes the functional
contributions of its blocks using Shapley values.

## Validation Outcomes

The Validator can produce four outcomes:

- `ERROR EN LA ESPECIFICACIÓN`
- `NO APLICABLE BAJO EL ALCANCE DECLARADO`
- `REQUIERE CONTRATO SEMÁNTICO`
- `APLICABLE`

Shapley values are computed only when the validation outcome is `APLICABLE`.

## Granularity

The granularity of the analysis is a methodological decision made by the
researcher and is not automatically determined by QSMEF.

The artifact includes an optional granularity assistant as a supporting tool
for exploring and comparing candidate granularities. Its use is optional and
does not replace the selection and justification made by the researcher.

## Included Cases

The artifact contains constructors and supports the execution of the QSMEF
procedure on six quantum implementations:

- QAOA-MaxCut;
- TwoLocal-H2;
- SKW quantum search on the hypercube;
- QPE applied to the order-finding procedure of Shor's algorithm;
- quantum classifier;
- graph partitioning.

The first four correspond to the main cases analyzed in the thesis. The
remaining two are included as additional cases of the computational artifact.

## Execution

The artifact can be executed with:

```bash
python qsmef_universal_validator.py
```

By default, the execution uses the granularities declared in the case
constructors. The optional granularity assistant remains disabled during
this execution.

The multi-case orchestrator constructs the case instances, invokes the same
Universal Validator for each one, and gathers the resulting validation
reports.
