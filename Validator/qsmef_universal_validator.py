# -*- coding: utf-8 -*-
"""QSMEF Universal Validator.

Artefacto computacional para validar la aplicabilidad de QSMEF de forma
independiente del algoritmo. Incluye el contrato QSMEFCase, el motor
universal de validación, constructores de casos, cálculo de Shapley,
asistente opcional de granularidad y orquestador multicaso.
"""

from __future__ import annotations

import contextlib
import io
import itertools
from dataclasses import dataclass, field
from math import factorial, log2
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

print("Dependencias comunes cargadas correctamente.")

"""## 2. Contrato común y motor validador

"""


# ============================================================
# 3. CONTRATO COMÚN Y MOTOR VALIDADOR UNIVERSAL QSMEF
# ============================================================

"""
Asistente Universal de Aplicabilidad de QSMEF.

El motor de validación es independiente del algoritmo.

Cada experimento se proporciona al validador mediante un objeto
QSMEFCase. De esta manera, el motor no contiene lógica específica de
MaxCut, H2, SKW, QPE ni de ningún otro algoritmo particular.

Cada algoritmo debe adaptarse previamente al formato común definido
por QSMEFCase.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    # Si el código no se ejecuta dentro de Jupyter/Colab,
    # se utiliza print como alternativa a display.
    display = print


# ============================================================
# TIPOS AUXILIARES
# ============================================================

Array = np.ndarray

# Función que recibe un estado cuántico y devuelve el estado
# resultante luego de aplicar una operación.
ApplyFunction = Callable[[Array], Array]

# Función utilizada para evaluar el observable funcional.
ObservableFunction = Callable[[Array, str], complex]

# Función utilizada para comprobar una restricción del caso.
ConstraintFunction = Callable[
    [Array, str, float],
    tuple[bool, float, str]
]

# Función utilizada para comprobar un criterio semántico.
SemanticCriterionFunction = Callable[
    [Array, str, frozenset[str], complex, float],
    tuple[bool, float, str],
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _dagger(matrix: Array) -> Array:
    """
    Devuelve la adjunta de una matriz.

    La adjunta se obtiene conjugando los elementos complejos
    y transponiendo la matriz.
    """
    return np.asarray(matrix, dtype=complex).conj().T


def _max_abs(matrix: Array) -> float:
    """
    Devuelve el mayor valor absoluto presente en una matriz.

    Esta función se utiliza para medir errores o desviaciones
    numéricas durante las validaciones.
    """
    values = np.asarray(matrix)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _coalition_label(
    indices: Sequence[int],
    names: Sequence[str]
) -> str:
    """
    Construye una etiqueta legible para representar una coalición.

    Por ejemplo:
        ∅
        {R0}
        {R0, E0}
    """
    if not indices:
        return "∅"

    return "{" + ", ".join(
        names[index] for index in indices
    ) + "}"


# ============================================================
# ESPECIFICACIÓN DE UNA OPERACIÓN
# ============================================================

@dataclass(frozen=True)
class OperationSpec:
    """
    Representa una operación del circuito cuántico.

    La operación puede proporcionarse de dos maneras:

    1. Mediante una matriz.
    2. Mediante una función que aplica directamente la operación
       sobre un estado cuántico.

    Si ``declared_unitary=True``, se indica que la operación es
    unitaria por construcción y que el validador no necesita
    comprobar su unitariedad mediante una matriz explícita.

    Esta opción resulta útil para operaciones grandes cuya matriz
    completa no se construye, como el operador shift de SKW.
    """

    action: Union[Array, ApplyFunction]

    # Indica si la operación se declara unitaria por construcción.
    declared_unitary: bool = False

    # Justificación de por qué la operación puede considerarse unitaria
    # sin realizar una comprobación numérica.
    unitarity_rationale: str = ""

    def is_matrix(self) -> bool:
        """
        Indica si la operación fue proporcionada como una matriz.
        """
        return not callable(self.action)

    def apply(
        self,
        state: Array,
        state_kind: str
    ) -> Array:
        """
        Aplica la operación sobre el estado cuántico recibido.

        Si la operación está definida mediante una función,
        solamente puede aplicarse sobre un vector de estado.

        Si está definida mediante una matriz:
        - para un vector de estado se calcula U|ψ>;
        - para una matriz densidad se calcula UρU†.
        """

        # Caso 1: la operación está implementada mediante una función.
        if callable(self.action):

            if state_kind != "statevector":
                raise ValueError(
                    "Las operaciones definidas mediante funciones "
                    "solo admiten vectores de estado."
                )

            return np.asarray(
                self.action(state),
                dtype=complex
            )

        # Caso 2: la operación está representada mediante una matriz.
        matrix = np.asarray(
            self.action,
            dtype=complex
        )

        if state_kind == "statevector":
            # Evolución de un vector de estado:
            # |ψ'> = U|ψ>
            return matrix @ state

        # Evolución de una matriz densidad:
        # ρ' = UρU†
        return matrix @ state @ _dagger(matrix)

# ============================================================
# ESPECIFICACIÓN DE RESTRICCIONES DE ADMISIBILIDAD
# ============================================================

@dataclass(frozen=True)
class ConstraintSpec:
    """
    Representa una restricción de admisibilidad que el validador puede ejecutar.

    Cada restricción tiene:

    - un nombre;
    - una función que verifica si la restricción se cumple;
    - una descripción opcional.

    La restricción se evalúa sobre cada estado generado por las coaliciones.
    """

    name: str
    evaluator: ConstraintFunction
    description: str = ""

    def evaluate(
        self,
        state: Array,
        state_kind: str,
        tolerance: float
    ) -> tuple[bool, float, str]:
        """
        Evalúa la restricción sobre un estado cuántico.

        Devuelve:

        - True o False según se cumpla la condición;
        - el valor numérico obtenido;
        - un detalle de la evaluación.
        """

        passed, value, detail = self.evaluator(
            state,
            state_kind,
            tolerance
        )

        return bool(passed), float(value), str(detail)


# ============================================================
# RESTRICCIÓN DEFINIDA MEDIANTE UN PROYECTOR
# ============================================================

def projector_constraint(
    name: str,
    projector: Array,
    description: str = "",
) -> ConstraintSpec:
    """
    Crea una restricción basada en un proyector P.

    La condición exige que el estado se encuentre completamente
    dentro del subespacio definido por P.

    Para un vector de estado se verifica:

        <ψ|P|ψ> = 1

    Para una matriz densidad se verifica:

        Tr(Pρ) = 1

    Debido a errores numéricos, se admite una pequeña tolerancia.
    """

    P = np.asarray(
        projector,
        dtype=complex
    )

    def evaluator(
        state: Array,
        state_kind: str,
        tolerance: float
    ) -> tuple[bool, float, str]:

        # Si el estado está representado mediante un vector |ψ>,
        # calculamos <ψ|P|ψ>.
        if state_kind == "statevector":
            value = float(
                np.real(
                    np.vdot(
                        state,
                        P @ state
                    )
                )
            )

        # Si el estado está representado mediante una matriz densidad ρ,
        # calculamos Tr(Pρ).
        else:
            value = float(
                np.real(
                    np.trace(
                        P @ state
                    )
                )
            )

        # La condición ideal es value = 1.
        # El residuo mide cuánto se aleja el resultado de ese valor.
        residual = abs(
            1.0 - value
        )

        # La restricción se considera satisfecha si el residuo
        # está dentro de la tolerancia numérica permitida.
        return (
            residual <= tolerance,
            value,
            f"residuo={residual:.3e}"
        )

    return ConstraintSpec(
        name=name,
        evaluator=evaluator,
        description=description
    )

@dataclass(frozen=True)
class SemanticCriterion:
    ##Invariante semántico formalizable que se evalúa para cada coalición.

    name: str
    evaluator: SemanticCriterionFunction
    description: str = ""

    def evaluate(
        self,
        state: Array,
        state_kind: str,
        coalition: frozenset[str],
        expected: complex,
        tolerance: float,
    ) -> tuple[bool, float, str]:
        passed, value, detail = self.evaluator(
            state, state_kind, coalition, expected, tolerance
        )
        return bool(passed), float(value), str(detail)


@dataclass(frozen=True)
class SemanticContract:
 ##  """Soporte ejecutable para una interpretación común entre coaliciones.

##El investigador establece el alcance semántico en ``description`` y
##proporciona invariantes formalizables en ``criteria``. El validador,
##y no el investigador, determina si todas las coaliciones satisfacen
##dichos invariantes.

    description: str = ""
    criteria: Sequence[SemanticCriterion] = field(default_factory=tuple)


def semantic_projector_criterion(
    name: str,
    projector: Array,
    description: str = "",
) -> SemanticCriterion:
   ##"""Requiere que el estado de cada coalición permanezca en un subespacio declarado."""
    P = np.asarray(projector, dtype=complex)

    def evaluator(
        state: Array,
        state_kind: str,
        coalition: frozenset[str],
        expected: complex,
        tolerance: float,
    ) -> tuple[bool, float, str]:
        del coalition, expected
        if state_kind == "statevector":
            value = float(np.real(np.vdot(state, P @ state)))
        else:
            value = float(np.real(np.trace(P @ state)))
        residual = abs(1.0 - value)
        return residual <= tolerance, value, f"residuo={residual:.3e}"

    return SemanticCriterion(name=name, evaluator=evaluator, description=description)


def semantic_interval_criterion(
    name: str,
    lower: float,
    upper: float,
    description: str = "",
) -> SemanticCriterion:
    ##Requiere que el observable común conserve un rango numérico declarado."""

    low = float(lower)
    high = float(upper)
    if low > high:
        raise ValueError("El límite inferior no puede superar al superior.")

    def evaluator(
        state: Array,
        state_kind: str,
        coalition: frozenset[str],
        expected: complex,
        tolerance: float,
    ) -> tuple[bool, float, str]:
        del state, state_kind, coalition
        value = float(expected.real)
        imaginary_ok = abs(expected.imag) <= tolerance
        interval_ok = low - tolerance <= value <= high + tolerance
        detail = (
            f"intervalo=[{low:.6g}, {high:.6g}], "
            f"parte_imaginaria={expected.imag:.3e}"
        )
        return imaginary_ok and interval_ok, value, detail

    return SemanticCriterion(name=name, evaluator=evaluator, description=description)


@dataclass(frozen=True)
class QSMEFCase:
    ##Contrato de entrada estándar e independiente del algoritmo para la validación de QSMEF.
    ##operations`` contiene tanto las operaciones correspondientes a los jugadores
    ##como las operaciones fijas. ``players`` identifica las operaciones que las
    ##coaliciones pueden neutralizar. Toda operación incluida en ``schedule`` que no
    ##sea un jugador permanece fija para todas las coaliciones.

    name: str
    initial_state: Array
    operations: Mapping[str, Union[OperationSpec, Array, ApplyFunction]]
    players: Sequence[str]
    schedule: Sequence[str]
    observable: Union[Array, ObservableFunction]
    constraints: Sequence[ConstraintSpec] = field(default_factory=tuple)
    semantic_contract: SemanticContract = field(default_factory=SemanticContract)
    property_description: str = "No especificada"
    tolerance: float = 1e-10


def _normalize_operations(
    operations: Mapping[str, Union[OperationSpec, Array, ApplyFunction]]
) -> dict[str, OperationSpec]:
    normalized: dict[str, OperationSpec] = {}
    for raw_name, raw_operation in operations.items():
        name = str(raw_name)
        if name in normalized:
            raise ValueError(f"Operación repetida: {name}")
        normalized[name] = (
            raw_operation
            if isinstance(raw_operation, OperationSpec)
            else OperationSpec(raw_operation)
        )
    if not normalized:
        raise ValueError("El caso debe declarar al menos una operación.")
    return normalized


def _check_state(
    state: Array, tolerance: float
) -> tuple[str, bool, float, float]:
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        norm = float(np.real(np.vdot(state, state)))
        residual = abs(norm - 1.0)
        return "statevector", residual <= tolerance, norm, residual

    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        hermiticity = _max_abs(state - _dagger(state))
        trace_value = np.trace(state)
        trace_residual = float(abs(trace_value - 1.0))
        eigenvalues = np.linalg.eigvalsh((state + _dagger(state)) / 2)
        minimum = float(np.min(eigenvalues).real)
        valid = (
            hermiticity <= tolerance
            and trace_residual <= tolerance
            and minimum >= -tolerance
        )
        return "density_matrix", valid, float(np.real(trace_value)), max(
            hermiticity, trace_residual, max(0.0, -minimum)
        )

    raise ValueError(
        "initial_state debe ser un vector o una matriz de densidad cuadrada."
    )


def _expectation(
    observable: Union[Array, ObservableFunction],
    state: Array,
    state_kind: str,
) -> complex:
    if callable(observable):
        return complex(observable(state, state_kind))
    H = np.asarray(observable, dtype=complex)
    if state_kind == "statevector":
        return complex(np.vdot(state, H @ state))
    return complex(np.trace(H @ state))


def _validate_qsmef_case_core(case: QSMEFCase, show_tables: bool = True) -> dict[str, Any]:
   ##Valida una instancia de QSMEFCase sin incluir lógica específica de ningún algoritmo."""
    tolerance = float(case.tolerance)
    if tolerance <= 0:
        raise ValueError("La tolerancia debe ser positiva.")

    state0 = np.asarray(case.initial_state, dtype=complex)
    state_kind, initial_valid, initial_value, initial_residual = _check_state(
        state0, tolerance
    )
    dimension = state0.shape[0]
    operations = _normalize_operations(case.operations)
    players = [str(name) for name in case.players]
    schedule = [str(name) for name in case.schedule]

    specification_errors: list[str] = []
    if len(players) != len(set(players)):
        specification_errors.append("La lista de jugadores contiene duplicados.")
    unknown_players = sorted(set(players) - set(operations))
    unknown_schedule = sorted(set(schedule) - set(operations))
    if unknown_players:
        specification_errors.append(f"Jugadores no definidos: {unknown_players}")
    if unknown_schedule:
        specification_errors.append(
            f"Operaciones de la secuencia no definidas: {unknown_schedule}"
        )
    if not schedule:
        specification_errors.append("La secuencia de ejecución está vacía.")
    unused_players = sorted(set(players) - set(schedule))
    if unused_players:
        specification_errors.append(
            f"Jugadores que no aparecen en la secuencia: {unused_players}"
        )

    observable_hermitian: Optional[bool]
    observable_residual: float
    if callable(case.observable):
        observable_hermitian = None
        observable_residual = np.nan
    else:
        H = np.asarray(case.observable, dtype=complex)
        if H.shape != (dimension, dimension):
            specification_errors.append(
                f"El observable tiene dimensión {H.shape}; "
                f"se esperaba {(dimension, dimension)}."
            )
        observable_residual = _max_abs(H - _dagger(H))
        observable_hermitian = observable_residual <= tolerance

    operation_rows: list[dict[str, Any]] = []
    all_operations_unitary = True
    for name, operation in operations.items():
        role = "player" if name in players else "fixed"
        if operation.is_matrix():
            matrix = np.asarray(operation.action, dtype=complex)
            if matrix.shape != (dimension, dimension):
                specification_errors.append(
                    f"La operación {name} tiene dimensión {matrix.shape}; "
                    f"se esperaba {(dimension, dimension)}."
                )
                unitary = False
                residual = np.inf
                evidence = "dimensión inválida"
            elif operation.declared_unitary:
                unitary = True
                residual = np.nan
                evidence = "declarada: " + (
                    operation.unitarity_rationale or "sin justificación textual"
                )
            else:
                identity = np.eye(dimension, dtype=complex)
                residual = max(
                    _max_abs(_dagger(matrix) @ matrix - identity),
                    _max_abs(matrix @ _dagger(matrix) - identity),
                )
                unitary = residual <= tolerance
                evidence = "verificación matricial"
        else:
            unitary = bool(operation.declared_unitary)
            residual = np.nan
            evidence = "declarada: " + (
                operation.unitarity_rationale or "requerida para operador funcional"
            )
        all_operations_unitary = all_operations_unitary and unitary
        operation_rows.append(
            {
                "operación": name,
                "rol": role,
                "apariciones": schedule.count(name),
                "unitaria": unitary,
                "evidencia_unitariedad": evidence,
                "residuo_unitariedad": residual,
            }
        )

    operations_df = pd.DataFrame(operation_rows)

    if specification_errors:
        verdict = "ERROR EN LA ESPECIFICACIÓN"
        reason = " ".join(specification_errors)
        report = {
            "case": case.name,
            "verdict": verdict,
            "reason": reason,
            "specification_errors": specification_errors,
            "operations": operations_df,
            "coalitions": pd.DataFrame(),
            "constraints": pd.DataFrame(),
        }
        print(f"RESULTADO: {verdict}\nMOTIVO: {reason}")
        return report

    coalition_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for size in range(len(players) + 1):
        for indices in itertools.combinations(range(len(players)), size):
            coalition_players = {players[index] for index in indices}
            state = state0.copy()
            for operation_name in schedule:
                is_player = operation_name in players
                if is_player and operation_name not in coalition_players:
                    continue
                state = operations[operation_name].apply(state, state_kind)

            _, state_valid, norm_or_trace, state_residual = _check_state(
                state, tolerance
            )
            expected = _expectation(case.observable, state, state_kind)
            expected_is_real = abs(expected.imag) <= tolerance
            coalition = _coalition_label(indices, players)

            coalition_constraints_pass = True
            for constraint in case.constraints:
                passed, value, detail = constraint.evaluate(
                    state, state_kind, tolerance
                )
                coalition_constraints_pass = coalition_constraints_pass and passed
                constraint_rows.append(
                    {
                        "coalición": coalition,
                        "restricción": constraint.name,
                        "cumple": passed,
                        "valor": value,
                        "detalle": detail,
                    }
                )

            coalition_semantics_pass = bool(case.semantic_contract.criteria)
            coalition_set = frozenset(coalition_players)
            for criterion in case.semantic_contract.criteria:
                passed, value, detail = criterion.evaluate(
                    state, state_kind, coalition_set, expected, tolerance
                )
                coalition_semantics_pass = coalition_semantics_pass and passed
                semantic_rows.append(
                    {
                        "coalición": coalition,
                        "criterio_semántico": criterion.name,
                        "cumple": passed,
                        "valor": value,
                        "detalle": detail,
                    }
                )

            coalition_rows.append(
                {
                    "coalición": coalition,
                    "tamaño": size,
                    "estado_válido": state_valid,
                    "norma_o_traza": norm_or_trace,
                    "residuo_estado": state_residual,
                    "M_H": float(expected.real),
                    "parte_imaginaria": float(expected.imag),
                    "valor_real": expected_is_real,
                    "restricciones_cumplidas": coalition_constraints_pass,
                    "contrato_semántico_cumplido": coalition_semantics_pass,
                }
            )

    coalitions_df = pd.DataFrame(coalition_rows)
    constraints_df = pd.DataFrame(constraint_rows)
    semantics_df = pd.DataFrame(semantic_rows)
    baseline = float(
        coalitions_df.loc[coalitions_df["coalición"] == "∅", "M_H"].iloc[0]
    )
    coalitions_df["v(C)"] = coalitions_df["M_H"] - baseline

    all_states_valid = bool(coalitions_df["estado_válido"].all())
    all_values_real = bool(coalitions_df["valor_real"].all())
    all_constraints_pass = bool(coalitions_df["restricciones_cumplidas"].all())
    semantic_contract_specified = bool(case.semantic_contract.criteria)
    all_semantic_criteria_pass = bool(
        coalitions_df["contrato_semántico_cumplido"].all()
    )
    technical_admissible = (
        initial_valid
        and all_operations_unitary
        and observable_hermitian is not False
        and all_states_valid
        and all_values_real
    )
    automatic_admissible = technical_admissible and all_constraints_pass

    if not technical_admissible:
        verdict = "NO APLICABLE BAJO EL ALCANCE DECLARADO"
        failed = []
        if not initial_valid:
            failed.append("estado inicial inválido")
        if not all_operations_unitary:
            failed.append("operaciones no unitarias")
        if observable_hermitian is False:
            failed.append("observable no hermitiano")
        if not all_states_valid:
            failed.append("estados de coalición inválidos")
        if not all_values_real:
            failed.append("valores esperados no reales")
        reason = ", ".join(failed) + "."

    elif not all_constraints_pass:
       verdict = "NO APLICABLE BAJO EL ALCANCE DECLARADO"
       failed_constraints = constraints_df.loc[
        ~constraints_df["cumple"],
        ["coalición", "restricción"]
       ]
       reason = (
        "Las restricciones funcionales no se cumplen en "
        f"{len(failed_constraints)} coalición(es). "
        "Por lo tanto, no se satisface el alcance funcional declarado."
       )


    elif not semantic_contract_specified:
        verdict = "REQUIERE CONTRATO SEMÁNTICO"
        reason = (
            "Las condiciones técnicas se satisfacen, pero el caso no especifica "
            "invariantes semánticos ejecutables."
        )
    elif not all_semantic_criteria_pass:
         verdict = "NO APLICABLE BAJO EL ALCANCE DECLARADO"
         reason = (
              "Los criterios del contrato semántico no se satisfacen "
              "en todas las coaliciones. Por lo tanto, no se conserva "
              "la interpretación funcional común requerida bajo el "
              "alcance declarado."
        )

    else:

        verdict = "APLICABLE"

        reason = (
            "Se satisfacen las condiciones técnicas, las restricciones "
            "funcionales y todos los invariantes del contrato semántico."
        )


    summary_df = pd.DataFrame(
        [
            {
                "nivel": "Especificación",
                "condición": "Caso completo",
                "resultado": True
            },
            {
                "nivel": "Automático",
                "condición": "Estado inicial válido",
                "resultado": initial_valid
            },
            {
                "nivel": "Automático",
                "condición": "Operaciones unitarias",
                "resultado": all_operations_unitary
            },
            {
                "nivel": "Automático",
                "condición": "Observable hermitiano",
                "resultado": observable_hermitian
            },
            {
                "nivel": "Automático",
                "condición": "Estados válidos",
                "resultado": all_states_valid
            },
            {
                "nivel": "Automático",
                "condición": "Valores esperados reales",
                "resultado": all_values_real
            },
            {
                "nivel": "Restricciones",
                "condición": "Todas satisfechas",
                "resultado": all_constraints_pass
            },
            {
                "nivel": "Semántico",
                "condición": "Contrato especificado",
                "resultado": semantic_contract_specified
            },
            {
                "nivel": "Semántico",
                "condición": "Mismo espacio de estados",
                "resultado": all_states_valid
            },
            {
                "nivel": "Semántico",
                "condición": "Mismo observable del caso",
                "resultado": True
            },
            {
                "nivel": "Semántico",
                "condición": "Invariantes satisfechos",
                "resultado": all_semantic_criteria_pass
            },
        ]
    )


    print("=" * 76)
    print("QSMEF APPLICABILITY ASSISTANT")
    print("=" * 76)

    print(f"Caso: {case.name}")
    print(f"Propiedad: {case.property_description}")
    print(f"Representación: {state_kind}; dimensión: {dimension}")

    print(
        f"Jugadores: {len(players)}; "
        f"operaciones fijas: {len(set(schedule) - set(players))}"
    )

    print(
        f"Pasos de ejecución: {len(schedule)}; "
        f"coaliciones: {len(coalitions_df)}"
    )

    print(f"Baseline: {baseline:.12f}")
    print(f"RESULTADO: {verdict}")
    print(f"MOTIVO: {reason}\n")


    if show_tables:

        print("RESUMEN")
        display(summary_df)

        print("OPERACIONES")
        display(operations_df)

        print("COALICIONES")
        display(coalitions_df)

        if not constraints_df.empty:
            print("RESTRICCIONES FUNCIONALES")
            display(constraints_df)

        if not semantics_df.empty:
            print("CONTRATO SEMÁNTICO")
            display(semantics_df)


    return {
        "case": case.name,
        "verdict": verdict,
        "reason": reason,
        "automatic_admissible": automatic_admissible,
        "summary": summary_df,
        "operations": operations_df,
        "coalitions": coalitions_df,
        "constraints": constraints_df,
        "semantics": semantics_df,
        "semantic_contract_specified": semantic_contract_specified,
        "all_semantic_criteria_pass": all_semantic_criteria_pass,
        "baseline": baseline,
    }

# ============================================================
# 4. INTERFAZ PÚBLICA DEL VALIDATOR UNIVERSAL QSMEF
# ============================================================

def validate_qsmef_case(
    case: QSMEFCase,
    show_tables: bool = True
) -> dict[str, Any]:
    """
    Ejecuta el procedimiento universal de validación de QSMEF.

    La función recibe exclusivamente una instancia de QSMEFCase.
    No identifica el algoritmo ni incorpora lógica específica
    de MaxCut, H2, SKW, QPE u otras implementaciones.

    Toda la información específica del caso debe estar contenida
    en QSMEFCase mediante:

    - estado inicial;
    - operaciones;
    - jugadores;
    - secuencia;
    - observable;
    - restricciones funcionales;
    - contrato semántico.
    """

    return _validate_qsmef_case_core(
        case,
        show_tables=show_tables
    )


print("Validator universal QSMEF cargado correctamente.")

"""## 3. Constructores de casos cuánticos

### 3.1 Graph Partitioning
"""


def build_graph_partitioning_case(
    n=6,
    edges=None,
    gamma=0.5,
    beta=0.5,
):
    """
    Construye el caso QSMEF para Minimum Bisection.

    Problema: dado un grafo G con n par, encontrar un subconjunto
    V0 de V con |V0| = n/2 tal que se minimice el número de aristas
    entre V0 y V \\ V0.

    Invariante físico: peso de Hamming = n/2.

    Mixer: XY ring mixer sobre el grafo completo (para n=6, se
    conecta cada qubit con su vecino en un anillo).
    """

    import numpy as np

    if n % 2 != 0:
        raise ValueError("n debe ser par para Minimum Bisection.")

    if edges is None:
        # Por defecto: anillo C_n
        edges = [(i, (i + 1) % n) for i in range(n)]

    dim = 2 ** n

    # ========================================================
    # ESTADO INICIAL: superposición uniforme de estados con HW = n/2
    # ========================================================

    half = n // 2

    psi0 = np.zeros(dim, dtype=complex)
    for z in range(dim):
        if bin(z).count("1") == half:
            psi0[z] = 1.0

    # Normalización
    n_states = np.sum(np.abs(psi0) ** 2)
    psi0 = psi0 / np.sqrt(n_states)

    # ========================================================
    # OPERADOR DE FASE (diagonal)
    # ========================================================

    def cost_value(z):
        """Cuenta aristas cortadas para la cadena de bits z."""
        bits = [(z >> (n - 1 - q)) & 1 for q in range(n)]
        return sum(bits[i] != bits[j] for i, j in edges)

    costs = np.array([cost_value(z) for z in range(dim)])

    # Fase: exp(-i gamma H_C)
    # H_C diagonal con autovalores = número de aristas cortadas
    phase_diagonal = np.exp(-1j * gamma * costs)
    C_fase = np.diag(phase_diagonal)

    # ========================================================
    # MIXER XY RING (preserva peso de Hamming)
    # H_ring = sum_{u} (X_u X_{u+1} + Y_u Y_{u+1})
    # ========================================================

    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1.0 + 0.0j]])
        for op in ops:
            result = np.kron(result, op)
        return result

    def two_qubit_op(op_a, op_b, i, j, n):
        """Aplica op_a en i, op_b en j, identidad en el resto."""
        ops = []
        for q in range(n):
            if q == i:
                ops.append(op_a)
            elif q == j:
                ops.append(op_b)
            else:
                ops.append(I2)
        return kron_all(ops)

    H_XY = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        j = (i + 1) % n
        H_XY += two_qubit_op(X, X, i, j, n)
        H_XY += two_qubit_op(Y, Y, i, j, n)

    # exp(-i beta H_XY) usando exponencial matricial
    from scipy.linalg import expm
    M_XY = expm(-1j * beta * H_XY)

    # ========================================================
    # MIXER ESTÁNDAR (NO preserva peso de Hamming)
    # Para comparar: H_std = sum X_j
    # ========================================================

    H_std = np.zeros((dim, dim), dtype=complex)
    for q in range(n):
        ops = [X if p == q else I2 for p in range(n)]
        H_std += kron_all(ops)

    M_std = expm(-1j * beta * H_std)

    # ========================================================
    # OBSERVABLE: número de aristas cortadas
    # ========================================================

    H_cost = np.diag(costs.astype(complex))

    # ========================================================
    # PROYECTOR SOBRE EL SUBESPACIO HW = n/2
    # ========================================================

    P_half = np.zeros((dim, dim), dtype=complex)
    for z in range(dim):
        if bin(z).count("1") == half:
            P_half[z, z] = 1.0

    # ========================================================
    # INFORMACIÓN DESCRIPTIVA
    # ========================================================

    print("\nConfiguración del caso Graph Partitioning (Min Bisection)")
    print("-" * 72)

    print(f"n = {n} (número de vértices del grafo)")
    print(f"Grafo: anillo C_{n} con {len(edges)} aristas")
    print(f"Cardinalidad fija: |V0| = {half}")
    print(f"Espacio de Hilbert: dimensión {dim}")
    print(f"Estado inicial: superposición uniforme de estados con HW = {half}")
    print(f"Número de estados iniciales: {int(n_states)}")

    print("\nInvariante físico:")
    print(f"  Peso de Hamming = {half}")

    print("\nProyector:")
    print("  P_half = suma de proyectores sobre estados con HW = n/2")

    print("\nJugadores QSMEF:")
    print("  C_fase = operador de fase (diagonal)")
    print("  M_XY   = mixer XY ring (preserva HW)")

    print("\nObservable:")
    print("  H_cost = número de aristas cortadas")

    print("-" * 72)

    # ========================================================
    # CONTRATO SEMÁNTICO
    # ========================================================

    semantic_criterion = semantic_projector_criterion(
        name="Permanencia en el sector de cardinalidad n/2",
        projector=P_half,
        description=(
            f"Toda coalición debe preservar el peso de Hamming = {half}. "
            "El mixer XY lo preserva por construcción; el mixer estándar no."
        ),
    )

    # ========================================================
    # CASO QSMEF (partición admisible)
    # ========================================================

    return QSMEFCase(
        name=f"Graph Partitioning (Min Bisection) n={n}",
        initial_state=psi0,
        operations={
            "C_fase": C_fase,
            "M_XY": M_XY,
        },
        players=["C_fase", "M_XY"],
        schedule=["C_fase", "M_XY"],
        observable=H_cost,
        constraints=[],
        semantic_contract=SemanticContract(
            description=(
                "La cardinalidad del corte se preserva bajo la evolución. "
                "El mixer XY respeta el invariante."
            ),
            criteria=(semantic_criterion,),
        ),
        property_description="Número esperado de aristas cortadas",
        tolerance=1e-8,
    )


print("build_graph_partitioning_case cargado correctamente.")



# ============================================================
# 2. CONSTRUCTORES DE CASOS QSMEF
# CONSTRUCTOR COMPLETO QAOA-MAXCUT C6 p=2
# ============================================================

"""### 3.2 QAOA–MaxCut

"""


def build_maxcut_case():

    n = 6

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
    ]

    # Parámetros QAOA p=2
    gamma1 = 0.6446548125
    beta1  = 0.3223274063
    gamma2 = 1.2484689205
    beta2  = 0.9261415143

    dim = 2 ** n

    # --------------------------------------------------------
    # Matrices básicas
    # --------------------------------------------------------

    I2_local = np.eye(2, dtype=complex)

    X_local = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)


    def kron_local(operators):

        result = np.array([[1.0 + 0.0j]])

        for op in operators:
            result = np.kron(result, op)

        return result


    # --------------------------------------------------------
    # Función de costo MaxCut
    # --------------------------------------------------------

    costs = np.zeros(dim)

    for z in range(dim):

        bits = [
            (z >> (n - 1 - q)) & 1
            for q in range(n)
        ]

        costs[z] = sum(
            bits[i] != bits[j]
            for i, j in edges
        )


    # Observable:
    # número de aristas cortadas
    H_cost = np.diag(
        costs.astype(complex)
    )


    # --------------------------------------------------------
    # Operador de costo
    # exp(-i gamma H_C)
    # --------------------------------------------------------

    def cost_unitary(gamma):

        return np.diag(
            np.exp(
                -1j * gamma * costs
            )
        )


    # --------------------------------------------------------
    # Mixer
    # exp(-i beta X)
    # --------------------------------------------------------

    def rx_beta(beta):

        return (
            np.cos(beta) * I2_local
            - 1j
            * np.sin(beta)
            * X_local
        )


    def mixer_unitary(beta):

        return kron_local([
            rx_beta(beta)
            for _ in range(n)
        ])


    # --------------------------------------------------------
    # Cuatro jugadores
    # --------------------------------------------------------

    C1 = cost_unitary(gamma1)
    B1 = mixer_unitary(beta1)

    C2 = cost_unitary(gamma2)
    B2 = mixer_unitary(beta2)


    # --------------------------------------------------------
    # Estado inicial |+>^6
    # --------------------------------------------------------

    psi0 = np.ones(
        dim,
        dtype=complex
    ) / np.sqrt(dim)


    #-----------------------------------------------------------
    # --------------------------------------------------------
    # Información descriptiva del caso QAOA-MaxCut
    # --------------------------------------------------------

    print("\nConfiguración del caso QAOA-MaxCut")
    print("-" * 60)

    print("Grafo: C6")
    print("  C6 = grafo ciclo de 6 vértices y 6 aristas.")

    print(f"Número de qubits: {n}")
    print("  Cada vértice del grafo se representa mediante un qubit.")

    print("Profundidad QAOA: p = 2")
    print("  p = número de capas QAOA costo-mixer.")

    print("\nParámetros QAOA:")
    print(f"  gamma1 = {gamma1:.10f}  -> parámetro del operador de costo C1")
    print(f"  beta1  = {beta1:.10f}  -> parámetro del operador mixer B1")
    print(f"  gamma2 = {gamma2:.10f}  -> parámetro del operador de costo C2")
    print(f"  beta2  = {beta2:.10f}  -> parámetro del operador mixer B2")

    print("\nBloques funcionales / jugadores QSMEF:")
    print("  C1 = primera operación de costo, exp(-i gamma1 H_cost)")
    print("  B1 = primera operación mixer, exp(-i beta1 sum X)")
    print("  C2 = segunda operación de costo, exp(-i gamma2 H_cost)")
    print("  B2 = segunda operación mixer, exp(-i beta2 sum X)")

    print("\nOrden físico:")
    print("  C1 -> B1 -> C2 -> B2")

    print("\nObservable:")
    print("  H_cost = número de aristas cortadas del grafo C6")

    print("-" * 60)


    # --------------------------------------------------------
    # Construcción del caso
    # --------------------------------------------------------

    return QSMEFCase(

        name="QAOA-MaxCut C6 p=2",

        initial_state=psi0,

        operations={
            "C1": C1,
            "B1": B1,
            "C2": C2,
            "B2": B2,
        },

        players=[
            "C1",
            "B1",
            "C2",
            "B2",
        ],

        schedule=[
            "C1",
            "B1",
            "C2",
            "B2",
        ],

        observable=H_cost,

        constraints=(),

        semantic_contract=SemanticContract(
            description=(
                "Cada coalición produce una distribución sobre los mismos "
                "bitstrings del grafo C6 y se evalúa con la misma función "
                "de costo MaxCut."
            ),
            criteria=(
                semantic_interval_criterion(
                    "Rango de cortes del grafo C6",
                    0.0,
                    float(len(edges)),
                    "El número esperado de aristas cortadas debe permanecer "
                    "entre cero y el número de aristas del mismo grafo.",
                ),
            ),
        ),

        property_description=(
            "Número esperado de aristas cortadas"
        ),

        tolerance=1e-10,
    )


print("build_maxcut_case cargado correctamente.")


#########################################################
# ============================================================
# CONSTRUCTOR COMPLETO TWOLOCAL - H2
# CON OPTIMIZACIÓN REPRODUCIBLE DE LOS PARÁMETROS THETA
# ============================================================

"""### 3.3 TwoLocal–H₂

"""


def build_h2_case():

    import numpy as np
    from scipy.optimize import minimize

    # ========================================================
    # DIMENSIÓN DEL SISTEMA
    # ========================================================

    n = 4
    dim = 2 ** n

    # ========================================================
    # MATRICES BÁSICAS
    # ========================================================

    I2_local = np.eye(2, dtype=complex)

    X_local = np.array([
        [0, 1],
        [1, 0],
    ], dtype=complex)

    Y_local = np.array([
        [0, -1j],
        [1j, 0],
    ], dtype=complex)

    Z_local = np.array([
        [1, 0],
        [0, -1],
    ], dtype=complex)

    H_local = (
        np.array([
            [1, 1],
            [1, -1],
        ], dtype=complex)
        / np.sqrt(2)
    )

    # ========================================================
    # PRODUCTO DE KRONECKER
    # ========================================================

    def kron_local(operators):

        result = np.array(
            [[1.0 + 0.0j]]
        )

        for operator in operators:
            result = np.kron(
                result,
                operator
            )

        return result

    # ========================================================
    # ROTACIÓN RY
    # ========================================================

    def ry_local(theta):

        return np.array([
            [
                np.cos(theta / 2),
                -np.sin(theta / 2),
            ],
            [
                np.sin(theta / 2),
                np.cos(theta / 2),
            ],
        ], dtype=complex)

    # ========================================================
    # OPERADOR PAULI SOBRE 4 QUBITS
    # ========================================================

    def pauli_term(mapping):

        operators = []

        for q in range(n):

            if q in mapping:
                operators.append(
                    mapping[q]
                )
            else:
                operators.append(
                    I2_local
                )

        return kron_local(
            operators
        )

    # ========================================================
    # CZ ENTRE DOS QUBITS
    # ========================================================

    def cz_full(a, b):

        diagonal = np.ones(
            dim,
            dtype=complex
        )

        for z in range(dim):

            bits = [
                (
                    z >> (n - 1 - q)
                ) & 1
                for q in range(n)
            ]

            if (
                bits[a] == 1
                and bits[b] == 1
            ):
                diagonal[z] = -1.0

        return np.diag(
            diagonal
        )

    # ========================================================
    # CAPA DE ENTRELAZAMIENTO ALL-TO-ALL CZ
    # ========================================================

    def all_to_all_cz():

        U = np.eye(
            dim,
            dtype=complex
        )

        for a in range(n):

            for b in range(
                a + 1,
                n
            ):
                U = (
                    cz_full(a, b)
                    @ U
                )

        return U

    # ========================================================
    # ESTADO INICIAL HARTREE-FOCK
    #
    # |1100>
    #
    # H2 posee dos electrones.
    # ========================================================

    psi0 = np.zeros(
        dim,
        dtype=complex
    )

    psi0[
        int("1100", 2)
    ] = 1.0

    # ========================================================
# HAMILTONIANO ELECTRÓNICO H2
# JORDAN-WIGNER, 4 QUBITS
#
# Fuente:
# Seeley, J. T., Richard, M. J., & Love, P. J. (2012).
# The Bravyi-Kitaev transformation for quantum computation
# of electronic structure.
# Journal of Chemical Physics, 137, 224109.
#
# Hamiltoniano Jordan-Wigner:
# Ecuaciones (88) y (89).
# ========================================================

    # ========================================================
    # HAMILTONIANO ELECTRÓNICO H2
    # JORDAN-WIGNER, 4 QUBITS
    # ========================================================

    terms = [

        (-0.81261, {}),

        (0.171201, {
            0: Z_local,
        }),

        (0.171201, {
            1: Z_local,
        }),

        (-0.2227965, {
            2: Z_local,
        }),

        (-0.2227965, {
            3: Z_local,
        }),

        (0.16862325, {
            0: Z_local,
            1: Z_local,
        }),

        (0.12054625, {
            0: Z_local,
            2: Z_local,
        }),

        (0.165868, {
            1: Z_local,
            2: Z_local,
        }),

        (0.165868, {
            0: Z_local,
            3: Z_local,
        }),

        (0.12054625, {
            1: Z_local,
            3: Z_local,
        }),

        (0.17434925, {
            2: Z_local,
            3: Z_local,
        }),

        (-0.04532175, {
            0: Y_local,
            1: Y_local,
            2: X_local,
            3: X_local,
        }),

        (0.04532175, {
            0: X_local,
            1: Y_local,
            2: Y_local,
            3: X_local,
        }),

        (0.04532175, {
            0: Y_local,
            1: X_local,
            2: X_local,
            3: Y_local,
        }),

        (-0.04532175, {
            0: X_local,
            1: X_local,
            2: Y_local,
            3: Y_local,
        }),
    ]

    H_h2 = np.zeros(
        (dim, dim),
        dtype=complex
    )

    for coefficient, mapping in terms:
        H_h2 += (
            coefficient
            * pauli_term(mapping)
        )

    # ========================================================
    # CAPAS DE ENTRELAZAMIENTO
    # ========================================================

    E0 = all_to_all_cz()
    E1 = all_to_all_cz()

    # ========================================================
    # CONSTRUCCIÓN DE UNA CAPA DE ROTACIÓN TWOLOCAL
    #
    # Para cada qubit:
    #
    #       Ry(theta_q) H
    #
    # ========================================================

    def build_rotation_layer(theta_layer):

        operators = []

        for q in range(n):

            gate = (
                ry_local(
                    theta_layer[q]
                )
                @ H_local
            )

            operators.append(
                gate
            )

        return kron_local(
            operators
        )

    # ========================================================
    # ESTADO GENERADO POR EL ANSATZ PARA UN VECTOR THETA
    #
    # Arquitectura:
    #
    # R0 -> E0 -> R1 -> E1 -> R2
    #
    # ========================================================

    def state_from_theta(theta):

        R0_test = build_rotation_layer(
            theta[0:4]
        )

        R1_test = build_rotation_layer(
            theta[4:8]
        )

        R2_test = build_rotation_layer(
            theta[8:12]
        )

        psi = (
            R2_test
            @ E1
            @ R1_test
            @ E0
            @ R0_test
            @ psi0
        )

        return psi

    # ========================================================
    # FUNCIÓN OBJETIVO VARIACIONAL
    #
    # E(theta) =
    #
    # <psi(theta)| H_h2 |psi(theta)>
    #
    # ========================================================

    def energy_from_theta(theta):

        psi = state_from_theta(
            theta
        )

        energy = np.vdot(
            psi,
            H_h2 @ psi
        )

        return float(
            np.real(
                energy
            )
        )

    # ========================================================
    # OPTIMIZACIÓN DE LOS PARÁMETROS VARIACIONALES
    #
    # Se parte de theta = 0.
    #
    # theta* = arg min_theta E(theta)
    #
    # ========================================================

    theta_initial = np.zeros(
        12,
        dtype=float
    )

    optimization_result = minimize(
        energy_from_theta,
        theta_initial,
        method="BFGS",
        options={
            "maxiter": 5000,
            "gtol": 1e-8,
        },
    )

    theta = np.asarray(
        optimization_result.x,
        dtype=float
    )

    # ========================================================
    # INFORMACIÓN DE TRAZABILIDAD
    # ========================================================

    print(
        "\nOptimización variacional TwoLocal-H2"
    )

    print(
        "Energía inicial:",
        energy_from_theta(
            theta_initial
        )
    )

    print(
        "Energía optimizada:",
        energy_from_theta(
            theta
        )
    )

    print(
        "Parámetros theta:"
    )

    print(
        theta
    )

    # ========================================================
    # CONSTRUCCIÓN DEFINITIVA DE LOS JUGADORES QSMEF
    # ========================================================

    R0 = build_rotation_layer(
        theta[0:4]
    )

    R1 = build_rotation_layer(
        theta[4:8]
    )

    R2 = build_rotation_layer(
        theta[8:12]
    )

    # ========================================================
    # PROYECTOR P2
    #
    # Subespacio correspondiente a exactamente
    # dos electrones.
    #
    # peso de Hamming = 2
    # ========================================================

    P2 = np.zeros(
        (dim, dim),
        dtype=complex
    )

    for z in range(dim):

        if (
            bin(z).count("1")
            == 2
        ):
            P2[z, z] = 1.0

    # ========================================================
    # RESTRICCIÓN QSMEF
    # ========================================================

    constraint_2e = (
        projector_constraint(

            name=(
                "Sector de dos electrones"
            ),

            projector=P2,

            description=(
                "La configuración parcial debe "
                "permanecer completamente en el "
                "subespacio correspondiente a "
                "exactamente dos electrones."
            ),
        )
    )
    #-----------------------------------------------------
        # ========================================================
    # INFORMACIÓN DESCRIPTIVA DEL CASO TWOLOCAL-H2
    # ========================================================

    print("\nConfiguración del caso TwoLocal-H2")
    print("-" * 65)

    print("Sistema físico: H2")
    print("  H2 = molécula de hidrógeno neutra con 2 electrones.")

    print(f"Número de qubits: {n}")
    print("  Los 4 qubits representan 4 modos fermiónicos mediante Jordan-Wigner.")

    print("Estado inicial:")
    print("  |psi0> = |1100>")
    print("  Representa exactamente 2 electrones ocupando dos modos.")

    print("\nArquitectura variacional TwoLocal:")
    print("  R0 -> E0 -> R1 -> E1 -> R2")
    print("  R0, R1, R2 = capas de rotación.")
    print("  E0, E1 = capas de entrelazamiento CZ all-to-all.")

    print("\nParámetros variacionales:")
    print("  theta = (theta0, ..., theta11)")
    print("  Son 12 ángulos de las puertas Ry.")
    print("  Distribución:")
    print("    R0 <- theta0, theta1, theta2, theta3")
    print("    R1 <- theta4, theta5, theta6, theta7")
    print("    R2 <- theta8, theta9, theta10, theta11")

    print("\nOptimización:")
    print("  Método: BFGS")
    print("  Objetivo: minimizar <psi(theta)| H_H2 |psi(theta)>")
    print("  Los theta optimizados fijan las operaciones concretas R0, R1 y R2.")

    print("\nHamiltoniano:")
    print("  H_H2 = Hamiltoniano electrónico de H2 en representación Jordan-Wigner.")
    print("  Se utiliza como observable del caso.")

    print("\nPropiedad funcional evaluada:")
    print("  Valor esperado de H_H2 = energía electrónica.")

    print("\nBloques funcionales / jugadores QSMEF:")
    print("  R0 = primera capa de rotación")
    print("  E0 = primera capa de entrelazamiento")
    print("  R1 = segunda capa de rotación")
    print("  E1 = segunda capa de entrelazamiento")
    print("  R2 = tercera capa de rotación")

    print("\nRestricción física:")
    print("  P2 = proyector sobre el sector de exactamente 2 electrones.")
    print("  Para cada coalición C debe verificarse:")
    print("  Tr(P2 rho_C) >= 1 - epsilon")

    print("\nContrato semántico:")
    print("  La energía electrónica se compara únicamente entre configuraciones")
    print("  que permanezcan en el mismo sector físico de 2 electrones.")

    print("-" * 65)





    # ========================================================
    # CASO QSMEF
    # ========================================================

    return QSMEFCase(

        name=(
            "TwoLocal-H2 "
            "4 qubits"
        ),

        initial_state=psi0,

        operations={
            "R0": R0,
            "E0": E0,
            "R1": R1,
            "E1": E1,
            "R2": R2,
        },

        players=[
            "R0",
            "E0",
            "R1",
            "E1",
            "R2",
        ],

        schedule=[
            "R0",
            "E0",
            "R1",
            "E1",
            "R2",
        ],

        observable=H_h2,

        constraints=[
            constraint_2e
        ],

        semantic_contract=SemanticContract(

            description=(
                "La energía electrónica solo se compara "
                "dentro del sector físico de exactamente "
                "dos electrones."
            ),

            criteria=(
                semantic_projector_criterion(
                    "Permanencia en el sector de dos electrones",
                    P2,
                    (
                        "Para toda coalición debe cumplirse "
                        "Tr(P2 rho_C) >= 1-epsilon."
                    ),
                ),
            ),
        ),

        property_description=(
            "Energía electrónica de H2 "
            "en representación Jordan-Wigner"
        ),

        tolerance=1e-10,
    )


print(
    "build_h2_case cargado correctamente."
)




# ==========================================================
# ============================================================
# CONSTRUCTOR COMPLETO SKW - HIPERCUBO
# ============================================================
# ============================================================
# CRITERIO SEMÁNTICO DEL CASO SKW
# ============================================================

# ============================================================
# CRITERIO SEMÁNTICO DEL CASO SKW
# ============================================================

"""### 3.4 Criterio del espacio moneda–posición de SKW

"""


def skw_state_space_criterion(
    name: str,
    expected_dim: int,
    description: str = "",
) -> SemanticCriterion:
    """
    Verifica que todas las coaliciones del caso SKW produzcan
    vectores de estado pertenecientes al mismo espacio
    moneda-posicion y que permanezcan normalizados.
    """

    expected_dim = int(expected_dim)

    def evaluator(
        state: Array,
        state_kind: str,
        coalition: frozenset[str],
        expected: complex,
        tolerance: float,
    ) -> tuple[bool, float, str]:

        del coalition, expected

        state = np.asarray(
            state,
            dtype=complex
        )

        # SKW se representa mediante vectores de estado.
        representation_ok = (
            state_kind == "statevector"
            and state.ndim == 1
        )

        # Todas las coaliciones deben permanecer
        # en el mismo espacio moneda-posicion.
        dimension_ok = (
            representation_ok
            and state.shape[0] == expected_dim
        )

        # Verificación de normalización.
        if representation_ok:
            norm = float(
                np.real(
                    np.vdot(
                        state,
                        state
                    )
                )
            )
        else:
            norm = np.nan

        normalized_ok = (
            representation_ok
            and abs(norm - 1.0) <= tolerance
        )

        passed = (
            dimension_ok
            and normalized_ok
        )

        detail = (
            f"dimension="
            f"{state.shape[0] if state.ndim == 1 else 'invalida'}, "
            f"dimension_esperada={expected_dim}, "
            f"norma={norm:.12f}, "
            f"dimension_ok={dimension_ok}, "
            f"normalizacion_ok={normalized_ok}"
        )

        return (
            passed,
            norm,
            detail
        )

    return SemanticCriterion(
        name=name,
        evaluator=evaluator,
        description=description,
    )


# ============================================================
# CONSTRUCTOR COMPLETO SKW - HIPERCUBO
# ============================================================

"""### 3.5 SKW sobre el hipercubo

"""


def build_skw_case(
    n=8,
    marked_vertex=0,
    time_steps=19,
    gamma=1.0,
):
    """
    Construye el caso QSMEF para SKW reproduciendo la definición
    utilizada en el experimento original.

    Primero se evoluciona el paseo completo hasta psi_t.
    Luego QSMEF evalúa una única intervención O -> G -> S
    sobre ese mismo estado psi_t para cada coalición.
    """

    import numpy as np

    N = 2 ** n
    dim = n * N

    # ========================================================
    # ESTADO UNIFORME ORIGINAL
    # ========================================================

    psi_uniform = np.ones(
        dim,
        dtype=complex
    ) / np.sqrt(dim)

    # ========================================================
    # ORÁCULO
    # ========================================================

    def apply_oracle(state):

        amplitudes = state.reshape(
            n,
            N
        ).copy()

        amplitudes[
            :,
            marked_vertex
        ] *= -1.0

        return amplitudes.reshape(-1)

    # ========================================================
    # GROVER COIN
    # ========================================================

    G_coin = (
        2.0
        * np.ones(
            (n, n),
            dtype=complex
        )
        / n
        - np.eye(
            n,
            dtype=complex
        )
    )

    def apply_grover(state):

        amplitudes = state.reshape(
            n,
            N
        )

        result = (
            G_coin
            @ amplitudes
        )

        return result.reshape(-1)

    # ========================================================
    # SHIFT CONDICIONAL
    # ========================================================

    def apply_shift(state):

        amplitudes = state.reshape(
            n,
            N
        )

        output = np.zeros_like(
            amplitudes
        )

        for c in range(n):

            bit_mask = (
                1
                << c
            )

            for x in range(N):

                destination = (
                    x ^ bit_mask
                )

                output[
                    c,
                    destination
                ] = amplitudes[
                    c,
                    x
                ]

        return output.reshape(-1)

    # ========================================================
    # CONSTRUCCIÓN DEL ESTADO psi_t
    # ========================================================

    psi_t = psi_uniform.copy()

    for _ in range(time_steps):

        psi_t = apply_oracle(
            psi_t
        )

        psi_t = apply_grover(
            psi_t
        )

        psi_t = apply_shift(
            psi_t
        )

    # ========================================================
    # OBSERVABLE ENERGÉTICO
    #
    # H = I_C ⊗ (-gamma A_P - |w><w|)
    # ========================================================

    def skw_observable(
        state,
        state_kind=None
    ):

        amplitudes = state.reshape(
            n,
            N
        )

        adjacency_expectation = (
            0.0 + 0.0j
        )

        for coin in range(n):

            vector = amplitudes[
                coin
            ]

            for bit in range(n):

                mask = (
                    1
                    << bit
                )

                permuted = np.array([
                    x ^ mask
                    for x in range(N)
                ])

                adjacency_expectation += (
                    np.vdot(
                        vector,
                        vector[permuted]
                    )
                )

        marked_probability = np.sum(
            np.abs(
                amplitudes[
                    :,
                    marked_vertex
                ]
            ) ** 2
        )

        value = (
            -gamma
            * adjacency_expectation
            - marked_probability
        )

        return float(
            np.real_if_close(
                value
            )
        )

    # ========================================================
    # OPERACIONES / JUGADORES QSMEF
    # ========================================================

    O = OperationSpec(
        action=apply_oracle,
        declared_unitary=True,
        unitarity_rationale=(
            "El oráculo es una reflexión de fase "
            "sobre el vértice marcado."
        ),
    )

    G = OperationSpec(
        action=apply_grover,
        declared_unitary=True,
        unitarity_rationale=(
            "La moneda de Grover es un operador unitario "
            "sobre el espacio de moneda."
        ),
    )

    S = OperationSpec(
        action=apply_shift,
        declared_unitary=True,
        unitarity_rationale=(
            "El shift es una permutación reversible "
            "sobre los estados moneda-posicion."
        ),
    )

    # ========================================================
    # SECUENCIA DEL JUEGO QSMEF
    #
    # Cada coalición actúa UNA SOLA VEZ sobre psi_t.
    # ========================================================

    schedule = [
        "O",
        "G",
        "S",
    ]

    # ========================================================
    # INFORMACIÓN DESCRIPTIVA
    # ========================================================

    print()
    print("Configuración del caso SKW - Hipercubo")
    print("-" * 72)

    print(f"n = {n}")
    print(f"N = {N}")
    print(f"Vértice marcado: {marked_vertex}")
    print(f"Instante evaluado t = {time_steps}")
    print(f"gamma = {gamma}")

    print()
    print("Espacio de estados:")
    print(f"  Espacio de moneda: dimensión {n}")
    print(f"  Espacio de posición: dimensión {N}")
    print(f"  Dimensión total: {dim} = n * N")

    print()
    print("Preparación del estado:")
    print("  Se parte de la superposición uniforme.")
    print(
        f"  Se ejecutan {time_steps} pasos SKW completos O -> G -> S."
    )
    print(
        f"  QSMEF recibe como estado inicial del juego el estado psi_{time_steps}."
    )

    print()
    print("Jugadores QSMEF:")
    print("  {O, G, S}")

    print()
    print("Secuencia evaluada por QSMEF:")
    print("  O -> G -> S")
    print("  El schedule contiene 3 operaciones.")
    print("  Cada coalición se aplica una sola vez sobre psi_t.")

    print()
    print("Observable energético:")
    print("  H = I_C ⊗ (-gamma A_P - |w><w|)")

    print()
    print("Juego característico:")
    print("  v_t(C) = M_H(U_C psi_t) - M_H(psi_t)")

    print()
    print("Coaliciones QSMEF:")
    print(
        "  {}, {O}, {G}, {S}, {O,G}, {O,S}, {G,S}, {O,G,S}"
    )

    print("-" * 72)

    # ========================================================
    # CONSTRUCCIÓN DEL CASO QSMEF
    # ========================================================

    return QSMEFCase(

        name=(
            f"SKW hipercubo "
            f"n={n}, t={time_steps}"
        ),

        initial_state=psi_t,

        operations={
            "O": O,
            "G": G,
            "S": S,
        },

        players=[
            "O",
            "G",
            "S",
        ],

        schedule=schedule,

        observable=skw_observable,

        constraints=(),

        semantic_contract=SemanticContract(

            description=(
                "Todas las coaliciones del caso SKW actúan "
                "sobre el mismo estado psi_t y deben permanecer "
                "en el mismo espacio moneda-posicion. "
                "Además, deben producir estados normalizados "
                "y ser evaluadas mediante el mismo observable "
                "energético. El criterio de intervalo se utiliza "
                "como verificación numérica complementaria "
                "del observable."
            ),

            criteria=(

                skw_state_space_criterion(
                    name=(
                        "Conservación del espacio moneda-posicion"
                    ),
                    expected_dim=dim,
                    description=(
                        "Verifica que cada coalición produzca "
                        "un vector de estado normalizado "
                        "perteneciente al mismo espacio "
                        "moneda-posicion del paseo SKW."
                    ),
                ),

                semantic_interval_criterion(
                    name=(
                        "Rango espectral del observable SKW"
                    ),
                    lower=-float(
                        abs(gamma) * n + 1
                    ),
                    upper=float(
                        abs(gamma) * n + 1
                    ),
                    description=(
                        "Verifica que el valor esperado del "
                        "observable permanezca dentro de una "
                        "cota compatible con "
                        "-gamma*A-|w><w| en el hipercubo "
                        "n-dimensional."
                    ),
                ),
            ),
        ),

        property_description=(
            "Variación del valor esperado del observable energético "
            f"en el paso funcional evaluado desde psi_{time_steps}"
        ),

        tolerance=1e-10,
    )


print(
    "build_skw_case cargado correctamente."
)

# ============================================================
# CONSTRUCTOR COMPLETO QPE - SHOR
# ============================================================

"""### 3.6 QPE de Shor

"""


def build_qpe_shor_case(
    N=21,
    a=2,
    counting_qubits=5,
):

    import numpy as np

    m = counting_qubits

    # Cantidad de qubits necesarios para representar 0,...,N-1
    work_qubits = int(
        np.ceil(
            np.log2(N)
        )
    )

    work_dim = 2 ** work_qubits

    total_qubits = (
        m + work_qubits
    )

    dim = 2 ** total_qubits


    # ========================================================
    # ESTADO INICIAL
    #
    # |0...0> |1>
    # ========================================================

    psi0_qpe = np.zeros(
        dim,
        dtype=complex
    )

    psi0_qpe[1] = 1.0


    # ========================================================
    # MATRIZ DE HADAMARD
    # ========================================================

    H1_local = (
        np.array([
            [1, 1],
            [1, -1]
        ], dtype=complex)
        / np.sqrt(2)
    )

    I2_local = np.eye(
        2,
        dtype=complex
    )


    def kron_local(operators):

        result = np.array(
            [[1.0 + 0.0j]]
        )

        for operator in operators:
            result = np.kron(
                result,
                operator
            )

        return result


    # ========================================================
    # PREPARACIÓN DEL REGISTRO DE CONTEO
    #
    # H^⊗m ⊗ I
    # ========================================================

    Hprep_matrix = kron_local(
        [H1_local] * m
        +
        [I2_local] * work_qubits
    )


    def apply_hprep(state):

        return (
            Hprep_matrix
            @ state
        )


    # ========================================================
    # MULTIPLICACIÓN MODULAR CONTROLADA
    #
    # |y> -> |a^(2^k) y mod N>
    #
    # cuando el qubit de control k vale 1.
    # ========================================================

    def make_controlled_modular_multiplication(k):

        multiplier = pow(
            a,
            2 ** k,
            N
        )

        # q0 = qubit más significativo
        control_mask = (
            1
            << (m - 1 - k)
        )


        def apply_controlled_modular_multiplication(
            state
        ):

            output = np.zeros_like(
                state
            )


            for index, amplitude in enumerate(
                state
            ):

                counting_value = (
                    index
                    >> work_qubits
                )

                work_value = (
                    index
                    & (work_dim - 1)
                )


                control_is_one = (
                    counting_value
                    & control_mask
                ) != 0


                if (
                    control_is_one
                    and work_value < N
                ):

                    new_work_value = (
                        multiplier
                        * work_value
                    ) % N

                else:

                    new_work_value = (
                        work_value
                    )


                new_index = (
                    counting_value
                    << work_qubits
                ) | new_work_value


                output[
                    new_index
                ] += amplitude


            return output


        return (
            apply_controlled_modular_multiplication
        )


    # ========================================================
    # QFT INVERSA SOBRE EL REGISTRO DE CONTEO
    # ========================================================

    size_count = 2 ** m

    row = np.arange(
        size_count
    )[:, None]

    column = np.arange(
        size_count
    )[None, :]


    IQFT_matrix = (
        np.exp(
            -2j
            * np.pi
            * row
            * column
            / size_count
        )
        / np.sqrt(size_count)
    )


    def apply_iqft(state):

        amplitudes = state.reshape(
            size_count,
            work_dim
        )

        result = (
            IQFT_matrix
            @ amplitudes
        )

        return result.reshape(-1)


    # ========================================================
    # OBSERVABLE FUNCIONAL
    #
    # Probabilidad de obtener un valor distinto de cero
    # en el registro de conteo.
    #
    # Se utiliza únicamente como magnitud común de evaluación.
    # ========================================================

    def qpe_observable(
        state,
        state_kind=None
    ):

        amplitudes = state.reshape(
            size_count,
            work_dim
        )

        probability_zero = np.sum(
            np.abs(
                amplitudes[0]
            ) ** 2
        )

        return float(
            1.0
            - probability_zero
        )


    # ========================================================
    # OPERACIONES FIJAS
    # ========================================================

    operations = {

        "Hprep": OperationSpec(

            action=apply_hprep,

            declared_unitary=True,

            unitarity_rationale=(
                "La preparación aplica Hadamard "
                "al registro de conteo."
            ),
        ),

        "IQFT": OperationSpec(

            action=apply_iqft,

            declared_unitary=True,

            unitarity_rationale=(
                "La transformada inversa de Fourier "
                "es unitaria."
            ),
        ),
    }


    # ========================================================
    # JUGADORES:
    # U0, U1, ..., U(m-1)
    # ========================================================

    players = []


    for k in range(m):

        operation_name = (
            f"U{k}"
        )

        operations[
            operation_name
        ] = OperationSpec(

            action=(
                make_controlled_modular_multiplication(
                    k
                )
            ),

            declared_unitary=True,

            unitarity_rationale=(
                "Multiplicación modular controlada "
                f"por a^(2^{k}) mod N."
            ),
        )

        players.append(
            operation_name
        )


    # ========================================================
    # ORDEN DEL CIRCUITO
    #
    # Hprep permanece fijo
    # U0...Um-1 son jugadores
    # IQFT permanece fija
    # ========================================================

    schedule = (
        ["Hprep"]
        +
        players
        +
        ["IQFT"]
    )


    # ========================================================
    # CASO QSMEF
    #
    # IMPORTANTE:
    #
    # El rechazo es SEMÁNTICO.
    # El circuito puede ser matemáticamente válido, pero
    # las coaliciones parciales no mantienen necesariamente
    # la misma interpretación de "estimación de fase".
    # ========================================================

    def complete_qpe_instance(
        state,
        state_kind,
        coalition,
        expected,
        tolerance,
    ):
        del state, state_kind, expected, tolerance
        complete = coalition == frozenset(players)
        value = float(len(coalition))
        detail = (
            f"potencias_controladas_presentes={len(coalition)}/{len(players)}"
        )
        return complete, value, detail

    qpe_semantic_criterion = SemanticCriterion(
        name="Instancia completa de estimación de fase",
        evaluator=complete_qpe_instance,
        description=(
            "Una estimación de la fase de la evolución declarada requiere "
            "el conjunto completo de potencias controladas."
        ),
    )

   #######################################################################
    # ========================================================
    # INFORMACIÓN DESCRIPTIVA DEL CASO QPE-SHOR
    # ========================================================

    print("\nConfiguración del caso QPE-Shor")
    print("-" * 70)

    print(f"N = {N}")
    print("  N = número entero que se desea factorizar mediante Shor.")

    print(f"a = {a}")
    print("  a = base elegida para estudiar el orden modular respecto de N.")
    print("  Se utilizan potencias de a módulo N dentro de QPE.")

    print(f"m = {m}")
    print("  m = cantidad de qubits del registro de conteo.")
    print("  Estos qubits se utilizan para obtener información sobre la fase.")

    print(f"Qubits del registro de trabajo: {work_qubits}")
    print("  Son los qubits necesarios para representar los valores 0,...,N-1.")

    print(f"Qubits totales: {total_qubits}")
    print("  total_qubits = qubits de conteo + qubits de trabajo.")

    print(f"Dimensión total del espacio de estados: {dim}")
    print("  dim = 2^(total_qubits).")

    print("\nEstado inicial:")
    print("  |psi0> = |0...0>|1>")
    print("  El registro de conteo comienza en cero.")
    print("  El registro de trabajo comienza en el estado |1>.")

    print("\nOperaciones fijas:")
    print("  Hprep = Hadamard sobre todos los qubits del registro de conteo.")
    print("  IQFT  = Transformada Cuántica de Fourier inversa sobre el registro de conteo.")

    print("\nJugadores QSMEF:")

    for k in range(m):

        multiplier = pow(
            a,
            2 ** k,
            N
        )

        print(
            f"  U{k} = multiplicación modular controlada "
            f"por a^(2^{k}) mod N = {multiplier}"
        )

    print("\nInterpretación de los jugadores:")
    print("  Cada Uk representa una potencia controlada de la operación modular.")
    print("  Para QPE completo se utilizan todas las potencias controladas.")

    print("\nOrden del circuito:")
    print(
        "  Hprep -> "
        + " -> ".join(players)
        + " -> IQFT"
    )

    print("\nObservable funcional:")
    print("  Se calcula la probabilidad de obtener un valor distinto de cero")
    print("  en el registro de conteo.")

    print("\nContrato semántico:")
    print("  Una coalición se considera semánticamente comparable únicamente")
    print("  si contiene el conjunto completo de operaciones controladas.")
    print("  Las coaliciones parciales no representan necesariamente")
    print("  la misma instancia completa de estimación de fase.")

    print("-" * 70)
   # #########################################################



    return QSMEFCase(

        name=(
            f"QPE-Shor "
            f"N={N}, a={a}, m={m}"
        ),

        initial_state=psi0_qpe,

        operations=operations,

        players=players,

        schedule=schedule,

        observable=qpe_observable,

        constraints=(),

        semantic_contract=SemanticContract(

            description=(
                "Las coaliciones solo serían comparables si conservaran "
                "una instancia completa de la misma estimación de fase."
            ),

            criteria=(qpe_semantic_criterion,),
        ),

        property_description=(
            "Información obtenida en el "
            "registro de estimación de fase"
        ),

        tolerance=1e-10,
    )


print(
    "build_qpe_shor_case cargado correctamente."
)

########################################################################
#clasificador cuántico circuit-centric de 3 qubits
#
#------------------------------------------------------------------------

"""### 3.7 Clasificador cuántico

"""


def build_classifier_case():

    import numpy as np

    # ========================================================
    # MATRICES BÁSICAS
    # ========================================================

    I2 = np.eye(2, dtype=complex)

    Z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)


    # ========================================================
    # KRONECKER
    # ========================================================

    def kron_all(operators):

        result = np.array([[1.0 + 0.0j]])

        for op in operators:
            result = np.kron(
                result,
                op
            )

        return result


    # ========================================================
    # PUERTA G(alpha,beta,gamma)
    #
    # Basada en Schuld et al., Ec. (6)
    # ========================================================

    def G(alpha, beta, gamma):

        return np.array([

            [
                np.exp(1j * beta) * np.cos(alpha),
                np.exp(1j * gamma) * np.sin(alpha)
            ],

            [
                -np.exp(-1j * gamma) * np.sin(alpha),
                np.exp(-1j * beta) * np.cos(alpha)
            ]

        ], dtype=complex)


    # ========================================================
    # PUERTA DE UN QUBIT SOBRE REGISTRO DE n QUBITS
    # ========================================================

    def single_qubit_full(
        gate,
        target,
        n
    ):

        operators = []

        for q in range(n):

            if q == target:
                operators.append(gate)

            else:
                operators.append(I2)

        return kron_all(
            operators
        )


    # ========================================================
    # PUERTA CONTROLADA
    # ========================================================

    def controlled_gate_full(
        gate,
        control,
        target,
        n
    ):

        dim = 2 ** n

        U = np.zeros(
            (dim, dim),
            dtype=complex
        )


        for basis in range(dim):

            bits = [
                (
                    basis
                    >> (n - 1 - q)
                ) & 1

                for q in range(n)
            ]


            # Si el control está en 0:
            # identidad
            if bits[control] == 0:

                U[
                    basis,
                    basis
                ] = 1.0

                continue


            input_target = (
                bits[target]
            )


            for output_target in (
                0,
                1
            ):

                new_bits = (
                    bits.copy()
                )

                new_bits[target] = (
                    output_target
                )


                new_index = 0

                for bit in new_bits:

                    new_index = (
                        new_index << 1
                    ) | bit


                U[
                    new_index,
                    basis
                ] += gate[
                    output_target,
                    input_target
                ]


        return U


    # ========================================================
    # BLOQUE DEL CLASIFICADOR
    #
    # capa local G
    # +
    # capa controlada circular
    # ========================================================

    def classifier_block(
        single_params,
        controlled_params,
        n=3
    ):

        U = np.eye(
            2 ** n,
            dtype=complex
        )


        # ----------------------------------------------------
        # CAPA LOCAL
        # ----------------------------------------------------

        for q in range(n):

            alpha, beta, gamma = (
                single_params[q]
            )

            Uq = single_qubit_full(
                G(
                    alpha,
                    beta,
                    gamma
                ),
                target=q,
                n=n
            )

            U = Uq @ U


        # ----------------------------------------------------
        # CAPA CONTROLADA
        #
        # q0 -> q1
        # q1 -> q2
        # q2 -> q0
        # ----------------------------------------------------

        connections = [
            (0, 1),
            (1, 2),
            (2, 0),
        ]


        for (
            control,
            target
        ), params in zip(
            connections,
            controlled_params
        ):

            alpha, beta, gamma = params

            Uc = controlled_gate_full(
                G(
                    alpha,
                    beta,
                    gamma
                ),
                control=control,
                target=target,
                n=n
            )

            U = Uc @ U


        return U


    # ========================================================
    # DATO DE ENTRADA
    #
    # amplitude encoding
    # ========================================================

    x = np.array([
        0.40,
        0.10,
        0.30,
        0.20,
        0.50,
        0.35,
        0.15,
        0.45,
    ], dtype=float)


    x = (
        x
        / np.linalg.norm(x)
    )


    psi0_classifier = (
        x.astype(complex)
    )


    # ========================================================
    # PARÁMETROS DEL BLOQUE 1
    # ========================================================

    block1_single = [

        (
            0.31,
            0.17,
            -0.21
        ),

        (
            0.52,
            -0.14,
            0.33
        ),

        (
            0.27,
            0.41,
            0.18
        ),
    ]


    block1_controlled = [

        (
            0.22,
            0.13,
            -0.19
        ),

        (
            0.36,
            -0.25,
            0.28
        ),

        (
            0.19,
            0.32,
            0.11
        ),
    ]


    # ========================================================
    # PARÁMETROS DEL BLOQUE 2
    # ========================================================

    block2_single = [

        (
            0.44,
            -0.12,
            0.26
        ),

        (
            0.29,
            0.37,
            -0.16
        ),

        (
            0.51,
            0.08,
            0.24
        ),
    ]


    block2_controlled = [

        (
            0.33,
            -0.20,
            0.17
        ),

        (
            0.41,
            0.15,
            -0.23
        ),

        (
            0.25,
            0.28,
            0.31
        ),
    ]


    # ========================================================
    # DOS JUGADORES QSMEF
    # ========================================================

    B1 = classifier_block(
        block1_single,
        block1_controlled,
        n=3
    )


    B2 = classifier_block(
        block2_single,
        block2_controlled,
        n=3
    )


    # ========================================================
    # OBSERVABLE
    #
    # Z sobre el primer qubit
    # ========================================================

    H_classifier = kron_all([
        Z,
        I2,
        I2
    ])
    #####--------------------------------------------------------
        # ========================================================
    # INFORMACIÓN DESCRIPTIVA DEL CASO CLASSIFIER
    # ========================================================

    print("\nConfiguración del caso Circuit-Centric Quantum Classifier")
    print("-" * 72)

    print("Tipo de caso:")
    print("  Clasificador cuántico circuit-centric de 3 qubits.")

    print("\nNúmero de qubits: 3")
    print("  Con 3 qubits se dispone de un espacio de dimensión 2^3 = 8.")

    print("\nDato de entrada:")
    print("  x = vector clásico de 8 componentes.")
    print("  El vector se normaliza y se codifica mediante amplitude encoding.")
    print("  Por lo tanto, sus 8 componentes se convierten en las amplitudes")
    print("  del estado cuántico inicial.")

    print("\nEstado inicial:")
    print("  psi0_classifier = estado cuántico obtenido a partir de x.")
    print("  Todas las coaliciones QSMEF parten del mismo dato codificado.")

    print("\nPuerta parametrizada G(alpha, beta, gamma):")
    print("  G = operación unitaria de un qubit utilizada en el clasificador.")
    print("  alpha, beta y gamma son los tres parámetros angulares de G.")
    print("  Estos parámetros determinan la transformación aplicada al qubit.")

    print("\nParámetros del bloque B1:")
    print("  block1_single = parámetros de las puertas G locales de B1.")
    print("  block1_controlled = parámetros de las puertas G controladas de B1.")

    print("\nParámetros del bloque B2:")
    print("  block2_single = parámetros de las puertas G locales de B2.")
    print("  block2_controlled = parámetros de las puertas G controladas de B2.")

    print("\nImportante sobre los parámetros:")
    print("  Los valores alpha, beta y gamma están fijados en esta instancia.")
    print("  Este constructor no realiza entrenamiento ni optimización.")

    print("\nEstructura interna de cada bloque:")
    print("  1. Capa local: una puerta G sobre cada uno de los 3 qubits.")
    print("  2. Capa controlada circular:")
    print("       q0 -> q1")
    print("       q1 -> q2")
    print("       q2 -> q0")

    print("\nBloques funcionales / jugadores QSMEF:")
    print("  B1 = primer bloque del clasificador.")
    print("  B2 = segundo bloque del clasificador.")

    print("\nOrden del circuito:")
    print("  |psi0> -> B1 -> B2 -> medición de Z")

    print("\nObservable:")
    print("  H_classifier = Z ⊗ I ⊗ I")
    print("  Z actúa sobre el primer qubit.")
    print("  I representa la identidad sobre los otros dos qubits.")

    print("\nPropiedad funcional evaluada:")
    print("  <Z> = valor esperado de Z sobre el primer qubit.")
    print("  Su rango posible es [-1, 1].")

    print("\nContrato semántico:")
    print("  Todas las coaliciones parten del mismo dato de entrada")
    print("  y son evaluadas mediante el mismo observable Z.")
    print("  Se verifica que el valor esperado permanezca en [-1, 1].")

    print("\nCoaliciones QSMEF:")
    print("  Con 2 jugadores existen 2^2 = 4 coaliciones:")
    print("  {}, {B1}, {B2}, {B1,B2}")

    print("-" * 72)



    # ========================================================
    # CASO QSMEF
    # ========================================================

    return QSMEFCase(

        name=(
            "Circuit-centric quantum classifier"
        ),

        initial_state=psi0_classifier,

        operations={
            "B1": B1,
            "B2": B2,
        },

        players=[
            "B1",
            "B2",
        ],

        schedule=[
            "B1",
            "B2",
        ],

        observable=H_classifier,

        constraints=(),

        semantic_contract=SemanticContract(

            description=(
                "Todas las coaliciones parten del mismo dato codificado y "
                "la salida conserva la interpretación del observable Z."
            ),

            criteria=(
                semantic_interval_criterion(
                    "Rango de la salida Z",
                    -1.0,
                    1.0,
                    "El valor esperado de Z debe permanecer en su rango espectral.",
                ),
            ),
        ),

        property_description=(
            "Valor esperado de Z sobre "
            "el qubit de salida"
        ),

        tolerance=1e-10,
    )


print(
    "build_classifier_case cargado correctamente."
)

"""## 4. Cálculo de las contribuciones de Shapley

"""


# ============================================================
# 4. CÁLCULO DE CONTRIBUCIONES DE SHAPLEY
# ============================================================

def compute_shapley_values(
    case: QSMEFCase,
    validation_result: dict[str, Any],
    show_table: bool = True,
) -> dict[str, Any]:
    """
    Calcula las contribuciones de Shapley para un caso QSMEF.

    El cálculo se realiza únicamente cuando el caso fue clasificado
    como APLICABLE.

    Utiliza los valores v(C) ya obtenidos durante la validación:

        v(C) = M_H(rho_C) - M_H(rho_empty)

    Para cada jugador i:

        phi_i =
            sum_{S subseteq B \\ {i}}
            |S|! (|B|-|S|-1)! / |B|!
            * [v(S union {i}) - v(S)]
    """

    # --------------------------------------------------------
    # Verificación de aplicabilidad
    # --------------------------------------------------------

    if validation_result["verdict"] != "APLICABLE":

        print("\n" + "-" * 76)
        print("CONTRIBUCIONES DE SHAPLEY")
        print("-" * 76)

        print(
            "No se calculan las contribuciones porque "
            "el caso no fue clasificado como APLICABLE."
        )

        return {
            "computed": False,
            "reason": (
                "El caso no satisface todas las condiciones "
                "de aplicabilidad de QSMEF."
            ),
            "shapley_values": {},
            "table": pd.DataFrame(),
            "efficiency_ok": None,
            "efficiency_residual": None,
        }


    # --------------------------------------------------------
    # Datos del caso
    # --------------------------------------------------------

    players = list(case.players)

    n = len(players)

    coalitions_df = validation_result[
        "coalitions"
    ].copy()


    # --------------------------------------------------------
    # Conversión de etiquetas de coalición a conjuntos
    # --------------------------------------------------------

    coalition_values = {}

    for _, row in coalitions_df.iterrows():

        label = row["coalición"]

        if label == "∅":
            coalition = frozenset()

        else:
            content = (
                label
                .replace("{", "")
                .replace("}", "")
                .strip()
            )

            coalition = frozenset(
                item.strip()
                for item in content.split(",")
                if item.strip()
            )

        coalition_values[
            coalition
        ] = float(
            row["v(C)"]
        )


    # --------------------------------------------------------
    # Fórmula de Shapley
    # --------------------------------------------------------

    shapley_values = {}

    for player in players:

        phi = 0.0

        other_players = [
            p
            for p in players
            if p != player
        ]

        for size in range(
            len(other_players) + 1
        ):

            for subset_tuple in itertools.combinations(
                other_players,
                size,
            ):

                S = frozenset(
                    subset_tuple
                )

                S_with_player = frozenset(
                    set(S) | {player}
                )

                weight = (
                    math.factorial(len(S))
                    * math.factorial(
                        n - len(S) - 1
                    )
                    / math.factorial(n)
                )

                marginal_contribution = (
                    coalition_values[
                        S_with_player
                    ]
                    - coalition_values[
                        S
                    ]
                )

                phi += (
                    weight
                    * marginal_contribution
                )

        shapley_values[
            player
        ] = float(phi)


    # --------------------------------------------------------
    # Verificación del axioma de eficiencia
    # --------------------------------------------------------

    grand_coalition = frozenset(
        players
    )

    v_grand = coalition_values[
        grand_coalition
    ]

    shapley_sum = sum(
        shapley_values.values()
    )

    efficiency_residual = abs(
        shapley_sum - v_grand
    )

    efficiency_ok = (
        efficiency_residual
        <= case.tolerance
    )


    # --------------------------------------------------------
    # Tabla de resultados
    # --------------------------------------------------------

    shapley_df = pd.DataFrame(
        [
            {
                "jugador": player,
                "phi": shapley_values[
                    player
                ],
            }
            for player in players
        ]
    )


    # --------------------------------------------------------
    # Salida
    # --------------------------------------------------------

    print("\n" + "-" * 76)
    print("CONTRIBUCIONES DE SHAPLEY")
    print("-" * 76)

    print(
        f"v(B) = {v_grand:.12f}"
    )

    print(
        f"Σ phi_i = {shapley_sum:.12f}"
    )

    print(
        "Residual de eficiencia = "
        f"{efficiency_residual:.3e}"
    )

    print(
        "Eficiencia: "
        + (
            "OK"
            if efficiency_ok
            else "NO CUMPLIDA"
        )
    )

    if show_table:

        display(
            shapley_df
        )


    return {
        "computed": True,
        "shapley_values":
            shapley_values,
        "table":
            shapley_df,
        "v_grand":
            v_grand,
        "shapley_sum":
            shapley_sum,
        "efficiency_ok":
            efficiency_ok,
        "efficiency_residual":
            efficiency_residual,
    }

"""## 5. Asistente semiautomático de granularidad

El constructor aporta los bloques atómicos. El asistente genera particiones contiguas, conserva el orden y las operaciones fijas, aplica el presupuesto $2^L$ y consulta el validador semántico para ordenar únicamente candidatas compatibles con el alcance declarado.

"""

"""Asistente semiautomático y explicable de granularidad para QSMEF.

Se carga DESPUÉS de las celdas que definen OperationSpec, SemanticCriterion,
SemanticContract, QSMEFCase y validate_qsmef_case.

El asistente no inventa la pregunta científica. A partir de una pregunta,
metadatos funcionales y un presupuesto de coaliciones:

1. infiere qué roles deben distinguirse;
2. genera particiones contiguas del circuito;
3. descarta particiones incompatibles con la pregunta o el presupuesto;
4. construye casos QSMEF agrupados sin alterar el orden;
5. ejecuta el validador QSMEF existente;
6. devuelve un ranking y la justificación de cada recomendación.
"""


from dataclasses import dataclass, field
from itertools import product
from math import log2
from typing import Any, Mapping, Optional, Sequence
import contextlib
import io
import re
import unicodedata

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlockMetadata:
    """Descripción funcional de un jugador atómico ya identificado."""

    name: str
    role: str
    description: str = ""
    keywords: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class GranularityObjective:
    """Alcance declarado para seleccionar la granularidad."""

    question: str
    distinguish_roles: Sequence[str] = field(default_factory=tuple)
    required_roles: Sequence[str] = field(default_factory=tuple)
    max_coalitions: int = 1024
    max_candidates: int = 4096
    validate_semantics: bool = True


@dataclass(frozen=True)
class GranularityCandidate:
    """Partición candidata expresada como grupos de jugadores atómicos."""

    groups: tuple[tuple[str, ...], ...]

    @property
    def player_count(self) -> int:
        return len(self.groups)

    @property
    def coalition_count(self) -> int:
        return 2 ** self.player_count

    @property
    def label(self) -> str:
        return " | ".join("+".join(group) for group in self.groups)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def infer_distinguished_roles(
    question: str,
    metadata: Mapping[str, BlockMetadata],
) -> tuple[str, ...]:
    """Infiere roles citados en la pregunta mediante metadatos explicables.

    No usa un modelo opaco: un rol se selecciona solamente si la pregunta
    contiene su nombre o alguna de sus palabras clave declaradas.
    """

    normalized_question = f" {_normalize_text(question)} "
    detected: list[str] = []
    for item in metadata.values():
        terms = (item.role, *item.keywords)
        if any(
            f" {_normalize_text(term)} " in normalized_question
            for term in terms
            if _normalize_text(term)
        ):
            if item.role not in detected:
                detected.append(item.role)
    return tuple(detected)


def objective_from_question(
    question: str,
    metadata: Mapping[str, BlockMetadata],
    *,
    max_coalitions: int = 1024,
    required_roles: Sequence[str] = (),
    distinguish_roles: Optional[Sequence[str]] = None,
    max_candidates: int = 4096,
    validate_semantics: bool = True,
) -> GranularityObjective:
    """Construye el objetivo e infiere roles cuando no se declaran."""

    inferred = infer_distinguished_roles(question, metadata)
    roles = tuple(distinguish_roles) if distinguish_roles is not None else inferred
    return GranularityObjective(
        question=question,
        distinguish_roles=roles,
        required_roles=tuple(required_roles),
        max_coalitions=max_coalitions,
        max_candidates=max_candidates,
        validate_semantics=validate_semantics,
    )


def _ordered_players(case: Any) -> tuple[str, ...]:
    """Obtiene jugadores en orden y exige una aparición por jugador."""

    players = tuple(str(p) for p in case.players)
    schedule = tuple(str(op) for op in case.schedule)
    missing = [p for p in players if p not in schedule]
    repeated = [p for p in players if schedule.count(p) != 1]
    if missing:
        raise ValueError(f"Jugadores ausentes del schedule: {missing}")
    if repeated:
        raise ValueError(
            "La generación automática requiere una aparición por jugador. "
            f"Jugadores problemáticos: {repeated}. Despliegue primero las "
            "apariciones como bloques atómicos diferentes."
        )
    return tuple(op for op in schedule if op in set(players))


def _can_join_in_schedule(case: Any, left: str, right: str) -> bool:
    """Solo fusiona jugadores adyacentes; nunca salta una operación fija."""

    schedule = tuple(str(op) for op in case.schedule)
    return schedule.index(right) == schedule.index(left) + 1


def generate_contiguous_candidates(
    case: Any,
    *,
    max_candidates: int = 4096,
) -> list[GranularityCandidate]:
    """Genera particiones contiguas, desde la más fina hasta la más gruesa."""

    ordered = _ordered_players(case)
    if not ordered:
        raise ValueError("El caso no contiene jugadores.")

    optional_boundaries = [
        i for i in range(len(ordered) - 1)
        if _can_join_in_schedule(case, ordered[i], ordered[i + 1])
    ]
    forced_boundaries = set(range(len(ordered) - 1)) - set(optional_boundaries)
    total = 2 ** len(optional_boundaries)
    if total > max_candidates:
        raise ValueError(
            f"Se generarían {total} particiones. Aumente max_candidates o "
            "declare previamente bloques de mayor nivel."
        )

    candidates: list[GranularityCandidate] = []
    for choices in product((False, True), repeat=len(optional_boundaries)):
        cuts = set(forced_boundaries)
        cuts.update(
            boundary
            for boundary, cut in zip(optional_boundaries, choices)
            if cut
        )
        groups: list[tuple[str, ...]] = []
        start = 0
        for i in range(len(ordered) - 1):
            if i in cuts:
                groups.append(tuple(ordered[start:i + 1]))
                start = i + 1
        groups.append(tuple(ordered[start:]))
        candidates.append(GranularityCandidate(tuple(groups)))

    return sorted(candidates, key=lambda c: (-c.player_count, c.label))


def _roles_of_group(
    group: Sequence[str], metadata: Mapping[str, BlockMetadata]
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(metadata[name].role for name in group))


def _candidate_filters(
    candidate: GranularityCandidate,
    metadata: Mapping[str, BlockMetadata],
    objective: GranularityObjective,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.coalition_count > objective.max_coalitions:
        reasons.append(
            f"costo {candidate.coalition_count} > presupuesto "
            f"{objective.max_coalitions}"
        )

    available_roles = {
        metadata[name].role for group in candidate.groups for name in group
    }
    missing = set(objective.required_roles) - available_roles
    if missing:
        reasons.append(f"faltan roles requeridos: {sorted(missing)}")

    separated = set(objective.distinguish_roles)
    for group in candidate.groups:
        mixed = set(_roles_of_group(group, metadata)) & separated
        if len(mixed) > 1:
            reasons.append(
                f"el grupo {'+'.join(group)} mezcla roles que la pregunta "
                f"debe distinguir: {sorted(mixed)}"
            )
    return not reasons, reasons


def _adapt_semantic_contract(original_contract: Any, group_map: Mapping[str, tuple[str, ...]]) -> Any:
    """Expande coaliciones agrupadas antes de aplicar criterios originales."""

    if original_contract is None or not getattr(original_contract, "criteria", ()):
        return original_contract

    adapted = []
    for criterion in original_contract.criteria:
        original_evaluator = criterion.evaluator

        def evaluator(state, state_kind, coalition, expected, tolerance,
                      _original=original_evaluator):
            expanded = frozenset(
                atomic
                for grouped in coalition
                for atomic in group_map.get(grouped, (grouped,))
            )
            return _original(state, state_kind, expanded, expected, tolerance)

        adapted.append(
            SemanticCriterion(
                name=criterion.name,
                evaluator=evaluator,
                description=criterion.description,
            )
        )
    return SemanticContract(
        description=original_contract.description,
        criteria=tuple(adapted),
    )


def build_grouped_case(
    original_case: Any,
    candidate: GranularityCandidate,
    *,
    name: Optional[str] = None,
) -> Any:
    """Construye un QSMEFCase agrupado y reescribe correctamente el schedule."""

    normalized = _normalize_operations(original_case.operations)
    group_names: list[str] = []
    group_map: dict[str, tuple[str, ...]] = {}
    atomic_to_group: dict[str, str] = {}
    new_operations = {
        op_name: op
        for op_name, op in normalized.items()
        if op_name not in set(original_case.players)
    }

    for number, group in enumerate(candidate.groups, start=1):
        proposed = "+".join(group)
        group_name = proposed if proposed not in new_operations else f"G{number}_{proposed}"
        group_names.append(group_name)
        group_map[group_name] = tuple(group)
        atomic_to_group.update({atomic: group_name for atomic in group})
        specs = tuple(normalized[atomic] for atomic in group)

        if all(spec.is_matrix() for spec in specs):
            matrices = tuple(np.asarray(spec.action, dtype=complex) for spec in specs)
            composed = np.eye(matrices[0].shape[0], dtype=complex)
            for matrix in matrices:
                composed = matrix @ composed
            new_operations[group_name] = OperationSpec(action=composed)
        else:
            declared = all(bool(spec.declared_unitary) for spec in specs)

            # OperationSpec.apply invoca las acciones funcionales con un solo
            # argumento y solo admite vectores de estado para ese caso.
            def apply_group(state, _specs=specs):
                result = state
                for spec in _specs:
                    result = spec.apply(result, "statevector")
                return result

            new_operations[group_name] = OperationSpec(
                action=apply_group,
                declared_unitary=declared,
                unitarity_rationale=(
                    "Composición ordenada de operaciones unitarias declaradas"
                    if declared else ""
                ),
            )

    new_schedule: list[str] = []
    emitted: set[str] = set()
    for operation in map(str, original_case.schedule):
        if operation not in atomic_to_group:
            new_schedule.append(operation)
            continue
        grouped = atomic_to_group[operation]
        if grouped not in emitted:
            new_schedule.append(grouped)
            emitted.add(grouped)

    contract = _adapt_semantic_contract(original_case.semantic_contract, group_map)
    return QSMEFCase(
        name=name or f"{original_case.name} | granularidad: {candidate.label}",
        initial_state=original_case.initial_state,
        operations=new_operations,
        players=tuple(group_names),
        schedule=tuple(new_schedule),
        observable=original_case.observable,
        constraints=original_case.constraints,
        semantic_contract=contract,
        property_description=original_case.property_description,
        tolerance=original_case.tolerance,
    )


def _interpretability(candidate: GranularityCandidate, metadata: Mapping[str, BlockMetadata]) -> float:
    """Premia grupos funcionalmente homogéneos."""

    scores = []
    for group in candidate.groups:
        roles = [metadata[name].role for name in group]
        dominant = max(roles.count(role) for role in set(roles))
        scores.append(dominant / len(roles))
    return float(np.mean(scores))


def recommend_granularity(
    case: Any,
    metadata: Mapping[str, BlockMetadata],
    objective: GranularityObjective,
    *,
    show_table: bool = True,
) -> dict[str, Any]:
    """Genera, valida y ordena granularidades candidatas.

    La recomendación prioriza: aplicabilidad QSMEF, alineación con la
    pregunta, homogeneidad funcional y menor costo. No afirma que exista una
    granularidad universalmente correcta.
    """

    players = set(map(str, case.players))
    missing_metadata = players - set(metadata)
    if missing_metadata:
        raise ValueError(f"Faltan metadatos para: {sorted(missing_metadata)}")
    if objective.max_coalitions < 2:
        raise ValueError("max_coalitions debe ser al menos 2.")

    candidates = generate_contiguous_candidates(
        case, max_candidates=objective.max_candidates
    )
    rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    max_players_budget = max(1.0, log2(objective.max_coalitions))

    for candidate in candidates:
        structurally_valid, reasons = _candidate_filters(
            candidate, metadata, objective
        )
        entry: dict[str, Any] = {
            "candidate": candidate,
            "case": None,
            "validation": None,
            "accepted": False,
            "reasons": list(reasons),
        }
        if structurally_valid:
            grouped_case = build_grouped_case(case, candidate)
            # El validador original imprime un informe aun cuando no muestra
            # tablas. Durante el ranking se captura esa salida para que cada
            # candidata no inunde el notebook.
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                validation = validate_qsmef_case(grouped_case, show_tables=False)
            applicable = validation["verdict"] == "APLICABLE"
            if objective.validate_semantics and not applicable:
                entry["reasons"].append(validation["reason"])
            else:
                entry["accepted"] = True
            entry["case"] = grouped_case
            entry["validation"] = validation

        interpretability = _interpretability(candidate, metadata)
        cost_score = max(0.0, 1.0 - candidate.player_count / max_players_budget)
        # La aplicabilidad domina; luego interpretabilidad y costo.
        score = (
            (100.0 if entry["accepted"] else 0.0)
            + 10.0 * interpretability
            + 5.0 * cost_score
        )
        entry["score"] = score
        rows.append({
            "Granularidad": candidate.label,
            "Jugadores": candidate.player_count,
            "Coaliciones": candidate.coalition_count,
            "Interpretabilidad": round(interpretability, 3),
            "Veredicto": (
                entry["validation"]["verdict"]
                if entry["validation"] is not None
                else "DESCARTADA"
            ),
            "Aceptada": entry["accepted"],
            "Puntaje": round(score, 3),
            "Justificación": "; ".join(entry["reasons"]) or "cumple todos los criterios",
        })
        if entry["accepted"]:
            accepted.append(entry)

    accepted.sort(key=lambda item: (-item["score"], item["candidate"].player_count))
    table = pd.DataFrame(rows).sort_values(
        ["Aceptada", "Puntaje", "Jugadores"], ascending=[False, False, True]
    ).reset_index(drop=True)
    recommendation = accepted[0] if accepted else None

    if show_table:
        print("=" * 80)
        print("ASISTENTE DE SELECCIÓN DE GRANULARIDAD QSMEF")
        print("=" * 80)
        print(f"Pregunta: {objective.question}")
        print(f"Roles a distinguir: {tuple(objective.distinguish_roles) or 'no inferidos'}")
        print(f"Presupuesto: {objective.max_coalitions} coaliciones")
        display(table)
        if recommendation:
            print(f"\nRECOMENDACIÓN: {recommendation['candidate'].label}")
            print(
                "Interpretación: granularidad recomendada para la pregunta, "
                "el alcance y el presupuesto declarados."
            )
        else:
            print("\nSIN RECOMENDACIÓN: ninguna candidata cumple todos los criterios.")

    return {
        "objective": objective,
        "recommendation": recommendation,
        "accepted": accepted,
        "all_results": rows,
        "table": table,
    }


def example_maxcut_granularity(case: Any) -> dict[str, Any]:
    """Ejemplo para el caso C1-M1-C2-M2 ya construido en el notebook."""

    metadata = {
        "C1": BlockMetadata("C1", "costo", keywords=("cost", "coste")),
        "M1": BlockMetadata("M1", "mezcla", keywords=("mixer", "mixing")),
        "C2": BlockMetadata("C2", "costo", keywords=("cost", "coste")),
        "M2": BlockMetadata("M2", "mezcla", keywords=("mixer", "mixing")),
    }
    objective = objective_from_question(
        "¿Cómo contribuyen los bloques de costo y mezcla de cada nivel?",
        metadata,
        max_coalitions=64,
        required_roles=("costo", "mezcla"),
    )
    return recommend_granularity(case, metadata, objective)


print("Asistente semiautomático de granularidad QSMEF cargado correctamente.")

"""## 6. Orquestador del pipeline QSMEF

"""


# ============================================================
# SOLICITUDES DE GRANULARIDAD PARA LOS CASOS DOCUMENTADOS
# ============================================================

def build_granularity_request(case_key: str):
    """Devuelve metadatos y objetivo para los casos ya documentados.

    Los demás casos conservan la granularidad declarada por su constructor
    hasta que el investigador formule su pregunta y sus metadatos funcionales.
    """

    if case_key == "maxcut":
        metadata = {
            "C1": BlockMetadata("C1", "costo", keywords=("cost", "coste")),
            "B1": BlockMetadata("B1", "mezcla", keywords=("mixer", "mixing", "mezclador")),
            "C2": BlockMetadata("C2", "costo", keywords=("cost", "coste")),
            "B2": BlockMetadata("B2", "mezcla", keywords=("mixer", "mixing", "mezclador")),
        }
        objective = objective_from_question(
            "¿Cómo contribuyen los bloques de costo y mezcla de cada nivel QAOA?",
            metadata,
            required_roles=("costo", "mezcla"),
            max_coalitions=16,
        )
        return metadata, objective

    if case_key == "h2":
        metadata = {
            "R0": BlockMetadata("R0", "rotación", keywords=("rotation", "rotacion", "rotaciones")),
            "E0": BlockMetadata("E0", "entrelazamiento", keywords=("entanglement", "entangler")),
            "R1": BlockMetadata("R1", "rotación", keywords=("rotation", "rotacion", "rotaciones")),
            "E1": BlockMetadata("E1", "entrelazamiento", keywords=("entanglement", "entangler")),
            "R2": BlockMetadata("R2", "rotación", keywords=("rotation", "rotacion", "rotaciones")),
        }
        objective = objective_from_question(
            "¿Cómo contribuyen las capas de rotación y entrelazamiento al cambio de la energía electrónica de H2?",
            metadata,
            required_roles=("rotación", "entrelazamiento"),
            max_coalitions=32,
        )
        return metadata, objective

    if case_key == "skw":
        metadata = {
            "O": BlockMetadata("O", "oráculo", keywords=("oracle", "marcado")),
            "G": BlockMetadata("G", "moneda", keywords=("coin", "grover")),
            "S": BlockMetadata("S", "desplazamiento", keywords=("shift", "flip-flop")),
        }
        objective = objective_from_question(
            "¿Cómo contribuyen el oráculo, la moneda de Grover y el desplazamiento flip-flop al cambio del observable energético en un paso de SKW?",
            metadata,
            required_roles=("oráculo", "moneda", "desplazamiento"),
            max_coalitions=8,
        )
        return metadata, objective

    return None


# ============================================================
# CONSTRUCTORES REGISTRADOS
# ============================================================

def build_case_registry():
    """Construye los seis casos con sus granularidades atómicas declaradas."""

    return {
        "maxcut": ("QAOA-MaxCut C6", build_maxcut_case()),
        "h2": ("TwoLocal-H2", build_h2_case()),
        "skw": (
            "SKW hipercubo",
            build_skw_case(n=8, marked_vertex=0, time_steps=19, gamma=1.0),
        ),
        "qpe": (
            "QPE-Shor",
            build_qpe_shor_case(N=21, a=2, counting_qubits=5),
        ),
        "classifier": ("Clasificador cuántico", build_classifier_case()),
        "partitioning": (
            "Graph Partitioning (Min Bisection)",
            build_graph_partitioning_case(n=6),
        ),
    }


# ============================================================
# PIPELINE UNIVERSAL DE UN CASO
# ============================================================

def run_qsmef_case(
    case_key: str,
    label: str,
    base_case: QSMEFCase,
    *,
    use_granularity_assistant: bool = False,
    show_granularity_table: bool = False,
    show_validation_tables: bool = False,
    show_shapley_table: bool = True,
):
    """Ejecuta la validación de un caso con la granularidad declarada.
    Opcionalmente, puede invocar el asistente de granularidad como
    herramienta auxiliar. Las contribuciones de Shapley se calculan
    únicamente cuando el caso resulta aplicable."""

    selected_case = base_case
    granularity_result = None
    selection_status = "GRANULARIDAD DECLARADA"

    request = build_granularity_request(case_key)
    if use_granularity_assistant and request is not None:
        metadata, objective = request
        granularity_result = recommend_granularity(
            base_case,
            metadata,
            objective,
            show_table=show_granularity_table,
        )
        recommendation = granularity_result["recommendation"]
        if recommendation is not None:
            selected_case = recommendation["case"]
            selection_status = "RECOMENDADA"
        else:
            # Se conserva el caso declarado para informar por qué no resulta
            # admisible; no se presenta como recomendación del asistente.
            selection_status = "SIN CANDIDATA ADMISIBLE"

    validation = validate_qsmef_case(
        selected_case,
        show_tables=show_validation_tables,
    )

    shapley = None
    if validation["verdict"] == "APLICABLE":
        shapley = compute_shapley_values(
            selected_case,
            validation,
            show_table=show_shapley_table,
        )
    else:
        print(
            "No se calculan contribuciones de Shapley porque la "
            "configuración no satisface la aplicabilidad declarada."
        )

    return {
        "key": case_key,
        "label": label,
        "base_case": base_case,
        "selected_case": selected_case,
        "selection_status": selection_status,
        "granularity": granularity_result,
        "validation": validation,
        "shapley": shapley,
    }


# ============================================================
# ORQUESTADOR MULTICASO
# ============================================================

def run_qsmef_multicase(
    *,
    use_granularity_assistant: bool = True,
    show_granularity_tables: bool = False,
    show_validation_tables: bool = False,
    show_shapley_tables: bool = True,
):
    """Ejecuta el pipeline uniforme y devuelve detalle y resumen."""

    registry = build_case_registry()
    results = {}
    rows = []

    for case_key, (label, case) in registry.items():
        print("\n" + "=" * 88)
        print("CASO:", label)
        print("=" * 88)

        result = run_qsmef_case(
            case_key,
            label,
            case,
            use_granularity_assistant=use_granularity_assistant,
            show_granularity_table=show_granularity_tables,
            show_validation_tables=show_validation_tables,
            show_shapley_table=show_shapley_tables,
        )
        results[case_key] = result

        shapley_text = "No calculado"
        if result["shapley"] is not None and result["shapley"].get("computed"):
            shapley_text = " | ".join(
                f"{player}: {value:.6f}"
                for player, value in result["shapley"]["shapley_values"].items()
            )

        selected = result["selected_case"]
        rows.append({
            "Caso": label,
            "Selección": result["selection_status"],
            "Granularidad analizada": " | ".join(selected.players),
            "Jugadores": len(selected.players),
            "Coaliciones": 2 ** len(selected.players),
            "Veredicto": result["validation"]["verdict"],
            "Motivo": result["validation"]["reason"],
            "Shapley": shapley_text,
        })

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("RESUMEN FINAL QSMEF")
    print("=" * 88)
    display(summary)

    return {
        "results": results,
        "summary": summary,
    }


print("Orquestador QSMEF cargado correctamente.")

"""## 7. Ejecución y resumen

"""

# ============================================================
# EJECUCIÓN CONTROLADA
# ============================================================

# Mantener False para cargar y revisar el notebook sin ejecutar todos
# los experimentos. Cambiar a True para ejecutar el pipeline completo.
RUN_ALL_CASES = True

if RUN_ALL_CASES:
    qsmef_results = run_qsmef_multicase(
        use_granularity_assistant=False,
        show_granularity_tables=False,
        show_validation_tables=False,
        show_shapley_tables=True,
    )
else:
    print(
        "Módulos cargados. Cambie RUN_ALL_CASES a True para ejecutar "
        "los seis casos."
    )
