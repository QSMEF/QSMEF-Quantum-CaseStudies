# Validador universal QSMEF

Este directorio contiene el artefacto computacional desarrollado para
operacionalizar la validación de aplicabilidad de QSMEF sobre diferentes
implementaciones cuánticas.

## Artefacto principal

El archivo `qsmef_universal_validator.py` contiene la implementación
ejecutable del artefacto computacional desarrollado para operacionalizar
la validación de aplicabilidad de QSMEF.

La implementación presenta una arquitectura modular compuesta por:

- constructores específicos de casos;
- la estructura común `QSMEFCase`;
- el Validador universal;
- el cálculo de contribuciones mediante valores de Shapley;
- un orquestador multicaso;
- un asistente opcional de granularidad.

## Arquitectura

Cada implementación cuántica se representa mediante un constructor específico
que genera una instancia de `QSMEFCase`.

La instancia contiene la información necesaria para realizar el análisis:
estado inicial, operaciones, bloques funcionales, secuencia de ejecución,
observable, restricciones funcionales y contrato semántico.

El Validador recibe exclusivamente una instancia de `QSMEFCase` y aplica el
mismo procedimiento de validación independientemente del algoritmo analizado.
La lógica específica de cada implementación se encuentra declarada en el
constructor correspondiente y no en el Validador universal.

De manera simplificada, el flujo es:

`Constructor del caso → QSMEFCase → Validador universal → Informe de validación`

Cuando el caso resulta aplicable, el artefacto calcula las contribuciones
funcionales de los bloques mediante el valor de Shapley.

## Resultados de validación

El Validador puede producir cuatro resultados:

- `ERROR EN LA ESPECIFICACIÓN`
- `NO APLICABLE BAJO EL ALCANCE DECLARADO`
- `REQUIERE CONTRATO SEMÁNTICO`
- `APLICABLE`

Los valores de Shapley se calculan únicamente cuando el resultado de la
validación es `APLICABLE`.

## Granularidad

La granularidad del análisis constituye una decisión metodológica del
investigador y no es determinada automáticamente por QSMEF.

El notebook incluye un asistente de granularidad como herramienta auxiliar
para explorar y comparar granularidades candidatas. Su utilización es
opcional y no reemplaza la selección y justificación realizada por el
investigador.

## Casos incluidos

El notebook contiene constructores y permite ejecutar el procedimiento sobre
seis implementaciones:

- QAOA-MaxCut;
- TwoLocal-H2;
- SKW en el hipercubo;
- QPE aplicado al procedimiento de búsqueda del orden de Shor;
- clasificador cuántico;
- particionamiento de grafos.

Los cuatro primeros corresponden a los casos principales analizados en la
tesis. Los dos restantes se incluyen como casos adicionales del artefacto.

## Ejecución

Para reproducir la ejecución multicaso, abrir
`QSMEF_Universal_Validator.ipynb` y ejecutar las celdas en orden.

Por defecto, la ejecución utiliza las granularidades declaradas en los
constructores de los casos. El asistente de granularidad permanece
desactivado durante esta ejecución.

El orquestador multicaso construye las instancias, invoca el mismo Validador
universal para cada una y reúne los resultados obtenidos.
