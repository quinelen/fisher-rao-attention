# Pruebas prerregistradas: qué aprende la escala σ (§4.9 del paper)

Cada carpeta contiene el protocolo (`PREREGISTRO_*.md`), su sello SHA-256 previo a la ejecución (`SELLO_*.txt`), el registro de la corrida (`log_*.txt`), los resultados (`resultados_*.json`) y un resumen (`RESULTADO_*.md`).

| Carpeta | Prueba | Semillas |
|---|---|---|
| `.` | ¿σ codifica la profundidad de anidamiento? | 401–405 |
| `intervencion/` | ¿El modelo usa la profundidad de σ_q? | 401–405 |
| `codigo/` | ¿Aparece el mismo código en Python real? | 501–505 |
| `confirmatorias/` | Consultas se estrechan y claves se ensanchan (capa 2); σ_k como señal de error | 601–605 |
| `simplificacion/` | σ fijo en la capa 1 (no inferioridad) | 701–710 |
| `asimetria/` | Alcance, intercambio σ_q↔σ_k y otras jerarquías | 801–805 |

`BUSQUEDA_ASIMETRIA.md` es la búsqueda de antecedentes (29-oct-2026) que respalda la frase «no lo encontramos en trabajos previos»; es un documento de trabajo y sus notas de redacción reflejan el estado del paper a esa fecha. `PLAN_QUE_APRENDE_SIGMA.md` es el borrador de diseño que cita el primer protocolo; de él se quitaron notas de logística ajenas al diseño y «decisión de Eduardo» pasó a «decisión de los autores».

**Sobre los archivos.** Los protocolos sellados no pueden editarse sin romper el sello, así que se publican tal como se sellaron: llevan el encabezado «uso interno», anterior a la decisión de publicarlos, y rutas absolutas de la máquina donde se sellaron; la carpeta `pruebas/sigma/…` que nombran corresponde aquí a `verificacion_claude/sigma/…`. Los resúmenes de resultados, el plan y la búsqueda de antecedentes conservan el mismo encabezado, de cuando eran documentos de trabajo.

**Scripts.** Los scripts (`exp_*.py`) no se incluyen; están disponibles a pedido de los autores. Sus hashes SHA-256 están en cada `SELLO_*.txt`, de modo que la versión entregada puede cotejarse con la sellada. Importan sin modificar `exp_tres_geometrias.py` y `fast_gaussian_test.py` de `verificacion_claude/`; la prueba de `codigo/` usa además `exp_estabilizado.py`, `exp_codigo_python.py` y `exp_codigo_extrapolativo.py`, y la de `asimetria/`, `exp1_profundidad.py`.
