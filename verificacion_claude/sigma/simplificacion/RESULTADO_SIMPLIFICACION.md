# Resultado — σ fijo en la capa 1 de la atención Fisher–Rao (no inferioridad, prueba prerregistrada)

25-oct-2026 · uso interno · protocolo `PREREGISTRO_SIMPLIFICACION.md`, sellado en `SELLO_SIMPLIFICACION.txt` (18:24 UTC) antes de la corrida · datos: `log_simplificacion.txt`, `resultados_simplificacion.json`.

**Veredicto prerregistrado: B es NO INFERIOR a A.** Fijar σ en la capa 1 (un escalar aprendido por cabeza y rol en lugar de σ por token) no reduce la exactitud de cierre fuera de distribución. La diferencia B − A es de +0,04 pp, con IC 95 % de −0,19 a +0,28 pp; el límite inferior queda sobre el margen de −0,5 pp.

Diseño: Dyck-2, entrenamiento idéntico al de §4.1, semillas 701–710 (n = 10). Los tres brazos parten de la misma inicialización en los parámetros compartidos. Con θ fijo, B reproduce la salida de A con sg anulado y sesgo = θ (máx. |dif| 5·10⁻⁷). No hubo NaN ni exclusiones.

| Exactitud de cierre (media de 10; rango) | A: σ por token | B: σ fijo en capa 1 | C: σ fijo en ambas | B − A (IC 95 %) | C − A (IC 95 %) |
|---|---|---|---|---|---|
| **FD, 80–120 (primario)** | 88,41 % (86,95–89,19) | 88,45 % (86,96–89,25) | 87,69 % (85,23–88,36) | **+0,04 pp (−0,19 a +0,28)** | −0,71 pp (−1,13 a −0,30) |
| DD, 20–60 | 98,27 % (97,35–99,07) | 98,50 % (98,00–98,85) | 97,05 % (96,09–97,67) | +0,23 pp (−0,07 a +0,53) | −1,22 pp (−1,63 a −0,81) |

- **Primario:** B − A por semilla va de −0,48 a +0,69 pp (5 negativas, 5 positivas). Superioridad, descriptiva: t(9) = 0,41; p = 0,69. B no mejora a A; solo no la empeora.
- **DD:** también no inferior (límite inferior −0,07 pp > −0,5); t(9) = 1,75; p = 0,11.
- **Control positivo C:** la predicción se cumple en ambos rangos. Fijar σ también en la capa 2 cuesta 0,71 pp en FD (p = 0,004; 9 de 10 semillas peores) y 1,22 pp en DD (p < 0,001; 10 de 10). La prueba distingue una simplificación inocua de una dañina.
- **Parámetros:** A tiene 415.255. B ahorra 1.024 (0,25 %) y C, 2.048 (0,49 %).
- **Tiempo de entrenamiento (5 épocas):** A 36,7 s, B 37,8 s, C 36,7 s; razón B/A 1,03 (B − A +1,1 s, IC −0,4 a +2,6). No hay ahorro de tiempo: el costo está en la distancia sobre L × L, no en las capas de σ. Las mediciones variaron entre 32 y 44 s por la carga del equipo.
- **Mecanismo en B:** la profundidad sigue en σ_q de la capa 2. El ΔR² máximo por semilla en DD es 0,565 en B y 0,572 en A (B − A −0,007; IC −0,032 a +0,018), muy sobre el umbral de 0,10. El coeficiente medio de profundidad es −0,33 en ambos. En FD: 0,468 en B y 0,474 en A.
- **σ aprendido en la capa 1 de B:** quedó cerca del valor inicial, entre 0,98 y 1,07 (media por cabeza 1,01–1,03). En C, la capa 2 bajó a ≈ 0,85.

**Post hoc (no prerregistrado).** En B, σ_q y σ_k de la capa 1 terminaron iguales en cada cabeza y semilla. Es una consecuencia de la fórmula: con σ_q = σ_k al inicio, el término (σ_q − σ_k)² tiene gradiente nulo y el resto es simétrico en ambos. Por eso σ_q y σ_k reciben el mismo gradiente y se mueven juntos. En la práctica, la capa 1 de B usa la distancia √2·arccosh(1 + ‖Δμ‖²/(4σ²)), con una sola escala por cabeza.

**Lectura.** En esta tarea, σ por token solo hace falta en la capa 2, donde codifica la profundidad. En la capa 1 basta un escalar por cabeza: se pierde la variación de σ entre tokens, pero no la exactitud ni el mecanismo de la capa 2. Quitar σ también en la capa 2 sí cuesta exactitud. El ahorro es pequeño (0,25 % de los parámetros, sin ganancia de tiempo). El hallazgo es sobre dónde se usa σ, no sobre eficiencia.

**Límites.** Es una sola tarea sintética, con un modelo pequeño de 2 capas, n = 10, y la única medida de desempeño es `close_acc`. Los conjuntos de datos son los de las pruebas previas. Las secundarias no tienen corrección por multiplicidad. En el JSON, `sigma_fijo` numera las capas desde 1, mientras que `dR2_q2` sigue el índice de `../exp_sigma_profundidad.py`, donde `capa1_q` corresponde a la capa 2.
