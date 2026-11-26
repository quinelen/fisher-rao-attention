# Resultado — ¿Qué aprende la varianza σ? (prueba prerregistrada)

24-oct-2026 · uso interno · protocolo `PREREGISTRO_SIGMA.md`, sellado en `SELLO_SIGMA.txt` (23:46 UTC) antes de la corrida · datos: `log_sigma.txt`, `resultados_sigma.json`.

**Veredicto prerregistrado: CONFIRMADA.** En la atención Fisher–Rao isotrópica entrenada en Dyck-2, σ codifica la profundidad de anidamiento. **Dirección: APOYADA** (5 de 5 semillas), con un matiz: solo σ_q baja con la profundidad; σ_k sube (ver *post hoc*).

Control de entrenamiento (`close_acc`): 98,5 % dentro de distribución (97,3–99,4 %) y 88,9 % fuera de distribución (88,1–89,7 %), en línea con el ≈ 88 % del brazo isotrópico del paper.

| Semilla | 401 | 402 | 403 | 404 | 405 | Media |
|---|---|---|---|---|---|---|
| S entrenado (DD) | 0,691 | 0,492 | 0,637 | 0,483 | 0,619 | **0,584** |
| S sin entrenar (DD) | 0,001 | 0,001 | 0,001 | 0,001 | 0,001 | 0,001 |
| Combinación de S¹ | c2·q·cab3 | c2·q·cab0 | c2·q·cab2 | c2·q·cab1 | c2·q·cab3 | — |
| Coef. profundidad (log σ/nivel) | −0,49 | −0,44 | −0,53 | −0,36 | −0,34 | −0,43 |
| S fuera de distribución (FD) | 0,549 | 0,424 | 0,551 | 0,367 | 0,531 | **0,484** |

¹ c2 = segunda capa (`capa1_*` en el JSON, que indexa desde 0).

- **Primario:** diferencia media +0,583; t(4) = 14,09; p = 1,5·10⁻⁴; media ≥ 0,10. Cada nivel de profundidad multiplica σ_q por ≈ 0,59–0,71.
- **Dirección:** coeficiente negativo en la combinación de S en 5/5 semillas. Fracción de las 16 combinaciones con coeficiente negativo: 0,44 (0,31–0,56).
- **σ_q vs σ_k:** ΔR² medio 0,205 (q) frente a 0,143 (k); diferencia +0,062; t(4) = 2,99; p = 0,040. La combinación de S es σ_q de la capa 2 en 5/5.
- **Secuencias más largas (80–120):** se mantiene. S_FD 0,484 frente a 0,0004 sin entrenar (t(4) = 12,92; p = 2,1·10⁻⁴). La combinación elegida dentro de distribución es la misma que alcanza el máximo fuera de distribución en 5/5 semillas, sin nueva selección. Su coeficiente se atenúa, de −0,34 a −0,53 a un rango de −0,17 a −0,28.
- **Exploratoria (AUC de error en cierres):** dentro de distribución, la mejor combinación alcanza AUC 0,80–0,87 (σ_k de la capa 2 en 5/5), frente a 0,69–0,73 de la profundidad sola (249–1.093 errores de 39.709). Fuera de distribución, 0,77–0,79 en 4 semillas y 0,21 (sentido inverso) en una, frente a 0,58–0,61 de la profundidad (≈ 10.300–11.900 errores de 99.944).

**Post hoc (no prerregistrado, descriptivo).** La capa 1 recibe solo símbolo + posición, sin mezcla previa: no puede ver la profundidad por construcción y su ΔR² es ≈ 0. Por eso la fracción de las 16 queda diluida. En la capa 2 el patrón es nítido: σ_q tiene coeficiente negativo en 20/20 cabezas×semilla y σ_k positivo en 20/20. La consulta se vuelve más estrecha con la profundidad y la clave más ancha.

**Límites.** Es asociación lineal condicionada a símbolo y tramos de posición, no causalidad: no hubo intervención sobre σ, y σ puede codificar un correlato de la profundidad (p. ej., distancia a la apertura pendiente). Una sola tarea sintética, un modelo pequeño y n = 5. S es un máximo sobre 16 combinaciones, aunque el control usa el mismo máximo y la transferencia a FD no reselecciona. La AUC es exploratoria, con selección sobre 16 y confundida con posición y largo. El hallazgo q↓/k↑ es *post hoc* y requiere su propio prerregistro antes de citarse.
