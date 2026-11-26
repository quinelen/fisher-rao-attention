# Resultado — ¿σ codifica el anidamiento en código Python real? (prueba prerregistrada)

25-oct-2026 · uso interno · protocolo `PREREGISTRO_CODIGO.md`, sellado en `SELLO_CODIGO.txt` (00:17 UTC) antes de entrenar · datos: `log_codigo.txt`, `resultados_codigo.json`.

**Veredicto prerregistrado: NO CONFIRMADA.** En el brazo Fisher–Rao estabilizado entrenado con código Python, σ no codifica la indentación de forma consistente. La media de S es 0,026, la mitad del umbral de 0,05, y el contraste con el modelo sin entrenar no alcanza significación (p = 0,056). **Dirección: NO APOYADA** (2 de 5).

Control de entrenamiento: 3,31 bpc en L = 128 (3,30–3,32) y 3,73 bpc en la zona extrapolativa de L = 512 (3,64–3,80). Coincide con la referencia del brazo estabilizado (3,30 y 3,74).

Inclusión: el análisis de indentación usa solo los caracteres cuya línea completa cabe en la ventana. Eso es el 56,5 % de las posiciones en L = 128 (se excluye el 43,5 %) y el 89,8 % en L = 512 (se excluye el 10,2 %).

| Semilla | 501 | 502 | 503 | 504 | 505 | Media |
|---|---|---|---|---|---|---|
| S entrenado (indentación, L = 128) | 0,041 | 0,003 | 0,020 | 0,014 | 0,054 | **0,026** |
| S sin entrenar | 0,005 | 0,001 | 0,001 | 0,002 | 0,002 | 0,002 |
| Combinación de S¹ | c2·k·cab0 | c2·k·cab0 | c2·q·cab1 | c2·k·cab1 | c2·k·cab1 | — |
| Coef. indentación (log σ/nivel) | −0,047 | +0,016 | +0,037 | +0,028 | −0,064 | — |
| S en L = 512 | 0,019 | 0,002 | 0,008 | 0,010 | 0,030 | 0,014 |

¹ c2 = segunda capa (`capa1_*` en el JSON, que indexa desde 0).

- **Primario:** diferencia +0,024; t(4) = 2,67; p = 0,056; media < 0,05.
- **Dirección:** el signo coincide con la predicción tomada de Dyck-2 (σ_q baja, σ_k sube) en 2 de 5 semillas. En la capa 2, el coeficiente es negativo en 3 de 20 cabezas×semilla para σ_q y en 11 de 20 para σ_k: sin patrón.
- **σ_q vs σ_k:** ΔR² medio de 0,0020 (q) frente a 0,0045 (k); p = 0,40.
- **L = 512:** S = 0,014 frente a 0,001 sin entrenar (t(4) = 2,67; p = 0,056); no se cumple. Sin reseleccionar, la combinación elegida en L = 128 da 0,014.
- **Profundidad de paréntesis (L = 128):** S_par = 0,0019 frente a 0,0001 (p = 0,061); no se cumple. Es σ_k de la capa 2 en 5 de 5, con coeficiente positivo en 5 de 5.
- **Sensibilidad a la columna:** agregar la columna dentro de la línea no cambia el resultado. ΔR² en la combinación de S: 0,026 con columna y 0,026 sin ella. Lo poco que hay no es un efecto de la columna.

**Post hoc (no prerregistrado, descriptivo).** El modelo entrenado supera al sin entrenar en 5 de 5 semillas en las tres medidas (indentación en L = 128, en L = 512 y paréntesis). Hay algo de información, pero es pequeña e irregular: dos semillas (501 y 505) concentran casi toda la señal. Lo que se encontró apunta sobre todo a σ_k de la capa 2, no a σ_q.

**Lectura.** El hallazgo de Dyck-2 (σ_q de la capa 2 como codificador nítido de la profundidad, ΔR² ≈ 0,58) no se generaliza a código real en esta configuración. A lo sumo, σ explica un 2–5 % de varianza atribuible a la indentación más allá del símbolo y la posición, y solo en algunas semillas, sin dirección estable.

**Límites.** Es un modelo pequeño y poco entrenado (≈ 3,3 bpc, 3 épocas), con un texto de prueba corto (37.771 caracteres), por lo que las 400 ventanas se solapan. Las dos variables se definen a nivel de carácter (la profundidad de paréntesis incluye cadenas y comentarios y no ve aperturas previas a la ventana), y en L = 128 la inclusión es parcial. Con n = 5 y p = 0,056, una señal pequeña real no queda descartada: esto no prueba que no exista, solo que no alcanzó el umbral prerregistrado.
