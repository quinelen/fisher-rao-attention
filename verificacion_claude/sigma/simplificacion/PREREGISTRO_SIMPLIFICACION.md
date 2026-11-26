# Prerregistro — Simplificación de σ: ¿se puede fijar σ en la capa 1 sin perder exactitud? (no inferioridad)

Fecha: 25-oct-2026 (UTC). Uso interno. Antecedentes: `../RESULTADO_SIGMA.md` (σ_q de la capa 2 codifica la profundidad), `../intervencion/RESULTADO_INTERVENCION.md` (quitar la profundidad de σ_q de la capa 1 no cambia la exactitud; en la capa 2 sí, poco) y `../confirmatorias/RESULTADO_CONFIRMATORIAS.md`. Este documento y el script `exp_simplificacion.py` se sellan (SHA-256, `SELLO_SIMPLIFICACION.txt`) **antes** de la corrida. Nada de lo fijado aquí se cambia después de ver datos; todo análisis adicional se rotula *post hoc*; se informan todos los resultados.

Convención: capa 1 = primera capa del modelo (`model.layers[0]`); capa 2 = segunda (`model.layers[1]`).

## 1. Entrenamiento (idéntico a §4.1 y a `../exp_sigma_profundidad.py`)
Dyck-2; `Model` y `Dyck2Dataset` del código del paper (solo lectura); d_model 128, 2 capas, 4 cabezas; 10.000 secuencias de entrenamiento de largo 20–60 (semilla de datos 42); AdamW, lr 5·10⁻⁴, weight decay 0,01 sobre todos los parámetros; 5 épocas; lote 64; `torch.set_num_threads(8)`. Evaluación: DD = 2.000 secuencias de 20–60 (semilla 123); FD = 2.000 de 80–120 (semilla 456). Medida: `close_acc` del paper. El bucle se copia sin cambios de `entrenar`, con un único agregado: la sustitución de σ (§2) se hace después de construir el modelo y antes de crear el optimizador.

**Semillas nuevas 701–710 (n = 10).** Los tres brazos se entrenan desde cero con la misma semilla por trío (`torch.manual_seed(s)` y `np.random.seed(s)` antes de construir el modelo). Como la sustitución ocurre después de construir, todos los parámetros compartidos parten de los mismos valores en A, B y C.

## 2. Brazos
- **A (referencia):** `IsotropicHyperbolicAttention` original en ambas capas: σ_q y σ_k por token, σ = softplus(sg(x)) + 0,05, con sg_q y sg_k lineales 128 → 4.
- **B (simplificado):** igual a A, pero en la **capa 1** se reemplazan sg_q y sg_k por un módulo sin entrada: un escalar aprendido θ por cabeza y rol (4 + 4 parámetros), de modo que σ = softplus(θ) + 0,05, igual para todos los tokens. θ se inicializa en softplus⁻¹(0,95) para que σ = 1,00 al inicio. La capa 2 queda igual a A. La atención usa el `forward` original del paper sin cambios (el módulo sustituto devuelve θ expandido a [lote, largo, cabezas]); θ recibe el mismo weight decay que el resto, como los sesgos de A.
- **C (control positivo, secundario):** la misma sustitución en **ambas** capas.
- **Verificación de equivalencia (se corre en la prueba de humo y al inicio de la corrida):** con θ fijo, B debe dar la misma salida que A con los pesos de sg_q/sg_k anulados y su sesgo = θ, a igualdad del resto de los pesos (máx. |diferencia| ≤ 10⁻⁵ en la atención aislada y en los logits del modelo completo). Si no se cumple, no se corre.

## 3. Endpoint primario (no inferioridad)
Exactitud de cierre en FD (80–120), en puntos porcentuales (pp). Por semilla, d = acc(B) − acc(A). Se calcula la media de d y su IC 95 % bilateral con t de Student (gl = 9). **Margen fijado: −0,5 pp.**
- **B es NO INFERIOR** si el límite inferior del IC 95 % > −0,5 pp.
- Si no: **no inferioridad no demostrada**. Si además el límite superior < −0,5 pp, B es **inferior**.

## 4. Secundarias (no decisorias; sin corrección por multiplicidad)
1. B − A en DD (20–60), mismo estadístico y mismo margen.
2. Superioridad B frente a A, descriptiva: t pareado bilateral (gl = 9) en FD y en DD.
3. C − A en FD y en DD (media, IC 95 %, t pareado). Predicción: C es peor que A porque en la capa 2 σ codifica la profundidad. Se considera **apoyada** en un rango si el límite superior del IC 95 % de C − A < 0.
4. Parámetros totales por brazo y ahorrados por B y C frente a A.
5. Tiempo de entrenamiento (segundos, 5 épocas) por brazo: media y rango; razón B/A y C/A. Es descriptivo (depende de la carga del equipo).
6. Mecanismo en B: ΔR² de la profundidad sobre log σ_q de la capa 2, con la definición de `../exp_sigma_profundidad.py` (regresión de log σ por cabeza sobre símbolo + 10 deciles de posición, con y sin profundidad; posiciones t = 0, …, largo − 2). Medida por semilla: S_q2 = máximo de ΔR² en las 4 cabezas de σ_q de la capa 2, en DD. Se calcula igual en A. **Se conserva** si la media de S_q2 en B ≥ 0,10 (umbral de la prueba σ–profundidad). Se informan además la diferencia B − A (t pareado), la media en las 4 cabezas y el mismo cálculo en FD.

## 5. Se informa
`close_acc` por semilla, brazo y rango; todas las diferencias por semilla; σ final aprendido en la capa 1 de B (y en ambas capas de C) por cabeza y rol; verificación de equivalencia; parámetros; tiempos. Ninguna semilla se excluye. Si un entrenamiento produce NaN, se informa y su par queda fuera del primario como desviación.

## 6. Ejecución
`/Users/eduardo/asreview/venv/bin/python -B exp_simplificacion.py` → `log_simplificacion.txt`, `resultados_simplificacion.json`; resumen en `RESULTADO_SIMPLIFICACION.md`. Antes del sello solo corre `--humo` (1 semilla, 256 secuencias, 1 época): formas, ausencia de NaN, equivalencia, conteo de parámetros, que el bucle copiado reproduzca el importado y que el ΔR² sea calculable, sin mirar exactitudes.
