# Prerregistro — Intervención sobre σ: ¿el modelo usa la profundidad codificada en σ_q?

Fecha: 25-oct-2026 (UTC). Uso interno. Antecedente: `../PREREGISTRO_SIGMA.md` y `../RESULTADO_SIGMA.md` (σ_q de la capa 2 codifica la profundidad en Dyck-2; ΔR² ≈ 0,58). Este documento y el script `exp_sigma_intervencion.py` se sellan (SHA-256, `SELLO_INTERVENCION.txt`) **antes** de evaluar cualquier intervención. Nada de lo fijado aquí se cambia después de ver datos; todo análisis adicional se rotula *post hoc*; se informan todos los resultados.

## 1. Modelos
- Reentrenamiento determinista con `entrenar` importado sin cambios de `../exp_sigma_profundidad.py` (sellado en `../SELLO_SIGMA.txt`): Dyck-2, configuración §4.1, brazo Fisher–Rao isotrópico, semillas 401–405, `torch.set_num_threads(8)`. Sin reentrenar después: todas las intervenciones son en evaluación.
- Verificación de determinismo: `close_acc` sin intervención (función del paper) en DD y FD se compara con `../resultados_sigma.json`. Si alguna semilla difiere en más de 10⁻⁹, se informa como desviación y se sigue con el modelo reentrenado (mismo protocolo).

## 2. Conjuntos
- Evaluación: DD = 2.000 secuencias de largo 20–60 (semilla de datos 123); FD = 2.000 de largo 80–120 (semilla 456). Son los conjuntos de la prueba anterior.
- Calibración, aparte y nunca usada para medir exactitud: CAL-DD = 2.000 secuencias de largo 20–60 (semilla 777); CAL-FD = 2.000 de largo 80–120 (semilla 888). Cada régimen se interviene con lo ajustado en su propia calibración (DD con CAL-DD; FD con CAL-FD), porque los tramos de posición dependen del largo.

## 3. Intervención
- Sitios (uno por vez; las 4 cabezas del sitio a la vez): **A = σ_q de la capa 2 (primario)**; B = σ_k de la capa 2; C = σ_q de la capa 1 (control: la capa 1 no ve la profundidad por construcción).
- Mecánica: gancho de salida sobre `sg_q`/`sg_k` del sitio que reemplaza su salida por softplus⁻¹(σ̃ − 0,05), de modo que la atención usa exactamente σ̃. Se reemplaza en todas las posiciones con token real; el relleno queda intacto (no afecta las predicciones medidas por la máscara causal). Escala de trabajo: log σ. Tope: σ̃ ≥ 0,05 + 10⁻⁴ (se informa la fracción topada).
- Variables (definiciones de la prueba anterior): símbolo del token t (4 categorías); tramo de posición (10 deciles de las posiciones de la calibración, aplicados al conjunto evaluado); profundidad = paréntesis abiertos **después** de leer t (término lineal).
- Manipulaciones, ajustadas por cabeza con mínimos cuadrados en la calibración (posiciones t = 0, …, largo − 2):
  - **M0** sin intervención (base).
  - **M1 «sin profundidad»**: log σ̃ = media de log σ por símbolo (regresión sobre símbolo; elimina la profundidad y conserva el símbolo).
  - **M2 «con profundidad»**: log σ̃ = predicción desde símbolo + tramo de posición + profundidad (conserva la profundidad; elimina el resto).
  - **M3 (secundaria) «símbolo + posición»**: predicción desde símbolo + tramo de posición, sin profundidad. Separa el aporte de la profundidad del de la posición.
  - Control del procedimiento: «identidad» (reemplazo por el mismo σ, ida y vuelta por softplus⁻¹) en cada sitio; debe reproducir la exactitud base (|Δ| ≤ 0,001) o se informa.
- Fidelidad: R² de log σ̃ frente al log σ original en las posiciones evaluadas, por cabeza (media), se informa.

## 4. Medida
`close_acc`: posiciones cuyo objetivo es un cierre; acierto si la preferencia entre `)` y `]` coincide con el objetivo (lógica idéntica a la función del paper). Caída(Mk) = acc(M0) − acc(Mk).

## 5. Endpoint primario (sitio A, FD 80–120)
Por semilla, Δ = caída(M1) − caída(M2) = acc(M2) − acc(M1). **Confirmada** («el modelo usa la información de profundidad que lleva σ_q de la capa 2») si la media de Δ > 0 y el t pareado bilateral (gl = 4, n = 5) da p < 0,05. Si no: **no confirmada**. Lectura fijada de antemano: si además caída(M1) es pequeña (media < 0,01), no hay evidencia de que el modelo dependa de σ_q de la capa 2; si caída(M1) y caída(M2) son ambas grandes y similares, σ_q se usa pero la manipulación no aísla la profundidad.

## 6. Secundarias (no decisorias; mismo estadístico, sin corrección por multiplicidad)
1. Sitio A en DD (20–60).
2. Sitio B (σ_k, capa 2) en FD y en DD.
3. Sitio C (σ_q, capa 1), control, en FD y en DD. Predicción: Δ ≈ 0 (sin profundidad que eliminar); un Δ distinto de 0 solo podría venir del término de posición.
4. M3 frente a M2 en el sitio A (FD y DD): caída(M3) − caída(M2), aporte específico de la profundidad.

## 7. Se informa
Exactitud por semilla, régimen, sitio y manipulación; caídas; fidelidad; fracción topada; verificación de determinismo y del control de identidad.

## 8. Ejecución
`/Users/eduardo/asreview/venv/bin/python -B exp_sigma_intervencion.py` → `log_intervencion.txt`, `resultados_intervencion.json`. Antes del sello solo corre `--humo` (1 semilla, pocas secuencias, 1 época): formas, ausencia de NaN y control de identidad, sin exactitudes de las manipulaciones.
