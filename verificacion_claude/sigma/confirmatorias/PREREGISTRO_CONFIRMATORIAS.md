# Prerregistro — Pruebas confirmatorias 3 y 5 sobre σ (atención Fisher–Rao isotrópica, Dyck-2)

Fecha: 25-oct-2026. Uso interno. Este documento y el script `exp_confirmatorias.py` se sellan (SHA-256, `SELLO_CONFIRMATORIAS.txt`) **antes** de entrenar los modelos de estas pruebas. Antes del sello solo se ejecuta una prueba de humo (`--humo`: 1 semilla, pocas secuencias, 1 época) que imprime formas y ausencia de NaN, sin coeficientes ni AUC. Nada de lo fijado aquí se cambia después de ver datos; todo análisis adicional se rotula *post hoc*; se informan todos los resultados.

## 1. Modelo, entrenamiento y medición (comunes)
- Se reutiliza sin modificar `../exp_sigma_profundidad.py` (importado): `entrenar` (réplica idéntica del bucle de §4.1: Dyck-2, 10.000 secuencias de 20–60, semilla de datos 42, lote 64, AdamW lr 5·10⁻⁴, wd 0,01, 5 épocas; d_model 128, 2 capas, 4 cabezas, `IsotropicHyperbolicAttention`), `medir` (ganchos sobre `sg_q`/`sg_k`, log σ con σ = softplus(·) + 0,05, posiciones t = 0…largo − 2), `delta_r2` y `tramos_posicion` (10 tramos por deciles de t, calculados por conjunto).
- **Semillas nuevas de inicialización: 601, 602, 603, 604, 605** (n = 5). No se usan 401–405.
- DD: 2.000 secuencias de 20–60 (semilla de datos 123). FD: 2.000 secuencias de 80–120 (semilla de datos 456). Son los mismos conjuntos de las pruebas anteriores; cambia solo la inicialización.
- Control de calidad: `close_acc` en DD y FD por semilla (se informa, no excluye).
- «Capa 2» = segunda capa del modelo (índice 1 en el código, `capa1_*`).
- Profundidad: paréntesis abiertos después de leer el token t (definición de `medir`).

## 2. Prueba 3 — las claves se ensanchan y las consultas se estrechan con la profundidad
Hipótesis (formulada *post hoc* en las semillas 401–405; aquí se confirma en semillas nuevas): en la capa 2, el coeficiente de profundidad de log σ_k es positivo y el de log σ_q es negativo.

- Medida por cabeza y rol: coeficiente de la profundidad (término lineal) en la regresión MCO log σ ~ símbolo (4 categorías) + posición (10 tramos) + profundidad, en todas las posiciones analizadas de DD (salida `coef_prof` de `delta_r2`).
- Casos: 4 cabezas de la capa 2 × 5 semillas = 20 por rol.
- **Endpoint primario (DD):** CONFIRMADA si se cumplen ambas: (a) coeficiente de σ_k > 0 en ≥ 18/20 y (b) coeficiente de σ_q < 0 en ≥ 18/20. Para cada uno se informa la prueba binomial unilateral frente a 0,5 (H₁: proporción > 0,5), y ambas deben dar p < 0,05. Si se cumple solo una, NO CONFIRMADA (parcial: se dice cuál).
- Secundarias: (i) por semilla, media de los 4 coeficientes de cada rol; media de las 5 semillas con IC 95 % t (gl = 4, la semilla como unidad); (ii) lo mismo (conteos, binomial, medias) en FD, con tramos de posición calculados en FD; (iii) control descriptivo: conteos de signo en los modelos sin entrenar (pesos iniciales de la misma semilla).

## 3. Prueba 5 — σ_k de la capa 2 como señal de error, sin selección
- **Posiciones evaluadas:** exactamente las de `close_acc`: las t cuyo objetivo `seq[t+1]` es un cierre (`)` o `]`). Error = la preferencia del modelo en t entre `)` y `]` (logit de `)` ≥ logit de `]` → `)`) no coincide con el objetivo.
- **Medida única, fijada de antemano:** s(t) = media aritmética de σ_k (en unidades de σ, no de log) sobre las 4 cabezas de la capa 2, en el token de entrada t (el token cuya salida predice t+1, el mismo que mide `close_acc`).
- **Sentido fijado de antemano** (de `resultados_sigma.json`, semillas 401–405, sin recalcular): las 4 cabezas de σ_k de la capa 2 tuvieron AUC > 0,5 en 5/5 semillas en DD (0,68–0,87) y en FD (0,54–0,79): **mayor σ_k → error**. AUC = `roc_auc_score(error, s)`, sin invertir.
- **Endpoint primario (DD, 20–60):** CONFIRMADA si se cumplen ambas: (a) AUC media de las semillas incluidas ≥ 0,70 y (b) el IC 95 % percentil por bootstrap de posiciones (2.000 réplicas, remuestreo con reemplazo de las posiciones evaluadas, generador `numpy.random.default_rng(20261025 + semilla)`) tiene límite inferior > 0,5 en ≥ 4 de las 5 semillas.
- **Regla de exclusión (fijada antes):** una semilla con < 30 errores en un régimen se excluye de ese régimen; no entra en la media y cuenta como «no cumple» en el conteo de 4/5. Si quedan < 3 semillas incluidas, el endpoint es «no evaluable».
- **Controles obligatorios (secundarios, mismo sentido «mayor valor → error», sin invertir):** AUC de la profundidad sola; AUC de la posición sola (t absoluta); **AUC de σ_k ajustada**: residuo MCO de log s(t) sobre profundidad (lineal) + posición (10 tramos de §1) + símbolo (4 categorías), ajustado en las posiciones evaluadas del régimen y semilla; se informa su AUC, IC bootstrap por posiciones y el número de semillas con límite inferior > 0,5.
- Secundaria adicional (información incremental): regresión logística (scikit-learn, C = 1,0, lbfgs, `max_iter` = 2.000) con validación cruzada de 5 particiones agrupadas por secuencia (`GroupKFold`); modelo A = símbolo + tramos de posición + profundidad como categorías (indicadoras por valor); modelo B = A + log s(t). Se informa la AUC fuera de partición de A y de B, ΔAUC = B − A por semilla y cuántas semillas tienen ΔAUC > 0.
- Sensibilidad: IC 95 % por bootstrap de secuencias (remuestreo de secuencias completas, 2.000 réplicas) para la AUC primaria.
- Secundaria: todo lo anterior en FD (80–120), con la misma regla de exclusión.

## 4. Ejecución y salidas
`/Users/eduardo/asreview/venv/bin/python -B exp_confirmatorias.py` (en `pruebas/sigma/confirmatorias/`) → `log_confirmatorias.txt`, `resultados_confirmatorias.json`; luego `RESULTADO_CONFIRMATORIAS.md`. Código del paper solo lectura.

## 5. Límites conocidos de antemano
Los 20 casos de la Prueba 3 no son independientes (4 cabezas por semilla); la binomial los trata como tales. Los conjuntos de datos son los mismos de las pruebas previas: se replica sobre inicializaciones nuevas, no sobre datos nuevos. Una tarea sintética y un modelo pequeño.
