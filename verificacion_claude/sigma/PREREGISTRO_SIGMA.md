# Prerregistro — ¿Qué aprende la varianza σ en la atención Fisher–Rao isotrópica?

Fecha: 24-oct-2026. Uso interno. Fuente del diseño: `PLAN_QUE_APRENDE_SIGMA.md`. Este documento se escribe y se sella (SHA-256, `SELLO_SIGMA.txt`) **antes** de entrenar ningún modelo de la prueba. Nada de lo fijado aquí se cambia después de ver datos; todo análisis adicional se rotula *post hoc*; se informan todos los resultados, favorables o no.

## 1. Modelo, datos y entrenamiento
- Código del paper, solo lectura: `Dyck2Dataset`, `collate_dyck`, `Model`, `close_acc` de `exp_tres_geometrias.py`; `IsotropicHyperbolicAttention` de `fast_gaussian_test.py` (σ_q = softplus(sg_q(x)) + 0,05 y σ_k = softplus(sg_k(x)) + 0,05; una σ escalar por token y cabeza).
- `run_one` no devuelve el modelo; su bucle se replica idéntico en `exp_sigma_profundidad.py`: `torch.manual_seed(semilla)`, `np.random.seed(semilla)`; entrenamiento 10.000 secuencias Dyck-2 de largo 20–60 (semilla de datos 42), lote 64, AdamW lr 5·10⁻⁴, wd 0,01, 5 épocas, `CrossEntropyLoss(ignore_index=0)`; d_model 128, 2 capas, 4 cabezas.
- Semillas de inicialización: 401, 402, 403, 404, 405 (n = 5).
- Conjunto de medición dentro de distribución (DD): las 2.000 secuencias de validación del código (largo 20–60, semilla de datos 123).
- Conjunto fuera de distribución (FD): las 2.000 secuencias de evaluación extrapolativa del código (largo 80–120, semilla de datos 456).
- Control de calidad del entrenamiento: `close_acc` en DD y FD por semilla. Referencia del brazo isotrópico en el código del paper: ≈ 98 % DD y ≈ 88 % FD. Se informa; no es criterio de exclusión.

## 2. Extracción de σ
- Entrada al modelo: `seq[:, :-1]`, igual que en entrenamiento y en `close_acc`.
- Ganchos (`register_forward_hook`) sobre `layers[l].attn.sg_q` y `layers[l].attn.sg_k`, l ∈ {0, 1}. Salida [B, L, 4]; σ = softplus(salida) + `eps_sigma` (0,05). Variable respuesta: log σ.
- 16 combinaciones = 2 capas × 4 cabezas × 2 roles (q, k).
- Posiciones analizadas: t = 0, …, largo − 2 de cada secuencia (las posiciones con objetivo de predicción siguiente; se excluye el relleno).

## 3. Covariables
- Símbolo: token en t, 4 categorías `(`, `)`, `[`, `]` (indicadoras con intercepto).
- Posición: t absoluta (índice 0) agrupada en 10 tramos por deciles calculados sobre todas las posiciones analizadas del conjunto (DD y FD por separado; los mismos tramos para modelo entrenado y sin entrenar). Indicadoras con intercepto.
- Profundidad: número de paréntesis abiertos en la pila **después** de leer el token t (cuenta el propio token: una apertura lo incluye; un cierre ya lo descontó). Término lineal. Nota: como profundidad-después = profundidad-antes ± 1 según el símbolo y el símbolo está en el modelo, ΔR² y el signo del coeficiente son idénticos con cualquiera de las dos definiciones.

## 4. Estadístico
- Por combinación: ΔR² = R²(log σ ~ símbolo + posición + profundidad) − R²(log σ ~ símbolo + posición), mínimos cuadrados ordinarios, R² sobre la media.
- Por semilla: **S = máximo de ΔR² en las 16 combinaciones**, calculado en DD.
- Control: el mismo modelo **sin entrenar** (pesos iniciales de la misma semilla, copiados antes del primer paso de optimización), con S calculado igual.

## 5. Endpoint primario (DD)
Hipótesis «σ codifica la profundidad» **confirmada** si se cumplen ambas:
1. media de S entrenado (5 semillas) ≥ 0,10, y
2. S entrenado > S sin entrenar: t pareado bilateral (gl = 4) con p < 0,05 y diferencia media positiva.
Si falla cualquiera: **no confirmada**. Si se cumple 2 pero no 1, se informa como «señal mayor que el azar pero bajo el umbral prerregistrado» (no confirmada).

## 6. Predicción de dirección
σ disminuye con la profundidad. **Apoyada** si el coeficiente de profundidad de la combinación que alcanza S es negativo en ≥ 4 de 5 semillas (modelos entrenados, DD). Se informa además la fracción de las 16 combinaciones con coeficiente negativo por semilla (media y rango) y el coeficiente (unidades de log σ por nivel de profundidad).

## 7. Secundarias
- **Q3, σ_q vs σ_k:** por semilla, ΔR² medio de las 8 combinaciones de cada rol (entrenado, DD); diferencia q − k con t pareado bilateral (n = 5). Se informa también el rol de la combinación que alcanza S.
- **Extensión a secuencias más largas (FD):** (a) S_FD (máximo de las 16 en FD) entrenado vs sin entrenar, con el mismo criterio del §5; (b) sin selección en FD: ΔR² en FD de la misma combinación que alcanzó S en DD, y su signo.

## 8. Exploratoria (rotulada exploratoria)
- **Q2:** ¿σ distingue las posiciones donde el modelo se equivoca? Posiciones cuyo objetivo es un cierre (las que mide `close_acc`); error = la preferencia del modelo entre `)` y `]` no coincide con el objetivo. Por combinación, AUC de log σ para error; por semilla se informa la combinación con mayor |AUC − 0,5| y su AUC, en DD y FD, con el número de errores. Referencia: AUC de la profundidad sola. Sin umbral de decisión.

## 9. Ejecución
`/Users/eduardo/asreview/venv/bin/python -B exp_sigma_profundidad.py` → `log_sigma.txt`, `resultados_sigma.json`. Antes del sello solo se ejecuta una prueba de humo (`--humo`: 1 semilla, pocas secuencias, 1 época) que imprime formas y ausencia de NaN, sin ΔR².
