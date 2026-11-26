# Prerregistro — ¿Qué explica la asimetría σ_q↓ / σ_k↑ de la capa 2?

29-oct-2026 · uso interno · script `exp_asimetria.py` · se sella en `SELLO_ASIMETRIA.txt` antes de la corrida completa. Antes del sello solo se corrió la prueba de humo (formas, NaN, equivalencias y descripción de los datos; ningún estadístico de resultado).

## Hallazgo a explicar

En Dyck-2, capa 2, al aumentar la profundidad de anidamiento la σ de las consultas baja (×0,69 por nivel) y la de las claves sube (×1,18), en 20/20 cabezas×semilla (`../RESULTADO_SIGMA.md`, `../confirmatorias/RESULTADO_CONFIRMATORIAS.md`).

**Hipótesis de alcance.** Con d = √2·arccosh(1 + (‖Δμ‖²/2 + (σ_q − σ_k)²)/(2σ_qσ_k)), la clave más cercana a una consulta (σ_q, separación ‖Δμ‖) tiene ancho σ′* = √(σ_q² + ‖Δμ‖²/2). La clave se ensancharía para ser alcanzable por su consulta-pareja; la consulta se estrecharía para seleccionar con precisión.

## Común a las tres pruebas

- Atención Fisher–Rao isotrópica del paper (`IsotropicHyperbolicAttention`, solo lectura). Entrenamiento §4.1 idéntico: d_model 128, 2 capas, 4 cabezas, 10.000 secuencias de 20–60, AdamW lr 5·10⁻⁴, wd 0,01, 5 épocas, lote 64. Evaluación: 2.000 secuencias de 20–60 (DD, semilla de datos 123) y de 80–120 (FD, 456); entrenamiento con semilla de datos 42.
- **Semillas nuevas 801–805** (n = 5 por tarea). Capa 2 = segunda capa (`layers[1]`). 4 cabezas × 5 semillas = 20 casos.
- Dyck-2 se entrena con `entrenar` de `../exp_sigma_profundidad.py`, importado sin modificar. Las tareas nuevas usan `entrenar_tarea`, copia del mismo bucle con conjunto de datos y vocabulario como parámetros. En la prueba de humo, `entrenar_tarea("dyck2")` produjo parámetros idénticos bit a bit a `entrenar`.
- Tres endpoints primarios, cada uno a α = 0,05, sin corrección entre pruebas (preguntas distintas). Las secundarias son descriptivas y sin corrección. Los 20 casos no son independientes (4 cabezas por semilla).

## Prueba 1 — ¿La clave está al ancho que maximiza su alcance hacia su pareja?

Modelos Dyck-2 (801–805), conjunto DD.

- **Pares estructurales (sin usar la atención).** Para cada posición t con t+1 un cierre (las posiciones de `close_acc`), o = apertura sin cerrar más reciente tras leer t (la que se cierra en t+1). Cada apertura cerrada es pareja de exactamente una t. Profundidad d = tamaño de la pila tras leer t (= profundidad de o).
- Por cabeza: σ′*(t,o) = √(σ_q(t)² + ‖μ_q(t) − μ_k(o)‖²/2), con ‖·‖ sobre las 32 dimensiones de la cabeza; se compara con σ_k(o).
- **Covariables Z:** profundidad como categorías (el tope es el mayor d con ≥ 30 pares de profundidad ≥ d; los niveles superiores se agrupan en él), símbolo de o, símbolo de t, deciles de posición de o y deciles de posición de t.
- **ρ_pareja** = Spearman entre los residuos de log σ_k(o) y de log σ′*(t,o), tras regresión lineal (MCO) sobre Z.
- **Control obligatorio.** Para cada par i se elige al azar (RandomState(2026), uniforme) otro par j ≠ i con la misma profundidad cruda, y t′ = t_j (consulta no pareja de o). σ′*_control = √(σ_q(t′)² + ‖μ_q(t′) − μ_k(o)‖²/2). **ρ_control** es el mismo estadístico, con Z usando el símbolo y la posición de t′. Los pares cuya profundidad no tiene otro par se excluyen de ambos estadísticos.
- **Endpoint primario:** ρ_pareja > 0 en ≥ 16/20 casos (binomial unilateral p = 0,0059).
- **Especificidad:** ρ_pareja > ρ_control en ≥ 16/20.
- **Veredicto:** CONFIRMADA si se cumplen el primario y la especificidad. ASOCIACIÓN NO ESPECÍFICA DE PAREJA si solo se cumple el primario. NO CONFIRMADA si falla el primario. Se califica de «débil» si la mediana de ρ_pareja es < 0,10.
- **Secundarias:** (a) ρ sin parcializar (pareja y control). (b) Calibración: mediana de σ_k(o)/σ′*(t,o) por caso; «calibrada» si cae en [0,8; 1,25] en ≥ 16/20. (c) Atención (pesos recalculados con la fórmula del módulo; error de reconstrucción en la prueba de humo = 0): peso medio de t sobre o frente al peso medio por cada otra apertura ≤ t, y fracción de t en que o recibe el máximo entre las aperturas ≤ t. (d) Control en la misma secuencia: t′ con la misma profundidad en la misma secuencia (RandomState(2027)), en el subconjunto de pares con candidato. (e) Prueba 1 en FD. (f) Prueba 1 en los modelos de la Prueba 3.

## Prueba 2 — ¿La asimetría importa?

Modelos Dyck-2 (801–805), sin reentrenar. Capa 2, todas las cabezas, todas las posiciones. Ganchos sobre `sg_q` y `sg_k`: desde la misma entrada se calculan σ_q y σ_k del mismo token, y se devuelve softplus⁻¹(σ_nuevo − ε).

- (0) **Identidad:** σ ← σ. (i) **Intercambio:** σ_q ← σ_k y σ_k ← σ_q. (ii) **Simetrización:** ambas ← √(σ_qσ_k).
- Se mide `close_acc` (del paper) en DD y FD: sin intervención, con (0), (i) y (ii).
- **Requisito de validez:** |exactitud(0) − exactitud base| ≤ 10⁻⁴ en las 10 mediciones; si no, la prueba es INVÁLIDA. En la prueba de humo la diferencia fue 0 (máx |Δlogit| ≈ 10⁻⁶).
- **Endpoint primario:** caída con (i) en FD = base − (i). Media > 0 y t pareado bilateral p < 0,05 (n = 5). CONFIRMADA o NO CONFIRMADA.
- **Secundarias:** (ii) frente a la base; (i) frente a (ii); las mismas comparaciones en DD; las mismas en los modelos de la Prueba 3.

## Prueba 3 — ¿Aparece en otras jerarquías?

Mismo bucle, semillas 801–805, dos tareas nuevas:

- **(a) Dyck-4:** generador del paper (`exp1_profundidad.gen_dyck4`, solo lectura), con RandomState(semilla de datos). Abre 1, 3, 5, 7; cierra 2, 4, 6, 8; vocabulario 9. Largos de §4.1 (no los 50–100 del experimento de Dyck-4 del paper). Con las mismas semillas de datos, consume el generador igual que Dyck-2: produce la misma secuencia de aperturas y cierres (misma distribución de profundidades) y solo cambia el tipo de paréntesis.
- **(b) Dyck-2 profundo:** generador Dyck-2 del paper con probabilidad de apertura 0,65 en vez de 0,5 (con 0,5 reproduce el original exactamente; verificado). Distribución resultante, medida antes del sello en los conjuntos que se usarán:

| Conjunto | Máx. por secuencia: mediana (p10–p90; máx.) | Por posición: mediana; p95 | Posiciones con prof. ≥ 10 |
|---|---|---|---|
| Dyck-2 estándar DD (20–60) | 6 (3–10; 19) | 2; 8 | 2,8 % |
| Dyck-2 profundo DD (20–60) | 10 (6–17; 25) | 5; 14 | 19,4 % |
| Dyck-2 estándar FD (80–120) | 11 (6–18; 31) | 4; 13 | 14,7 % |
| Dyck-2 profundo FD (80–120) | 25 (17–33; 44) | 12; 27 | 59,2 % |

Dyck-4 tiene las mismas cifras que Dyck-2 estándar.

- **Medida:** log σ de todos los tokens t = 0…largo−2 en DD. Regresión log σ ~ 1 + símbolo (dummies) + 9 dummies de deciles de posición + profundidad (lineal). Es la de `delta_r2`, generalizada; en Dyck-2 reproduce exactamente `medir`/`delta_r2`.
- **Umbral de exactitud (fijado ahora):** `close_acc` en DD (argmax entre los cierres de la tarea; en Dyck-2 coincide con la del paper) ≥ 0,80 por semilla. Las semillas bajo el umbral se excluyen. La tarea se informa como NO EVALUABLE si quedan < 4 semillas; con 4 semillas el criterio pasa a ≥ 13/16 (p = 0,011).
- **Endpoint primario (por tarea):** coeficiente de profundidad de σ_q de la capa 2 < 0 en ≥ 16/20 y de σ_k > 0 en ≥ 16/20 (binomial unilateral p = 0,0059 cada uno). CONFIRMADA si ambos; PARCIAL si uno; NO CONFIRMADA si ninguno.
- **Secundarias:** factor por nivel exp(coef) con IC 95 % sobre semillas; coeficientes en FD; modelos sin entrenar (control); ΔR² de la profundidad; perfil por nivel (dummies de profundidad, referencia nivel 1, tope 10); exactitud DD/FD de cada modelo. Como referencia, los mismos coeficientes en los modelos Dyck-2 801–805.

## Productos

`log_asimetria.txt`, `resultados_asimetria.json`, `RESULTADO_ASIMETRIA.md` (≤ 1 página; lo no prerregistrado se rotula *post hoc*).
