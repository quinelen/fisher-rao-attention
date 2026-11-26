# Prerregistro — ¿σ codifica el anidamiento en código Python real?

Fecha: 25-oct-2026 (UTC). Uso interno. Antecedente: `../PREREGISTRO_SIGMA.md` y `../RESULTADO_SIGMA.md` (en Dyck-2, σ_q de la capa 2 codifica la profundidad de anidamiento). Este documento y el script `exp_sigma_codigo.py` se sellan (SHA-256, `SELLO_CODIGO.txt`) **antes** de entrenar ningún modelo de la prueba. Nada de lo fijado aquí se cambia después de ver datos; todo análisis adicional se rotula *post hoc*; se informan todos los resultados.

## 1. Modelo, datos y entrenamiento
- Código del paper, solo lectura: `StableFR` de `exp_estabilizado.py` (Fisher–Rao isotrópico con ‖μ‖ ≤ 12) dentro de `Model` de `exp_tres_geometrias.py`; `build_corpus`, `make_vocab`, `encode` de `exp_codigo_python.py`; `seqs_of`, `bpc_zones` y constantes de `exp_codigo_extrapolativo.py`.
- Corpus: `.py` de la biblioteca estándar del intérprete del entorno (CPython 3.9.6), split 90/10 por archivo: 1.195.931 caracteres de entrenamiento, 37.771 de prueba, vocabulario 97 (verificado antes de este documento; el texto de prueba no tiene tabulaciones).
- Secuencias, exactamente como en `exp_codigo_extrapolativo.py`/`exp_estabilizado.py`: `RandomState(0)`; 4.000 ventanas de entrenamiento de largo 128; luego 400 ventanas de prueba de largo 128, 256 y 512, en ese orden. Se analizan 128 (primaria) y 512 (secundaria). El script recalcula los inicios de las ventanas con la misma secuencia aleatoria y verifica que coinciden.
- Entrenamiento: réplica idéntica del bucle `run` (`torch.manual_seed(semilla)`, `RandomState(semilla)` para las permutaciones, AdamW lr 5·10⁻⁴, wd 0,01, 3 épocas, lote 32, `CrossEntropyLoss(ignore_index=0)`), d_model 128, 2 capas, 4 cabezas, `max_len` 512, `torch.set_num_threads(8)`. Semillas nuevas 501–505 (n = 5).
- Control de calidad (se informa; no excluye): bpc en L = 128 y en la zona extrapolativa de L = 512 (`bpc_zones`). Referencia FR estabilizado, semillas 16–35: 3,30 y 3,74.
- Control: el mismo modelo **sin entrenar** (pesos iniciales de la misma semilla, copiados antes del primer paso).

## 2. Extracción de σ
Ganchos sobre `sg_q` y `sg_k` de las dos capas; σ = softplus(salida) + 0,05; respuesta log σ. Entrada `ventana[:-1]`; posiciones analizadas t = 0, …, L − 2. 16 combinaciones = 2 capas × 4 cabezas × 2 roles.

## 3. Covariables
- Símbolo: los 30 caracteres más frecuentes del texto de entrenamiento + «otro» (31 categorías; «otro» es la referencia).
- Posición: t en 10 tramos por deciles de las posiciones analizadas de cada conjunto (los mismos para el modelo entrenado y el sin entrenar).
- **Indentación (primaria)**: espacios iniciales de la línea a la que pertenece el carácter, divididos por 4 (valor real: las líneas de continuación pueden dar fracciones). Una línea va desde el carácter siguiente a un salto de línea (o el inicio del texto) hasta su salto de línea inclusive. **Inclusión:** solo caracteres cuya línea completa está dentro de las posiciones analizadas de la ventana (inicio de la línea ≥ inicio de la ventana y salto final ≤ posición L − 2). El resto se excluye del análisis de indentación; se informa la fracción excluida.
- **Profundidad de paréntesis (secundaria)**: `(`, `[`, `{` abiertos dentro de la ventana y aún no cerrados después de leer t; conteo carácter a carácter, sin distinguir cadenas ni comentarios; un cierre sin apertura visible se ignora. Se calcula en todas las posiciones analizadas.

## 4. Estadístico
ΔR² = R²(log σ ~ símbolo + posición + variable) − R²(log σ ~ símbolo + posición), mínimos cuadrados ordinarios, término lineal. Por semilla, **S = máximo de ΔR² en las 16 combinaciones**, en L = 128.

## 5. Endpoint primario (indentación, L = 128)
**Confirmada** si: (1) media de S entrenado ≥ 0,05 y (2) S entrenado > S sin entrenar, t pareado bilateral (gl = 4) con p < 0,05 y diferencia media positiva. Si se cumple (2) y no (1): «señal mayor que el azar pero bajo el umbral prerregistrado» (no confirmada). En otro caso: no confirmada. El umbral es la mitad del de Dyck-2 (0,10) por decisión previa: el código es ruidoso, el símbolo explica gran parte de la varianza de σ y la indentación es un correlato más grueso del anidamiento que la profundidad de pila.

## 6. Secundarias (no decisorias)
1. **Dirección:** signo del coeficiente de indentación en la combinación de S. Predicción por analogía con Dyck-2: negativo si la combinación es σ_q, positivo si es σ_k; apoyada si coincide en ≥ 4 de 5 semillas. Se informa además, en la capa 2, cuántas de las 20 cabezas×semilla tienen coeficiente negativo, por rol.
2. **σ_q vs σ_k:** ΔR² medio de las 8 combinaciones de cada rol (entrenado); diferencia q − k, t pareado bilateral.
3. **L = 512:** (a) S en L = 512 entrenado frente a sin entrenar, con el criterio del §5; (b) sin reselección, ΔR² en L = 512 de la combinación que alcanzó S en L = 128, y su coeficiente. Las posiciones 128–511 no se vieron en entrenamiento.
4. **Profundidad de paréntesis (L = 128):** S_par entrenado frente a sin entrenar, con el criterio del §5, y el signo de su coeficiente.
5. **Sensibilidad:** ΔR² de la indentación agregando al modelo base la columna (posición del carácter dentro de su línea, 10 tramos por deciles), en la combinación de S y como máximo de las 16. Motivo: la indentación se correlaciona con la columna.

## 7. Ejecución
`/Users/eduardo/asreview/venv/bin/python -B exp_sigma_codigo.py` → `log_codigo.txt`, `resultados_codigo.json`. Antes del sello solo corre `--humo` (1 semilla, 64 ventanas, 1 época): formas, ausencia de NaN y conteos de inclusión, sin ΔR².
