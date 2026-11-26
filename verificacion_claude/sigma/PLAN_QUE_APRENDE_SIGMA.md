# Plan — ¿Qué aprende la varianza σ? (borrador para prerregistrar)

Fecha: 24-oct-2026. Uso interno; no entra al paper sin decisión de los autores.

## Pregunta principal (Q1): ¿σ codifica la profundidad de anidamiento?
- Tarea y modelo: Dyck-2 con la configuración de §4.1 (d_model 128, 2 capas, 4 cabezas, 10.000 secuencias de largo 20–60, AdamW lr 5·10⁻⁴, wd 0,01, 5 épocas, lote 64), brazo Fisher–Rao isotrópico (`IsotropicHyperbolicAttention`, `fast_gaussian_test.py`), entrenado con `run_one` / `Model` / `Dyck2Dataset` de `exp_tres_geometrias.py` (importar sin modificar). Semillas nuevas 401–405.
- Medición: en 2.000 secuencias de validación, extraer σ_q y σ_k por token, capa y cabeza (16 combinaciones), con la profundidad de pila de cada token, su símbolo y su posición.
- Estadístico: ΔR² de agregar la profundidad a una regresión de log σ sobre símbolo (4 categorías) + posición (10 tramos). Por semilla, S = máximo de ΔR² en las 16 combinaciones.
- Control: el mismo modelo sin entrenar (misma semilla de inicialización).
- Endpoint primario (a fijar): media de S entrenado ≥ 0,10 y mayor que sin entrenar, t pareado bilateral, p < 0,05, n = 5.
- Predicción de dirección: σ disminuye con la profundidad (lo general arriba, lo específico abajo).

## Secundarias
- Q3: ¿σ_q o σ_k lleva más información de profundidad? (ΔR² medio por rol).
- Extensión fuera del largo de entrenamiento: ¿la relación se mantiene en secuencias más largas?
- Q2 (exploratoria): ¿la σ de las cabezas distingue las posiciones donde el modelo se equivoca? (AUC).

## Costo previsto
Entrenamiento y medición en CPU, minutos por semilla.
