# Resultado — ¿Qué explica la asimetría σ_q↓ / σ_k↑? (tres pruebas prerregistradas)

29-oct-2026 · uso interno · protocolo `PREREGISTRO_ASIMETRIA.md`, sellado en `SELLO_ASIMETRIA.txt` (18:16 UTC) antes de la corrida · datos: `log_asimetria.txt`, `resultados_asimetria.json`.

Semillas nuevas 801–805, entrenamiento §4.1 idéntico. Los 15 modelos superan el umbral de exactitud (0,80): `close_acc` DD/FD de 97,3–98,8 % / 87,6–89,1 % en Dyck-2, 97,1–98,7 % / 80,6–82,1 % en Dyck-4 y 98,5–99,4 % / 77,9–79,6 % en Dyck-2 profundo. Reemplazar cada σ por sí misma no cambia ninguna exactitud (diferencia 0).

| Prueba | Endpoint primario | Resultado | Veredicto |
|---|---|---|---|
| 1. Alcance | ρ parcial σ_k(o) ~ σ′*(t,o) > 0 en ≥ 16/20 | 12/20 (p = 0,25); mediana 0,046 | **NO CONFIRMADA** |
| 2. Intercambio σ_q↔σ_k | caída en FD > 0, p < 0,05 | +4,3 pp; t(4) = 2,75; p = 0,051 | **NO CONFIRMADA** (por poco) |
| 3a. Dyck-4 | σ_q < 0 y σ_k > 0 en ≥ 16/20 | 20/20 y 20/20 | **CONFIRMADA** |
| 3b. Dyck-2 profundo | ídem | 16/20 y 20/20 | **CONFIRMADA** (σ_q en el umbral) |

**Prueba 1 — la clave no está al ancho de alcance de su pareja.** El ρ parcial va de −0,37 a 0,61 entre cabezas. Supera al de una consulta no pareja de la misma profundidad en 13/20 (p = 0,13; mediana del control −0,021). Sin parcializar, la mediana es −0,098. Las claves son mucho más estrechas que el ancho óptimo: σ_k/σ′* tiene mediana 0,054 (0,039–0,122) y cae en [0,8; 1,25] en 0/20. Lo mismo ocurre fuera de distribución (9/20 positivos), en Dyck-4 (7/20) y en Dyck-2 profundo (7/20). La atención sí distingue a la pareja: recibe más peso que la apertura media en 20/20 cabezas (mediana 0,16 frente a 0,06) y es la apertura con más peso en el 36 % de las consultas (16–72 %).

**Prueba 2 — el sentido de la asimetría importa, pero el primario queda en p = 0,051.** Intercambiar σ_q y σ_k baja la exactitud en las 5 semillas: 1,7–10,3 pp en FD (media 4,3) y 4,2–13,8 pp en DD (media 7,1; p = 0,015). Simetrizar baja menos y sin significación: 1,7 pp en FD (p = 0,22) y 2,8 pp en DD (p = 0,12). El intercambio daña más que la simetrización: 2,6 pp más en FD (p = 0,004) y 4,3 pp más en DD (p = 0,0006). En los modelos de la Prueba 3 (secundaria), el intercambio baja la exactitud en FD 5,9 pp en Dyck-4 (p = 0,0008) y 1,9 pp en Dyck-2 profundo (p = 0,011).

**Prueba 3 — la asimetría aparece en las otras dos jerarquías.** Factor por nivel de profundidad (IC 95 %, semillas):

| Tarea (DD) | σ_q < 0 | σ_k > 0 | σ_q por nivel | σ_k por nivel | FD: q < 0 / k > 0 |
|---|---|---|---|---|---|
| Dyck-2 (referencia) | 20/20 | 20/20 | ×0,74 (0,72–0,75) | ×1,20 (1,17–1,22) | 19/20 · 20/20 |
| Dyck-4 | 20/20 | 20/20 | ×0,70 (0,68–0,72) | ×1,21 (1,20–1,22) | 20/20 · 20/20 |
| Dyck-2 profundo | 16/20 | 20/20 | ×0,89 (0,86–0,93) | ×1,13 (1,12–1,14) | 15/20 · 20/20 |

En los modelos sin entrenar los coeficientes son ≈ 0 (|media| ≤ 0,001) y el signo cae en 9–12/20. Dyck-4 tiene las mismas profundidades que Dyck-2 y solo cambia el número de tipos de paréntesis. En Dyck-2 profundo, la profundidad máxima por secuencia tiene mediana 10 y el 19 % de las posiciones están a profundidad ≥ 10 (frente a 6 y 2,8 % en Dyck-2).

**Post hoc (no prerregistrado, descriptivo).** (1) Con σ_k ≪ σ′*, la distancia a una clave baja al ensancharla, para toda consulta cuyo σ′* supere a σ_k. Ensanchar las claves profundas las acerca a cualquier consulta lejana, sin ajustarse a su pareja. Los datos son compatibles con un sesgo hacia lo profundo y no con un ajuste fino por pareja; es una hipótesis sin probar. (2) En el perfil por nivel, σ_q cae sobre todo en los primeros niveles y luego se satura (Dyck-2: −0,52 en log σ a nivel 2 y −2,18 a ≥ 10, ambos frente al nivel 1), mientras σ_k sube casi lineal (+0,14 y +1,55). La forma es la misma en las tres tareas. La saturación explicaría la pendiente lineal menor de σ_q en Dyck-2 profundo, donde abundan las posiciones profundas.

**Lectura.** La asimetría q↓/k↑ es robusta: se repite en semillas nuevas, en Dyck-4 y en Dyck-2 profundo. Invertirla daña en todas las semillas y más que borrarla, aunque el primario quedó en p = 0,051. Pero la explicación de «alcance» en su forma calibrada no se sostiene, porque las claves no se sitúan en σ′* ni lo siguen par a par.

**Límites.** Los 20 casos no son independientes (4 cabezas por semilla) y n = 5. La Prueba 1 es correlacional, con covariables lineales. El primario de la Prueba 2 falla por la dispersión entre semillas (una semilla cae 10,3 pp), no por el signo. Son tareas sintéticas con un modelo pequeño, la única medida de desempeño es `close_acc` y las secundarias van sin corrección por multiplicidad.
