# Resultado — Pruebas confirmatorias 3 y 5 sobre σ (Dyck-2, semillas nuevas)

25-oct-2026 · uso interno · protocolo `PREREGISTRO_CONFIRMATORIAS.md`, sellado en `SELLO_CONFIRMATORIAS.txt` (16:42 UTC) antes de la corrida · datos: `log_confirmatorias.txt`, `resultados_confirmatorias.json`.

Modelos: atención Fisher–Rao isotrópica, entrenamiento idéntico al de §4.1, semillas 601–605. `close_acc` fue de 98,4 % dentro de distribución (DD, 20–60; rango 97,6–99,2 %) y de 88,7 % fuera de distribución (FD, 80–120; rango 87,2–89,5 %). Ninguna semilla se excluyó: hubo entre 325 y 961 errores de 39.709 cierres en DD y entre 10.518 y 12.808 de 99.944 en FD.

## Prueba 3 — En la capa 2, las claves se ensanchan y las consultas se estrechan con la profundidad

**Veredicto prerregistrado: CONFIRMADA.** Coeficiente de σ_k > 0 en 20/20 cabezas×semilla y de σ_q < 0 en 20/20 (binomial unilateral p = 9,5·10⁻⁷ en cada uno; el umbral era 18/20).

| Coeficiente de profundidad (log σ por nivel) | DD: media (IC 95 %, n = 5 semillas) | DD: rango en 20 cabezas | FD: media (IC 95 %) |
|---|---|---|---|
| σ_k, capa 2 | **+0,167** (0,154 a 0,180) | +0,048 a +0,232 | +0,099 (0,091 a 0,106); 20/20 > 0 |
| σ_q, capa 2 | **−0,372** (−0,423 a −0,321) | −0,541 a −0,046 | −0,195 (−0,227 a −0,162); 20/20 < 0 |
| Sin entrenar: σ_k / σ_q (DD) | +0,0006 / −0,0004 | 16/20 > 0 / 11/20 < 0 | 12/20 / 10/20 |

Cada nivel de profundidad multiplica σ_k por ≈ 1,18 y σ_q por ≈ 0,69 (derivado de las medias). En secuencias más largas el patrón se mantiene en 40/40 cabezas, con coeficientes de 0,5 a 0,6 veces los de DD. En los modelos sin entrenar, el signo de σ_k sale positivo en 16/20 en DD (binomial p = 0,006, bajo el umbral de 18/20), pero con una magnitud unas 300 veces menor. El signo solo puede tener un sesgo leve de la arquitectura: lo que distingue al modelo entrenado es la magnitud.

## Prueba 5 — σ_k de la capa 2 como señal de error en cierres, sin selección

Medida única: la media de σ_k en las 4 cabezas de la capa 2, en el token que predice cada cierre (las posiciones de `close_acc`). El sentido se fijó antes de correr: mayor σ_k → error.

**Veredicto prerregistrado: CONFIRMADA.** La AUC media en DD es **0,835** (0,811–0,861), sobre el umbral de 0,70. El IC 95 % por bootstrap de posiciones queda sobre 0,5 en 5/5 semillas (el más bajo, 0,800–0,822), y el bootstrap por secuencias da lo mismo.

| AUC para error en cierres (media de 5 semillas; rango) | DD (20–60) | FD (80–120) |
|---|---|---|
| **σ_k capa 2 (medida prerregistrada)** | **0,835** (0,811–0,861) | 0,687 (0,645–0,729) |
| Posición sola | 0,823 (0,794–0,849) | 0,797 (0,792–0,806) |
| Profundidad sola | 0,661 (0,616–0,712) | 0,583 (0,573–0,597) |
| σ_k ajustada por profundidad + posición + símbolo | 0,526 (0,488–0,545); IC > 0,5 en 3/5 | 0,529 (0,524–0,533); IC > 0,5 en 5/5 |
| Logística sin σ_k → con σ_k (validación cruzada) | 0,901 → 0,901 (Δ +0,0001; Δ > 0 en 2/5) | 0,890 → 0,890 (Δ +0,0002; 1/5) |

- **FD (secundaria):** no se cumple el umbral. La AUC media es 0,687, aunque el IC queda sobre 0,5 en 5/5.
- **Controles:** la posición sola discrimina casi igual que σ_k en DD (0,823 frente a 0,835) y mejor en FD (0,797 frente a 0,687). Ajustada por profundidad, posición y símbolo, σ_k conserva una señal residual pequeña (AUC ≈ 0,53). No mejora la predicción del error: con validación cruzada, agregar σ_k al modelo que ya tiene profundidad, posición y símbolo cambia la AUC en 0,003 como máximo (media +0,0001).

**Lectura.** La Prueba 3 replica en semillas nuevas, sin selección, la división de roles en la capa 2: con la profundidad, la consulta se estrecha y la clave se ensancha. La Prueba 5 cumple su endpoint, pero sus controles cambian lo que significa. σ_k de la capa 2 marca dónde se equivoca el modelo sobre todo porque sigue a la posición, la profundidad y el símbolo. Fuera de eso aporta poco y no mejora la predicción. No sirve como señal de error independiente.

**Límites.** Los 20 casos de la Prueba 3 no son independientes (4 cabezas por semilla). Los conjuntos de datos son los de las pruebas previas: se replica sobre inicializaciones nuevas, no sobre datos nuevos. El criterio de IC de la Prueba 5 se aplicó como límite inferior > 0,5, en el sentido fijado. Las secundarias no tienen corrección por multiplicidad. Es una tarea sintética con un modelo pequeño, y la única medida de desempeño es `close_acc`.
