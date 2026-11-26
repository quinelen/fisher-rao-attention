# Resultado — Intervención sobre σ: ¿el modelo usa la profundidad de σ_q? (prueba prerregistrada)

25-oct-2026 · uso interno · protocolo `PREREGISTRO_INTERVENCION.md`, sellado en `SELLO_INTERVENCION.txt` (00:15 UTC) antes de la corrida · datos: `log_intervencion.txt`, `resultados_intervencion.json`.

**Veredicto prerregistrado: CONFIRMADA, con efecto pequeño.** En Dyck-2, quitar la profundidad de σ_q de la capa 2 (conservando el símbolo) baja la exactitud de cierre más que conservar solo la profundidad. La diferencia absoluta es de 0,15 puntos porcentuales fuera de distribución. Esa cifra queda bajo el 0,01 que el prerregistro fijó como caída «pequeña».

Modelos: reentrenamiento determinista idéntico al de la prueba anterior (semillas 401–405). `close_acc` es igual al de entonces en 10 de 10 mediciones (diferencia 0), y el control de identidad no cambia ninguna exactitud. Base: 98,50 % dentro de distribución (DD, 20–60) y 88,90 % fuera de distribución (FD, 80–120).

| Sitio intervenido | Régimen | Caída M1 (sin prof.) | Caída M2 (con prof.) | Caída M3 (símb. + pos.) | M1 − M2 | t(4) | p |
|---|---|---|---|---|---|---|---|
| **A: σ_q capa 2 (primario)** | **FD** | **0,0015** | **−0,0001** | 0,0009 | **+0,0015** | **3,04** | **0,038** |
| A: σ_q capa 2 | DD | 0,0046 | 0,0006 | 0,0034 | +0,0041 | 8,67 | 0,001 |
| B: σ_k capa 2 | FD | 0,0031 | 0,0012 | 0,0018 | +0,0018 | 3,28 | 0,031 |
| B: σ_k capa 2 | DD | 0,0019 | 0,0033 | 0,0040 | −0,0014 | −1,53 | 0,20 |
| C: σ_q capa 1 (control) | FD | −0,0002 | −0,0001 | −0,0001 | −0,0001 | −1,45 | 0,22 |
| C: σ_q capa 1 (control) | DD | 0,0007 | 0,0005 | 0,0005 | +0,0002 | 1,15 | 0,31 |

Caídas = exactitud base − exactitud intervenida (proporción, media de 5 semillas).

- **Primario:** M1 − M2 por semilla: +0,0022, −0,0002, +0,0017, +0,0013 y +0,0028 (positivo en 4 de 5). Con M2 la exactitud queda en la base (−0,0001), aunque M2 solo reproduce el 71 % de la varianza de log σ_q (R² de fidelidad). M1 reproduce el 28 %.
- **Aporte propio de la profundidad (M3 − M2, sitio A):** +0,0009 en FD (t(4) = 3,00; p = 0,040) y +0,0028 en DD (t(4) = 9,30; p = 0,0007). Agregar la posición al símbolo recupera poco; agregar la profundidad recupera casi todo.
- **σ_k de la capa 2:** el patrón se repite en FD (p = 0,031), pero no en DD, donde M2 daña más que M1 (fidelidad de M2: 0,86).
- **Control (capa 1):** todas las caídas son ≤ 0,0007 y M1 − M2 ≈ 0, como se predijo: no hay profundidad que quitar.
- **Tope:** M2 en el sitio A llevó σ̃ al piso (0,05 + 10⁻⁴) en el 4 % de las posiciones. Las demás manipulaciones, ≈ 0 %.

**Post hoc (no prerregistrado, descriptivo).** En términos relativos, en el sitio A dentro de distribución, M1 multiplica los errores de cierre por 1,43 (1,12–1,81): de 1,5 % a 2,0 %. M2 los multiplica por 1,06. Fuera de distribución, M1 los multiplica por 1,01, porque allí la tasa de error base es de 11 %.

**Lectura.** El modelo usa la profundidad que lleva σ_q de la capa 2: es la parte de σ_q cuya pérdida importa. Pero depende poco de ella. Los demás canales (μ, valores, otras capas) cargan casi toda la información de profundidad que usa la predicción.

**Límites.** Se interviene un sitio por vez, con sustitutos lineales ajustados en una calibración aparte. M2 pierde parte de la varianza y topa el 4 % de las posiciones. Es una sola tarea sintética y un modelo pequeño, con n = 5. Las secundarias van sin corrección por multiplicidad, y la única medida es `close_acc`.
