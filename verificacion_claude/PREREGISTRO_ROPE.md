# PRE-REGISTRO — Extrapolación con posiciones relativas (RoPE), n = 20

Escrito el 18-oct-2026, ANTES de ejecutar `exp_rope.py`.

## Motivación

Toda la extrapolación de longitud del paper se mide con embeddings posicionales
absolutos aprendidos: más allá de la longitud de entrenamiento, esas posiciones
son vectores que nunca se entrenaron. Una corrida exploratoria previa (5 semillas,
1–5; ya vistas, por eso excluidas aquí) sugirió que con posiciones relativas
ambos modelos extrapolan mucho mejor y que Fisher–Rao sigue adelante
(+1,8 pp, 4/5 semillas, p = 0,10). Esta prueba confirmatoria decide si la
ventaja sobrevive con posiciones justas.

## Diseño

- **Tarea:** Dyck-2, configuración §4.1 del paper (train 20–60, eval OUT 80–120,
  10.000 muestras, 2 capas, d_model = 128, 4 cabezas, 5 épocas, AdamW 5e-4).
- **Sin embedding posicional absoluto** en ningún brazo.
- **Brazos:**
  - euclidiano + RoPE sobre q y k (estándar, base 10000);
  - Fisher–Rao isotrópico + la misma rotación aplicada a μ_q y μ_k. Es una
    isometría exacta de H^{d+1}: ‖R_i μ_q − R_j μ_k‖ depende solo de i − j, y la
    descomposición en una multiplicación de matrices se conserva.
- **Semillas:** 6–25 (veinte, frescas para esta comparación), pareadas.

## Endpoints

- **PRIMARIO:** exactitud OUT de cierre en Dyck-2, Fisher–Rao+RoPE vs
  euclidiano+RoPE, t pareado bilateral, p < 0,05, n = 20.
  - Si se cumple: la ventaja de Fisher–Rao sobrevive con posiciones relativas.
  - Si no se cumple: la ventaja publicada no está demostrada con posiciones
    relativas a esta escala.
- **SECUNDARIOS** (reportados, no decisorios):
  1. semillas ganadas;
  2. IC 95 % de la diferencia;
  3. exactitud IN;
  4. tarea de copia (misma configuración que §5) con los mismos dos brazos y
     semillas 6–25: diferencia OUT, t pareado.
