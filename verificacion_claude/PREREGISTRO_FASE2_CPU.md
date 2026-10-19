# PRE-REGISTRO — Fase 2 CPU (escrito ANTES de ejecutar, 17-oct-2026)

Tres pruebas motivadas por la crítica externa del 17-oct: (1) comparación directa
con el incumbente de curvatura mixta (CAT) en tareas de secuencia; (2) prosa
natural, no solo código; (3) mecanismo del umbral d_h ≈ 16. Los tres endpoints
quedan fijados aquí antes de correr nada. Sea cual sea el resultado, se reporta.

---

## P1 — CAT-style vs Fisher-Rao en secuencias (`exp_cat_incumbente.py`)

CAT (Lin, Ojha & Bai, arXiv:2510.01634) no publica código y evalúa solo en
grafos de conocimiento. Se reimplementa fielmente su idea a nuestra escala:
enrutamiento diferenciable POR TOKEN entre tres ramas de atención en espacios
modelo (euclidiana / bola de Poincaré con curvatura aprendible / esfera unitaria),
proyecciones compartidas para igualar parámetros ("CAT-style", así se rotula).

- **Tarea:** Dyck-2 extrapolativo (train 20–60, eval 80–120), config §4.1 del paper.
- **Equidad:** piloto SOLO en semilla 1 para el brazo CAT: lr ∈ {5e-4, 1e-3} ×
  c_init ∈ {0.5, 1.0}; la mejor config pasa a confirmatorio. Euclid y FR usan la
  config publicada sin tocar.
- **Confirmatorio:** semillas 1–5 pareadas, brazos {euclid, FR-isotrópico, CAT-style}.
- **ENDPOINT PRIMARIO:** FR-isotrópico vs CAT-style en OUT acc (t pareado
  bilateral, n=5). Predicción (lectura superconjunto, §5.1): FR ≥ CAT.
  **Si CAT supera a FR con p<0.05, la lectura superconjunto queda refutada
  contra el incumbente real y así se reporta.**
- **Secundarios:** CAT vs euclid (¿la mezcla de espacios modelo supera al plano
  en secuencias?); FR vs euclid (sanidad: debe reproducir ≈ +2–3 pp).

## P2 — Prosa natural, piloto CPU (`exp_prosa_man.py`)

Corpus NUEVO jamás usado en el proyecto: prosa técnica en inglés de las páginas
man locales del sistema (~3 MB, split 90/10 por página, determinista). Es prosa
(oraciones en lenguaje natural), no código; se rotula como "prosa técnica" —
TinyStories/WikiText en GPU sigue siendo E2 del plan.

- **Diseño idéntico** a `exp_codigo_extrapolativo.py`: char-level, train L=128,
  eval {128, 256, 512}, 4000 secuencias, 3 épocas, d128/2capas/4cabezas.
- **Brazos:** euclidiano vs FR-estabilizado (cap μ=12, el recomendado en §3.3).
- **Semillas:** 46–55 (diez, frescas; nunca usadas).
- **ENDPOINT PRIMARIO:** bpc de la zona extrapolativa a L=512, FR-estable <
  euclidiano, p<0.05 (t pareado bilateral, n=10).
- **Secundarios:** (i) zona extrapolativa L=256; (ii) control IN L=128
  (predicción declarada: euclidiano gana IN, como siempre); (iii) estabilidad
  sd(FR) ≤ 1.5·sd(Euc) en extra L=512.
- **NEGATIVO DECLARADO** si el primario no se cumple: la transferencia quedaría
  (hasta GPU) confinada a código, y la crítica "prosa" quedaría respondida en
  contra nuestra.

## P3 — Mecanismo del umbral d_h ≈ 16 (`exp_mecanismo_umbral.py`)

**Hipótesis mecanística H-c (régimen de curvatura), escrita antes de medir:**
el kernel FR solo rinde cuando los tokens operan a distancias geodésicas
comparables al radio de curvatura R = 1/√|K| = √2; con d_h pequeño la
configuración aprendida queda en el régimen casi-plano (d ≪ R), donde FR no
puede superar al euclidiano pero paga su sobrecosto de optimización.

- **Diseño:** entrenar el brazo FR en d_model ∈ {16, 32, 64, 128}
  (d_h ∈ {4, 8, 16, 32}), 3 semillas {6, 7, 8}, config del barrido §4.7;
  instrumentar en eval IN la distribución de d/R por par (triángulo inferior
  estricto) y la d/R PONDERADA POR LA ATENCIÓN (lo que el softmax integra).
- **MÉTRICA PRIMARIA:** media ponderada por atención de d/R.
- **CRITERIOS DE ÉXITO (ambos):**
  (i) d/R ponderada crece monótonamente con d_h (mismo orden en las 3 semillas);
  (ii) cruza 1.0 entre d_h=8 y d_h=16 — exactamente donde el barrido §4.7
  localizó el cruce de eficiencia.
- (i) sin (ii) = apoyo parcial (el régimen importa pero el cruce no coincide).
  Ni (i) ni (ii) = **negativo declarado: el mecanismo sigue sin explicación.**
- Nota: estudio observacional; la prueba causal (desplazar el régimen manipulando
  β/σ-init y predecir el corrimiento del umbral) queda para después.

---

**Orden de ejecución:** P3 (minutos) → P1 (~1 h) → P2 (~1–2 h). Todos en esta
máquina (CPU, PyTorch 2.8, venv ~/asreview/venv). Resultados en JSON junto a
los demás del proyecto.

---

# ADENDA DE TRAZABILIDAD (19-oct-2026, posterior a la ejecución)

El texto anterior a esta adenda no se modificó: sus primeros 4387 bytes son
idénticos byte a byte a la versión escrita antes de ejecutar (17-oct-2026,
12:55, hora de Chile, UTC−3); SHA-256 de esos bytes:
c42073321df47d430140c0df26e9af9002ac4da429d6b51d97f05bb8d1507b8f. A diferencia
de PREREGISTRO_ROPE.md, este archivo no se selló con un hash antes de ejecutar;
lo que sigue se reconstruyó del registro de la sesión de trabajo y de las fechas
de modificación de los archivos.

- **Orden real de ejecución.** Los tres scripts se escribieron entre las 12:56 y
  las 12:57 del 17-oct y se lanzaron a las 12:57 en una sola orden secuencial,
  en el orden declarado: P3 (terminó a las 12:59) → P1 (13:11, `log_p1_cat.txt`)
  → P2 (13:20, `log_p2_prosa.txt`).
- **Ediciones posteriores de este archivo.** El 17-oct a las 19:44 se añadió al
  final una adenda que no modificaba P1–P3, y a las 19:48 se retiró de este
  archivo. Ninguna línea del texto anterior cambió.
- **Scripts.** `exp_prosa_man.py` (P2) no cambió después de ejecutarse. En
  `exp_cat_incumbente.py` (P1) solo se modificaron después comentarios y
  docstrings (18 y 19-oct: descripción de las diferencias con el CAT publicado);
  el código ejecutable es idéntico al que se ejecutó.
- **Re-ejecución de P3.** La instrumentación de P3 registraba d/R también en las
  consultas situadas en posiciones de relleno de cada lote de evaluación. El
  18-oct (11:20) se corrigió para contar solo tokens reales (`valid_len` en
  `exp_mecanismo_umbral.py`; en el mismo cambio se silenciaron advertencias de
  la biblioteca) y P3 se re-ejecutó (terminó a las 11:34). Diseño, semillas,
  configuración, métrica y regla de decisión no cambiaron. Los datos publicados
  (`log_p3_umbral.txt`, `resultados_mecanismo_umbral.json`) son de esa
  re-ejecución; la salida de la primera ejecución no se conservó como archivo.
  Ambas ejecuciones dan el mismo veredicto prerregistrado: **apoyo parcial**
  ((i) sí, (ii) no). Métrica primaria (d/R ponderada por la atención, media de
  3 semillas) para d_h = 4 / 8 / 16 / 32: primera ejecución 1,571 / 2,309 /
  3,015 / 4,007; re-ejecución 1,619 / 2,368 / 3,123 / 4,144.
