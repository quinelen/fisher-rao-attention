# Registro de búsqueda sistemática de novedad

**Fecha:** 11 de julio de 2026
**Objeto:** verificar si el mecanismo central del paper — atención con distancia geodésica
exacta de Fisher-Rao entre tokens gaussianos isotrópicos (§3.3) — o sus componentes
reclamados como novedosos en §7, ya están publicados.

## Fuentes y método

| Fuente | Método | Cobertura |
|---|---|---|
| arXiv | API oficial (`export.arxiv.org`), frase exacta y combinaciones abs: | completa para los queries listados |
| OpenReview | API v2 (`api2.openreview.net/notes/search`) | completa para los queries listados |
| Semantic Scholar | Graph API (sin key) | **parcial**: 5 de 8 queries fallaron por rate-limit |
| Búsqueda web general | motor de búsqueda (sesiones previas) | CAT, Curve Your Attention, HELM, Stochastic ViT, DAAM |

Queries ejecutados (copia de resultados abajo):
`"Fisher-Rao attention"`, `"information geometric attention"`, `"geodesic attention"`,
`abs:"Fisher-Rao" AND abs:"transformer" AND abs:"attention"`, `"Bhattacharyya attention
transformer"`, `"hyperbolic attention isotropic Gaussian"`, `"Chentsov attention"`,
`"hyperbolic Gaussian attention"`, entre otros.

## Resultado principal

**"Fisher-Rao attention" como frase exacta: 0 resultados en arXiv y 0 en OpenReview.**
No se encontró ningún trabajo que compute scores de atención como distancia geodésica
Fisher-Rao (exacta o aproximada) entre tokens representados como distribuciones
gaussianas.

## Vecinos más cercanos encontrados (todos citados en §7 del paper)

1. **CAT: Curvature-Adaptive Transformers** (arXiv:2510.01634, 2025) — gating por token
   entre ramas euclidiana/hiperbólica/esférica. **Anticipa nuestro §3.4/§5 (gating)**;
   usa espacios modelo (Poincaré/hiperesfera), no la métrica Fisher-Rao ni distribuciones.
2. **Gating Enables Curvature: A Geometric Expressivity Gap in Attention**
   (arXiv:2604.14702, 2026) — resultado TEÓRICO: analiza la geometría Fisher-Rao de las
   salidas de atención (como parámetros de medias gaussianas) y prueba que la atención
   sin gating queda restringida a variedades estadísticas planas mientras el gating
   multiplicativo habilita curvatura. Es teoría adyacente sobre la geometría *de* la
   atención estándar; nuestro trabajo construye la atención *desde* distancias geodésicas.
   Sin arquitectura nueva, sin Dyck. **Cita obligatoria**: dialoga directamente con
   nuestros hallazgos de gating (§5, §5.1).
3. **ISRE / Riemannian & Symplectic place recognition** (arXiv:2604.01598, 2026) — usa
   Fisher-Rao para "relation features" y self-atención riemanniana hiperbólica en otra
   parte del pipeline (nubes de puntos). No es FR-geodésica como score entre tokens.
4. **RieIF** (arXiv:2604.06971, 2026) — usa geometría Fisher-Rao para proyectar la
   entrada a una hiperesfera unitaria (similar a nuestro caso esférico §3.1) ANTES de un
   transformer estándar; la métrica no entra al score de atención.
5. **Stochastic ViT (Wasserstein)** (arXiv:2311.18645, 2023) — tokens gaussianos +
   atención por distancia entre distribuciones, con Wasserstein/Bures. Nuestra ablación
   §4.6 la usa como control: la geometría FR aporta +1,52 pp por encima (p = 0,029).
6. **Curve Your Attention** (arXiv:2309.04082), **HELM** (arXiv:2505.24722),
   **DAAM** (arXiv:2401.11143) — curvatura mixta / mixture-of-curvature / atención
   adaptativa por densidad; ninguna usa Fisher-Rao como score.

## Veredicto

- **El kernel §3.3 (FR geodésica exacta entre gaussianas isotrópicas como score de
  atención, a costo matmul) no fue encontrado en ninguna fuente consultada.** La
  afirmación de novedad-como-síntesis de §7 se sostiene con la evidencia disponible.
- El gating adaptativo de curvatura **no** es novedoso (CAT et al.) — ya retirado de las
  contribuciones del paper.
- Existe teoría reciente (arXiv:2604.14702) sobre Fisher-Rao y gating en atención que el
  paper debe citar y discutir.

## Barrido manual de Google Scholar (COMPLETADO 10-ago-2026)

Realizado vía navegador sobre scholar.google.com. Resultados por frase exacta:

| Query | Resultados | Veredicto |
|---|---|---|
| "Fisher-Rao attention" | **1** | DyCo-RL (arXiv:2606.08035, jun-2026, Lin et al.): usa la distancia FR para MEDIR desplazamientos de atención en un transformer multimodal estándar, asignar roles funcionales a los tokens y reponderar con ello las ventajas en aprendizaje por refuerzo — señal de entrenamiento derivada de la atención estándar, NO un kernel. Concurrente y adyacente; citado en el paper (Lin et al., 2026). |
| "information geometric attention" | 0 | — |
| "Gaussian hyperbolic attention" | 0 | — |
| "Fisher-Rao geodesic attention" | 0 | (nuestra frase de título: inédita) |
| "Fisher-Rao" + "attention kernel" + transformer (amplia) | 6 | Todos tangenciales. El más cercano: Mtetwa et al. (Mathematics/MDPI 2026) relaciona la temperatura de RoPE con la geometría FR de las distribuciones de atención — teoría adyacente del tipo de Bathula y Joshi (2026); citado en el paper (Mtetwa et al., 2026). Resto: teoría de flujos de medida, difusión de video, correspondencias cuánticas — menciones de pasada. |

**Conclusión del barrido:** ningún trabajo indexado en Google Scholar computa scores de
atención como distancia geodésica Fisher-Rao entre tokens distribucionales. Los dos usos
existentes del objeto matemático en atención (DyCo-RL, Mtetwa et al.) parten de la atención estándar —
miden su geometría o la usan como señal de entrenamiento; el nuestro construye la atención DESDE la
geometría. La afirmación de novedad-como-síntesis de §7 queda verificada en las tres
fuentes principales (arXiv, OpenReview, Google Scholar) más Semantic Scholar parcial.

## Búsqueda actualizada (20-oct-2026)

Búsqueda amplia de atención con distancia hiperbólica (API de arXiv, búsqueda web y lectura del HTML de cada
trabajo). Entre otros trabajos que usan la distancia hiperbólica como
puntaje de atención (p. ej., HNN++, Shimizu, Mukuta y Harada, ICLR 2021, en la bola de Poincaré; Zhou 2026,
ya citado), estos tres son los más cercanos; ninguno parte de tokens distribucionales ni de la métrica de
Fisher–Rao. Citados en §7 (y el primero también en §4.5) desde la v1.2.

| Trabajo | Puntaje de atención | Veredicto |
|---|---|---|
| Gulcehre et al., *Hyperbolic Attention Networks* (ICLR 2019; arXiv:1805.09786) | α(q,k) = f(−β·d_H(q,k) − c) en el hiperboloide; f = exp normalizado (softmax) o sigmoide. Traducción automática, grafos, VQA | Antecedente de la forma del puntaje laplaciano (§4.5), en su variante softmax. Ya citado en la introducción; ahora también como antecedente de la forma funcional |
| Fein-Ashley, Feng y Pham, *HVT* (arXiv:2409.16897, sep-2024) | A = −d_D(Q,K)²/(α_h·√D) con d_D = arccosh(·) en la bola de Poincaré (ecs. 19–23; su ec. 19 usa la resta de Möbius dentro de la fórmula con arccosh: tal como está impresa, no es la distancia geodésica de la bola). Clasificación en ImageNet-1k | Antecedente de la forma del puntaje de distancia al cuadrado (§3.3), con un espacio modelo elegido a priori |
| Bdeir y Landwehr, *Rethinking the Fully Hyperbolic Vision Transformer in Polar Coordinates* (arXiv:2609.32899, 26-sep-2026) | s_ij = −(γ/τ)·d_L²(q_i,k_j), distancia geodésica lorentziana en coordenadas polares; kernel fusionado (puntaje, softmax y centroide) que no guarda la matriz N×N. Visión | Trabajo concurrente e independiente (apareció 20 días antes de la v1.0 pública, 16-oct-2026): misma forma funcional que el puntaje de distancia al cuadrado (§3.3), alcanzada desde un espacio modelo |

La afirmación de novedad de §7 no cambia: ningún trabajo calcula puntajes de atención como distancia geodésica de
Fisher–Rao entre tokens distribucionales. Lo que sí tiene antecedentes es la forma funcional del puntaje.

## Re-barrido de frases exactas (21-oct-2026)

| Query | API de arXiv | API v2 de OpenReview |
|---|---|---|
| "Fisher-Rao attention" | 0 | 0 |
| "Fisher-Rao geodesic attention" | 0 | 0 |
| "Gaussian hyperbolic attention" | 0 | 0 |
| "information geometric attention" | 1 (arXiv:2506.23542, *Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention* (Wang, He, Zeng y Qiu, 30-jun-2025), eliminación de ruido en profundidad ToF: falso positivo por lematización, no pertinente) | 0 |

Control de que la búsqueda de OpenReview funciona: "hyperbolic attention networks" sí devuelve resultados.
Google Scholar no se repitió (requiere navegador manual); su barrido es del 10-ago-2026.

## Búsqueda de títulos (16-nov-2026)

Al revisar el título del paper se buscó en la API de arXiv (65 consultas con frases genéricas en el título: «Fisher-Rao» + attention, geodesic + attention, «information geometry» + attention, hyperbolic + attention + kernel, «constant curvature» + attention, spherical + hyperbolic + attention, entre otras) y en la web general. Ningún título repite la cabecera «Fisher-Rao Geodesic Attention» ni parafrasea el título elegido. Aparecieron cinco vecinos temáticos que no estaban citados; cada uno se leyó en su registro primario:

| Trabajo | Qué hace | Veredicto |
|---|---|---|
| Rubacher, *Attention implements the Fisher geometry of exponential families*, ICML 2026 (póster 64496) | Teoría: una cabeza de atención softmax estándar calcula la posterior de Bayes para observaciones de familia exponencial; la similitud es una divergencia de Bregman. Solo se leyó el resumen (OpenReview pidió verificación de navegador). | Complementario: geometría dualmente plana (la del control KL de §4.6), no distancia geodésica de Fisher–Rao. Citado. |
| Wibisono y Wang, *Exponential family attention*, arXiv:2501.16790 (2025) | La familia exponencial es la verosimilitud de cada observación; su parámetro natural sale de una atención de producto punto estándar. | Complementario: consultas y claves siguen siendo vectores. Citado. |
| Govindarajan, Sidén, Roll y Lindsten, *QUEST*, ICLR 2026, arXiv:2604.00199 | Normaliza las claves a una hiperesfera y deja libres las consultas; evaluado en estabilidad y robustez. | Complementario: vecino del miembro esférico, con puntaje de producto punto. Citado. |
| Song, *Riemannian attention mechanisms for transformers*, arXiv:2608.01283 (2026) | Teoría sin experimentos: distancia de Mahalanobis bajo métricas aprendidas por token, promediadas entre los dos extremos. | Complementario: métrica aprendida sobre tokens vectoriales, no fijada por una familia de distribuciones. Citado. |
| Ge, *Riemann GeoResolver*, arXiv:2608.10416 (2026) | Teoría sin experimentos: peso de distancia inversa con distancia euclidiana, de la bola de Poincaré o de círculo máximo, entre puntos de espacios modelo. | Se solapa en la forma (tres geometrías con una misma plantilla), no en el contenido: sin métrica de Fisher–Rao ni tokens distribucionales. Citado. |

En la misma pasada se agregó Hypformer (Yang et al., KDD 2024, arXiv:2407.01290), un transformer completo en el modelo de Lorentz con autoatención lineal, que faltaba entre los transformers hiperbólicos citados. Ninguno de los seis calcula puntajes de atención como distancias geodésicas de Fisher–Rao entre tokens distribucionales.

## Limitaciones residuales

- Semantic Scholar quedó cubierto solo parcialmente (rate-limit sin API key).
- Solo inglés. Frases exactas en arXiv y OpenReview repetidas el 21-oct-2026; Google Scholar, al 10-ago-2026.
