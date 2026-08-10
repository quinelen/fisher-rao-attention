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

Queries ejecutados (script: `../scratchpad/novelty_search.py`, copia de resultados abajo):
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
   §4.6 la usa como control: la geometría FR aporta +1.52 pp por encima (p=0.029).
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
| "Fisher-Rao attention" | **1** | DyCo-RL (arXiv:2606.08035, jun-2026, Lin et al.): usa la distancia FR para ANALIZAR distribuciones de atención existentes (asignar roles funcionales a tokens en MLLMs) — plugin diagnóstico sobre atención estándar, NO un kernel. Concurrente y adyacente; citado como [20]. |
| "information geometric attention" | 0 | — |
| "Gaussian hyperbolic attention" | 0 | — |
| "Fisher-Rao geodesic attention" | 0 | (nuestra frase de título: inédita) |
| "Fisher-Rao" + "attention kernel" + transformer (amplia) | 6 | Todos tangenciales. El más cercano: Mtetwa et al. (Mathematics/MDPI 2026) relaciona la temperatura de RoPE con la geometría FR de las distribuciones de atención — teoría adyacente tipo [17], citado como [21]. Resto: teoría de flujos de medida, difusión de video, correspondencias cuánticas — menciones de pasada. |

**Conclusión del barrido:** ningún trabajo indexado en Google Scholar computa scores de
atención como distancia geodésica Fisher-Rao entre tokens distribucionales. Los dos usos
existentes del objeto matemático en atención (DyCo-RL, Mtetwa et al.) son analíticos —
miden la geometría DE la atención estándar; el nuestro construye la atención DESDE la
geometría. La afirmación de novedad-como-síntesis de §7 queda verificada en las tres
fuentes principales (arXiv, OpenReview, Google Scholar) más Semantic Scholar parcial.

## Limitaciones residuales

- Semantic Scholar quedó cubierto solo parcialmente (rate-limit sin API key).
- Solo inglés; ventana hasta agosto 2026 (trabajo concurrente posterior puede aparecer;
  repetir el barrido de frases exactas la semana del envío es trivial y recomendado).
