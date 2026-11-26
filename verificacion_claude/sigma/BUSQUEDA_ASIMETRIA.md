# Búsqueda de antecedentes — σ y profundidad, asimetría consulta/clave

29-oct-2026 · uso interno · búsqueda dirigida para evaluar la novedad de los hallazgos (a)–(d) sobre σ en la atención Fisher–Rao (Dyck-2).

**Hallazgos evaluados.** (a) σ codifica la profundidad de anidamiento (ΔR² ≈ 0,58 frente a 0,001 sin entrenar). (b) En la capa 2, σ_q baja (×0,69 por nivel) y σ_k sube (×1,18), en 20/20 cabezas×semilla. (c) Quitar la profundidad de σ_q baja la exactitud (0,15 pp fuera de distribución). (d) σ por token solo hace falta en la capa 2; en la capa 1 basta un escalar por cabeza.

## Tabla de antecedentes

Cercanía: **directa** (mismo fenómeno), **análoga** (misma idea, otro objeto), **lejana**, — (no aplica). (T) = abrí el texto completo; (R) = solo el resumen o la ficha.

| Referencia verificada | Qué muestra | a | b | c | d |
|---|---|---|---|---|---|
| Vilnis y McCallum 2015, *Word representations via Gaussian embedding*, ICLR, arXiv:1412.6623 (T) | Las palabras específicas tienen menos varianza; las amplias o polisémicas, más (§6.1). KL asimétrica para la implicación. | análoga | lejana | — | — |
| Nickel y Kiela 2017, *Poincaré embeddings…*, NeurIPS, arXiv:1705.08039 (T) | Raíz cerca del origen, hojas cerca del borde: la norma codifica el nivel. | análoga | — | — | — |
| Ganea, Bécigneul y Hofmann 2018, *Hyperbolic entailment cones…*, ICML, arXiv:1804.01882 (T) | La semiapertura ψ(x) = arcsin(K(1−‖x‖²)/‖x‖) decrece con la norma: a más profundidad, cono más estrecho. | análoga | — | — | — |
| Vilnis, Li, Murty y McCallum 2018, *…box lattice measures*, ACL, doi:10.18653/v1/P18-1025 (R) | Cajas con una medida de probabilidad. Que el volumen codifique la generalidad no figura en el resumen [no verificado]. | lejana | — | — | — |
| **Tifrea, Bécigneul y Ganea 2019, *Poincaré GloVe*, ICLR, arXiv:1810.06546 (T)** | Gaussianas diagonales ↔ (H²)ⁿ por la distancia de Fisher: σ es la altura del semiplano y la generalidad «debería codificarse en la magnitud de σ». Puntaje de hiperonimia por log-volumen. | **análoga, la más cercana** | lejana | — | — |
| Hewitt y Manning 2019, *A structural probe…*, NAACL, doi:10.18653/v1/N19-1419 (R); Chen et al. 2021, *Probing BERT in hyperbolic spaces*, ICLR, arXiv:2104.03869 (R) | La profundidad sintáctica es la norma L2² de una proyección lineal de BERT o ELMo. La sonda de Poincaré recupera árboles. | análoga | — | — | — |
| Khrulkov et al. 2020, *Hyperbolic image embeddings*, CVPR, arXiv:1904.02239 (T) | La distancia al origen funciona como incertidumbre (§5.1). | lejana | — | — | — |
| Gulcehre et al. 2019, *Hyperbolic attention networks*, ICLR, arXiv:1805.09786 (T) | Atención por distancia hiperbólica con radio pseudo-polar. Histograma de radios (fig. 3), sin separar consulta y clave. | lejana | lejana | — | — |
| Bdeir y Landwehr 2026, *…fully hyperbolic ViT in polar coordinates*, arXiv:2609.32899 (R) | Radio aprendido aparte de la dirección. No interpreta el radio de consultas ni de claves. | lejana | — | — | — |
| Fan et al. 2022, *…stochastic self-attention* (STOSA), WWW, arXiv:2201.06035 (T) | Covarianzas separadas para consulta y clave (W^Σ_Q, W^Σ_K) y atención −W₂. Analiza la popularidad con t-SNE, no las varianzas. | — | permite, no informa | — | — |
| Erick et al. 2023, *Stochastic vision transformers with Wasserstein distance-aware attention*, arXiv:2311.18645 (T) | Q, K y V gaussianas con atención −W₂. No encontré un análisis de las varianzas aprendidas. | — | permite, no informa | — | — |
| Zamani y Bendersky 2023, SIGIR, arXiv:2304.14522 (R); Nguyen et al. 2022, ICML, arXiv:2110.08678 (R) | Consulta y documento gaussianos comparados por −KL; claves como mezcla gaussiana. Los resúmenes no analizan las varianzas. | — | lejana | — | — |
| Zhang et al. 2024, *Selective attention* (SSA), NeurIPS, arXiv:2411.12892 (T) | Temperatura aprendida por consulta y por valor; la de clave rinde igual o peor. En un grafo de juguete depende del número de vecinos (signo ambiguo en el texto). | — | **análoga** (nitidez del lado de la consulta) | — | — |
| Yeh et al. 2023, *AttentionViz*, arXiv:2305.03210 (T, §6.3) | Norma de consulta − norma de clave = −4,59 de media en GPT-2 (0,41 en BERT). Los autores no lo explican. | — | **análoga** (Q≠K sin estructura) | — | — |
| Ebrahimi, Gelda y Zhang 2020, Findings EMNLP, arXiv:2010.04303 (T); Bhattamishra, Ahuja y Goyal 2020, EMNLP, arXiv:2009.11264 (R) | Con un símbolo inicial, la atención de la capa 2 es compatible con una pila. Los mecanismos de contador aprendidos se parecen a la construcción teórica. | contexto | — | — | análoga |
| **Yao, Peng, Papadimitriou y Narasimhan 2021, *Self-attention networks can process bounded hierarchical languages*, ACL, arXiv:2105.11115 (T)** | Capa 1: cuenta la profundidad con Q = K = 1 (atención uniforme); la red feed-forward (FFN) la lleva a un ángulo. Capa 2: Q = [20D²·d_i, 1, 2], K = [d_j, p_j, o_j]. La profundidad va en la *dirección* y la ganancia de nitidez, en la *consulta*. | análoga (dirección, no escala) | análoga funcional | — | **análoga, la más cercana** |
| Wen, Li, Liu y Risteski 2023, *Transformers are uninterpretable with myopic methods…*, arXiv:2312.01429 (R; sede no verificada) | Muchas soluciones resuelven Dyck acotado; el patrón de atención de una capa puede ser casi aleatorio. | — | — | motiva | — |
| Tiwari, Gupta y Hahn 2025, arXiv:2502.01432 (R) | La profundidad de pila es decodificable por sondas en cada token. | análoga | — | — | — |
| **Sharma, Dawes y Raval 2026, *Dissociating decodability and causal use in bracket-sequence transformers*, arXiv:2604.22128 (R)** | Profundidad y cima de la pila son decodificables, pero su uso causal difiere. Enmascarar la atención a la cima daña mucho; ablacionar subespacios residuales de baja dimensión daña poco. | análoga | — | **análoga, la más cercana** | — |
| Kundu et al. 2026, *…Gaussian kernel attention*, arXiv:2605.02144 (R) | Núcleo RBF con un ancho aprendido por cabeza, sin proyecciones Q/K; competitivo. | — | — | — | análoga (en todas las capas) |

**No abiertos [no verificado]:** Hypformer (KDD 2024), HVT (2024), DAAM (2024), los trabajos sobre *attention sinks* (Xiao 2023; Gu 2024) y Costa et al. (2015).

## Veredicto por componente

- **(a) σ ↔ profundidad dentro de la atención: «existe algo análogo».** La escala de una representación como código de generalidad o nivel está establecida en los *embeddings* estáticos (Vilnis; Nickel; Ganea). Tifrea et al. lo plantean en esta misma geometría, con σ como altura del semiplano de Fisher. La profundidad también se decodifica de representaciones contextuales (Hewitt y Manning) y de transformers de Dyck (Tiwari; Sharma). **No encontrado:** escalas por token de consulta o clave *dentro del núcleo* de atención que codifiquen la profundidad.
- **(b) Asimetría consulta/clave: «no encontrado»**, con análogos parciales. Hay normas Q/K dispares sin explicación (Yeh), control de nitidez del lado de la consulta (SSA; la construcción de Yao) y varianzas Q/K separadas sin análisis (STOSA, SViT). Nada encontrado muestra escalas de consulta y clave moviéndose **en sentidos opuestos** con una variable estructural. Matiz: σ_q sigue la intuición hiperbólica (más profundo, más cerca del borde) y σ_k la invierte.
- **(c) Intervención: «existe algo análogo».** Sharma et al. (2026) intervienen las representaciones de profundidad y pila en Dyck. **No encontrado:** una intervención sobre la escala de la atención. El efecto pequeño concuerda con su conclusión: decodificable no equivale a usado con fuerza.
- **(d) Simplificación por capa: «existe algo análogo».** Yao et al. (2021) predicen la división: en la capa 1 basta una atención uniforme para contar la profundidad, y el emparejamiento por profundidad ocurre en la capa 2. **No encontrado:** una prueba empírica de no inferioridad que muestre que la escala por token solo hace falta en la capa 2.
- **Fórmula de alcance.** σ′* = √(σ² + ‖Δμ‖²/2) sale en una línea de la distancia cerrada (Costa et al. 2015, ya citada): es una derivación, no un aporte. La explicación «clave ancha para ser alcanzable, consulta estrecha para seleccionar» sigue siendo una hipótesis.

**Para el preprint.** (1) Citar a Tifrea et al. (2019), que hoy falta: es el antecedente directo de «σ = generalidad» en el semiplano de Fisher. (2) Matizar la frase que llama a σ un «*generality radius*»: vale para las consultas y se invierte en las claves. (3) Citar a Yao (2021) para (d), a Sharma (2026) para (c) y a Hewitt y Manning (2019) para (a).

## Redacción sugerida (inglés)

> Encoding generality or hierarchical level in the scale of a representation is well established for static embeddings — Gaussian variance (Vilnis & McCallum, 2015), whose Fisher geometry Tifrea et al. (2019) identify with the hyperbolic half-plane, and hyperbolic norm (Nickel & Kiela, 2017; Ganea et al., 2018) — and syntactic or stack depth is decodable from transformer representations (Hewitt & Manning, 2019; Sharma et al., 2026); separate query and key variances also exist in distribution-based attention (Fan et al., 2022; Erick et al., 2023), but we found no analysis of what they learn. What we add is evidence from inside the attention kernel: in models trained on Dyck-2, second-layer per-token query and key scales track nesting depth in opposite directions — queries narrow (×0.69 per level) while keys widen (×1.18; 20/20 head–seed pairs on held-out seeds) — removing depth from σ_q slightly lowers accuracy, and a per-head scalar suffices in the first layer, consistent with constructions in which the first layer only counts depth (Yao et al., 2021). We did not find this opposite-direction query/key scale code reported in prior work; the reachability account (the key nearest to a distant query is wider, σ′* = √(σ² + ‖Δμ‖²/2)) is a hypothesis, not a demonstrated mechanism.

Preferí «*we did not find … in prior work*» a «*to our knowledge … first*», porque la búsqueda fue dirigida y no exhaustiva. Si se usa «*to our knowledge*», que se limite a (b).

## Límites de la búsqueda

Motor web general (13 búsquedas) y unas 25 páginas abiertas (arXiv, ACL Anthology, NeurIPS/ICML). Texto completo en 12 fuentes; el resto, solo el resumen. No usé Google Scholar, Semantic Scholar ni la API de OpenReview. La búsqueda es del 29-oct-2026, solo en inglés, e incluye preprints de 2026 sin revisión por pares. Pueden faltar *workshops*, tesis y preprints no indexados. «No encontrado» significa que no apareció en esta búsqueda. Lo más expuesto es (b): el siguiente paso sería revisar las citas de STOSA y SViT en busca de análisis de varianzas consulta/clave.
