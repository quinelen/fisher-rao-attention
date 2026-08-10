# Fisher–Rao Geodesic Attention

**Exact constant-curvature attention kernels at matmul cost, with data-driven geometry selection.**

This repository accompanies the preprint [`paper_draft.md`](paper_draft.md). It contains everything needed to reproduce every number in the paper on a laptop CPU (~2.5 h total, no GPU required).

![Main results](verificacion_claude/fig_results.png)

## The idea in three sentences

Standard attention measures token similarity in an arbitrary flat vector space. We map queries and keys to probability distributions and use the **exact Fisher–Rao geodesic distance** — the unique geometry invariant under sufficient statistics (Chentsov's theorem) — where categorical distributions give a *spherical* manifold (K = +1/4), isotropic Gaussians give a genuinely *hyperbolic* space H^(d+1) (K = −1/2), and dot-product attention is the flat member of the same family. Both curved kernels compute in closed form at the cost of a single matrix multiplication, and on hierarchical length-extrapolation (Dyck-2) accuracy orders exactly by curvature: **Euclidean 85.8% < spherical 86.6% < hyperbolic-isotropic 88.4%** (parameter-matched, 5 paired seeds, p = 0.0009).

## Headline results

| Attention kernel | Curvature | Dyck-2 OUT accuracy | Params | Cost vs. standard |
| :--- | :---: | :---: | :---: | :---: |
| Euclidean (dot product) | 0 | 85.80% ± 0.65 | 413k | 1× |
| Spherical Fisher–Rao (exact) | +1/4 | 86.63% ± 0.64 | 413k | ≈1.1× |
| Ricci–Mahalanobis (quadratic approx.) | approx. | 86.87% ± 0.71 | 413k | ≈10× |
| Hyperbolic anisotropic (exact) | −1/2 | 87.58% ± 0.69 | 479k | ≈28× |
| **Hyperbolic isotropic (exact, ours)** | **< 0** | **88.43% ± 0.77** | **415k** | **≈2.3×** |
| Mixed-curvature adaptive (gated) | learned | 86.94% ± 0.68 | ≈415k | ≈2.7× |

Further findings (Sections 4.6, 4.7, 5 and 5.1 of the paper):

- **The gain comes from the curvature, not just from Gaussian tokens (§4.6).** Holding the Gaussian token representation fixed and swapping only the distance, the Wasserstein–Bures kernel (the geometry used by prior stochastic transformers, arXiv:2311.18645) gains +1.12 pp over point-token attention, while the Fisher–Rao (hyperbolic) geometry adds a further **+1.52 pp on top** (p = 0.029) — the larger share of the advantage.
- **A genuine, threshold-gated efficiency result (§4.7) — the paper's strongest positive claim.** Above a head-width threshold ($d_h \ge 16$), the hyperbolic kernel at dimension $D$ becomes statistically indistinguishable in accuracy from Euclidean attention at dimension $2D$ ($p > 0.25$) while using **≈3.8–3.9× fewer parameters**, and at the largest scale tested (d_model 128 vs. 256) is **1.5× faster in real wall-clock training time** despite its per-operation overhead — because the quadratic $O(d_{\text{model}}^2)$ projection cost dominates at scale. Below the threshold ($d_h \le 8$) the effect reverses and hyperbolic attention is significantly worse; this is not a universal speedup.
- **Geometry selection follows data structure (§5).** A gated mixture of the three kernels — re-instantiating the adaptive mixed-curvature idea of CAT (arXiv:2510.01634), which is prior work — allocates 0.42 of its gate weight to the hyperbolic kernel on hierarchical data and collapses it to 0.09 on flat data (Euclidean gate: 0.28 → 0.64). We use the gate as a diagnostic probe, not as a claimed novelty.
- **A pre-specified negative result (§5.1).** On a mixed-structure stress test the adaptive mixture *lost* to the pure hyperbolic kernel (p = 0.0013): negative curvature is not a hierarchy specialist but a **superset geometry** — it emulates flat attention at small scales while the converse is impossible. Practical prescription: on present evidence, hyperbolic is a safe default.
- **A second negative result against the real incumbents (§5.2).** On the canonical WordNet-mammals embedding benchmark, the Fisher–Rao kernel crushes Euclidean (+0.49 MAP at D=10) but is **beaten by Poincaré embeddings** (0.883 vs 0.941, p = 0.0002; our Poincaré reimplementation reproduces the published MAP ≈ 0.93). The kernel's value is specific to *sequence attention* (§4); it is **not** a better hierarchy embedder and does not compete with dedicated hyperbolic graph methods in the ontology/KG domains where those are deployed. This bounds the applied/commercial reading of the work.
- **Two further commercial hypotheses tested and refuted (out of scope of the attention mechanism, see `verificacion_claude/`).** The scale parameter $\sigma$ is not a usable predictive-uncertainty signal (AUC-ROC ≈ 0.49, chance level). In few-shot hierarchical classification, the Fisher–Rao geometry is *outperformed* by both Euclidean and Poincaré at low sample counts (p ≤ 0.0003 vs. Poincaré at k=10/class) — the opposite of what hyperbolic geometry's known embedding data-efficiency would suggest.

## Repository structure

```
.
├── paper_draft.md                  # The preprint (all claims cross-referenced to scripts below)
├── README.md
├── LICENSE                         # MIT (code); paper text © the author
├── requirements.txt
├── historico_ricci/                # Original Ricci-flow-era files (July 2026), kept for
│                                   #   provenance — the starting point this project evolved from
└── verificacion_claude/            # All code, raw outputs, and figures
    │
    │   # --- Core experiments (Sections 4–5 of the paper) ---
    ├── exp_tres_geometrias.py      # Main benchmark: Euclidean / spherical / hyperbolic-anisotropic,
    │                               #   5 seeds, paired t-tests  (~30 min)
    ├── exp_ricci_addendum.py       # 4th arm: quadratic Ricci–Mahalanobis approximation  (~25 min)
    ├── fast_gaussian_test.py       # IsotropicHyperbolicAttention (the paper's main kernel),
    │                               #   speed micro-benchmark + ablations  (~15 min)
    ├── exp_mixta.py                # MixedCurvatureAttention + gate analysis on Dyck-2 vs Copy  (~20 min)
    ├── exp_mixta2.py               # §5.1 mixed-structure stress test (negative result)  (~25 min)
    ├── exp_wasserstein_vs_fisherrao.py  # §4.6 geometry-vs-representation ablation
    │                               #   (Fisher-Rao vs Wasserstein, same Gaussian tokens)  (~10 min)
    ├── exp_wordnet_incumbentes.py  # §5.2 vs real incumbents (Poincaré/Lorentz) on
    │                               #   WordNet mammals; needs nltk+wordnet  (~40 min)
    ├── exp_eficiencia_dimensional.py  # §4.7 parameter/compute efficiency sweep
    │                               #   (the paper's strongest positive result)  (~20 min)
    │
    │   # --- Raw outputs (regenerated by the scripts above) ---
    ├── resultados_tres_geometrias.json
    ├── resultados_ricci_addendum.json
    ├── resultados_fast_gaussian.json
    ├── resultados_mixta.json
    ├── resultados_mixta2.json
    ├── resultados_wasserstein_vs_fr.json
    ├── resultados_wordnet_incumbentes.json
    ├── resultados_eficiencia_dimensional.json
    ├── fig_results.png             # Figure 1 of the paper
    │
    │   # --- Commercial due-diligence (out of scope of the attention mechanism;
    │   #     distance-based classifier, not attention — see §6 of the paper) ---
    ├── exp_fewshot_jerarquia.py     # Few-shot hierarchical classification vs.
    │                               #   Euclidean/Poincaré — refuted, FR loses at low k
    │
    │   # --- Historical / diagnostic (earlier project iterations, kept for provenance) ---
    ├── diag_ricci.py               # Diagnosis of the original Ricci-flow formulation
    │                               #   (metric saturation, curvature-sign analysis)
    ├── train_compare.py            # Pilot benchmark (smaller config)
    ├── train_fisher_rao.py         # First exact spherical kernel
    ├── v2_test.py, v2_baselines.py # Intermediate verification runs
    ├── multi_seed.py               # Pilot multi-seed comparison
    ├── resultados_*.txt, multi_seed_results.json
    └── paper_draft_v{1,2}_backup.md  # Earlier drafts, kept for transparency
```

## Reproducing the benchmark

Requires Python ≥ 3.9. CPU is sufficient (all reported numbers were produced on a laptop CPU).

```bash
pip install -r requirements.txt
cd verificacion_claude

# Main 3-geometry benchmark (5 seeds, paired stats)        ~30 min
python exp_tres_geometrias.py

# 4th arm: Ricci–Mahalanobis approximation                 ~25 min
python exp_ricci_addendum.py

# Isotropic hyperbolic kernel + speed micro-benchmark      ~15 min
python fast_gaussian_test.py bench    # speed table (seconds)
python fast_gaussian_test.py acc      # accuracy, 5 seeds

# Geometry-selection experiment (gates on Dyck-2 vs Copy)  ~20 min
python exp_mixta.py

# Mixed-structure stress test (§5.1)                       ~25 min
python exp_mixta2.py

# Geometry-vs-representation ablation (§4.6)               ~10 min
python exp_wasserstein_vs_fisherrao.py

# vs. real incumbents: Poincaré/Lorentz on WordNet (§5.2)   ~40 min
# requires: pip install nltk  (downloads WordNet corpus on first run)
python exp_wordnet_incumbentes.py

# Parameter/compute efficiency sweep (§4.7)                 ~20 min
python exp_eficiencia_dimensional.py
```

Each script prints per-seed results and summary statistics, and writes a `resultados_*.json` with the raw numbers. Seeds (1–5), data-generation seeds, and every hyperparameter are fixed in-line, so runs are deterministic on CPU.

### Using the kernel in your own model

The paper's main mechanism is self-contained in one class — `IsotropicHyperbolicAttention` in [`verificacion_claude/fast_gaussian_test.py`](verificacion_claude/fast_gaussian_test.py) (~40 lines). It is a drop-in replacement for a multi-head attention block: same input/output shapes, causal mask supported, one extra hyperparameter (`eps_sigma`). The exact Fisher–Rao distance between the isotropic Gaussian tokens is

```
d(i,j) = √2 · arccosh( 1 + (‖μᵢ−μⱼ‖²/2 + (σᵢ−σⱼ)²) / (2σᵢσⱼ) )
```

with the pairwise `‖μᵢ−μⱼ‖²` computed by a single matmul.

## Honest limitations

Synthetic task family (Dyck-2 and variants), 2-layer models, n = 5 seeds, CPU scale. Only the vs.-Euclidean contrasts survive Bonferroni correction. The efficiency result (§4.7) is threshold-gated: it holds for $d_h \ge 16$ and **reverses** below that (hyperbolic is significantly worse at small head width). Several other pre-specified hypotheses were tested and **failed**: geometry gating is unnecessary (§5.1), σ is not a usable uncertainty estimator (AUC ≈ 0.49), the kernel is beaten by Poincaré embeddings on static hierarchy embedding (§5.2), and it is outperformed by Euclidean and Poincaré alike in few-shot hierarchical classification at low sample counts — opposite to what hyperbolic geometry's embedding-side data efficiency would suggest. The demonstrated value is confined to sequence attention, above a scale threshold. See Section 6 of the paper before extrapolating any claim; this is a bounded workshop-scale contribution, not a general advance.

## Citation

```bibtex
@misc{quinelen2026fisherrao,
  title  = {Fisher--Rao Geodesic Attention: Exact Constant-Curvature Kernels
            at Matmul Cost, with Data-Driven Geometry Selection},
  author = {Qui{\~n}elen, Eduardo},
  year   = {2026},
  note   = {Preprint},
}
```

## License

Code released under the [MIT License](LICENSE). The paper text (`paper_draft.md`) is © 2026 Eduardo Quiñelen.
