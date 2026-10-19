# Fisher–Rao Geodesic Attention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23424909.svg)](https://doi.org/10.5281/zenodo.23424909)

**Exact constant-curvature attention kernels for sequence models at near-matmul cost.**

This repository accompanies the preprint [`paper_draft.md`](paper_draft.md). It contains everything needed to reproduce the attention experiments in the paper on a laptop CPU (~5 h total, no GPU required).

![Main results](verificacion_claude/fig_results.png)

## The idea in three sentences

Standard attention measures token similarity in an arbitrary flat vector space. We map queries and keys to probability distributions; choosing the distributional family is the one geometric design decision, after which Chentsov's theorem fixes the metric (Fisher–Rao, the unique one invariant under sufficient statistics): categorical distributions give a *spherical* manifold (K = +1/4), isotropic Gaussians give a genuinely *hyperbolic* space H^(d+1) (K = −1/2), and Euclidean distance attention arises as the *local* flat regime of the Gaussian family at fixed σ (the point-mass limit itself is singular). Both curved kernels compute exactly, in closed form, via one matrix multiplication plus an elementwise transcendental on an L×L matrix (2.3× the cost of a standard attention module; ≈1.4× end-to-end training time), and on hierarchical length-extrapolation (Dyck-2) accuracy ranks **Euclidean 85.8% < spherical 86.6% < hyperbolic-isotropic 88.4%** (parameter-matched, 5 paired seeds, p = 0.0009 for hyperbolic vs. Euclidean).

## Headline results

| Attention kernel | Curvature | Dyck-2 OUT accuracy | Params | Training time vs. standard |
| :--- | :---: | :---: | :---: | :---: |
| Euclidean (dot product) | 0 | 85.80% ± 0.65 | 413k | 1× |
| Spherical Fisher–Rao (exact) | +1/4 | 86.63% ± 0.64 | 413k | ≈1.1× |
| Ricci–Mahalanobis (quadratic approx.) | approx. | 86.87% ± 0.71 | 413k | ≈4× |
| Hyperbolic anisotropic (exact) | −1/2 | 87.58% ± 0.69 | 479k | ≈28× (module only) |
| **Hyperbolic isotropic (exact, ours)** | **< 0** | **88.43% ± 0.77** | **415k** | **≈1.4×** |
| Mixed-curvature adaptive (gated) | learned | 86.90% ± 0.72 | 418k | ≈1.8× |

**Extended evidence (§4.8):** the headline contrast holds over **20 paired seeds** (Δ=+2.10 pp, p=1.9×10⁻¹⁰, 20/20) and **widens at 46× model scale** (+4.7 pp at 19M params, 3/3 seeds). The advantage does **not** grow with hierarchy depth (present at all depths, front-loaded shallow–mid; pre-registered H1 refuted), and in-distribution modeling of natural data (code and prose) consistently favors Euclidean attention. **Far length-extrapolation on natural data transfers.** On real Python code, after two pre-registered failures of the original kernel traced to outlier μ norms at unseen positions, a one-line μ-norm ball cap met the pre-registered endpoint at n=20 seeds new to the code task (16–35; Δ=−0.084 bpc, p=0.0037, 15/20); all three attempts are reported (§4.8d). On a previously unused corpus of technical English prose, the same kernel met a separately pre-registered endpoint at the first attempt (n=10, Δ=−0.058 bpc, p=0.0002, 10/10 seeds; §4.8e).

Further findings (Sections 4.5–4.8, 5–5.3 and 6 of the paper):

- **The gain comes from the global hyperbolic geometry, not from Gaussian tokens or local information geometry (§4.6).** Holding the Gaussian token representation fixed and swapping only the score, the Wasserstein–Bures kernel (the geometry used by prior stochastic transformers, arXiv:2311.18645) gains +1.12 pp over point-token attention, and the Fisher–Rao geometry adds a further **+1.52 pp** (p = 0.029). A KL (Bregman) kernel, which shares the Fisher metric locally but lacks its global negative curvature, reaches 87.32%: Fisher–Rao beats it by **+1.11 pp** (95% CI [0.44, 1.78], p = 0.010, 5/5 seeds).
- **The advantage does not depend on absolute positional embeddings (§4.8f).** With relative (rotary) positions in both arms — an exact isometry of the hyperbolic space for the Fisher–Rao kernel — a pre-registered test on 20 paired seeds (6–25) keeps the lead on Dyck-2 (92.73% vs. 90.91%, +1.82 pp, p = 6×10⁻⁵, 16/20), equally within and beyond the training length, and on the copy task the lead beyond the training length is +5.60 pp (p = 0.0002), above the +2.79 pp within it.
- **A positive-definite variant keeps most of the advantage (§4.5).** The geodesic Gaussian kernel exp(−βd²) is not positive definite on curved spaces, but the geodesic Laplacian exp(−βd) is on hyperbolic space (Feragen et al., CVPR 2015). Swapping it in gives 87.88%: +2.08 pp over Euclidean (p = 0.008) and a non-significant −0.55 pp versus the Gaussian form — keeping open a route to linear-time variants.
- **A threshold-gated efficiency result (§4.7).** Above a head-width threshold ($d_h \ge 16$), the hyperbolic kernel at dimension $D$ shows no significant accuracy difference from Euclidean attention at dimension $2D$ ($p = 0.27$ and $0.74$; 95% CIs within ±1 pp) while using **≈3.8–3.9× fewer parameters**, and at the largest scale tested (d_model 128 vs. 256) trains **roughly 1.5× faster** — because the quadratic $O(d_{\text{model}}^2)$ projection cost dominates at scale. Below the threshold ($d_h \le 8$) the effect reverses and hyperbolic attention is significantly worse; this is not a universal speedup.
- **Geometry selection follows data structure (§5).** A gated mixture of the three kernels — re-instantiating the adaptive mixed-curvature idea of CAT (arXiv:2510.01634), which is prior work — allocates 0.42 of its gate weight to the hyperbolic kernel on hierarchical data and collapses it to 0.09 on flat data (Euclidean gate: 0.28 → 0.64). We use the gate as a diagnostic probe, not as a claimed novelty.
- **A pre-specified negative result (§5.1).** On a mixed-structure stress test the adaptive mixture *lost* to the pure hyperbolic kernel (p = 0.0013). One reading — consistent with the data but not yet demonstrated by it — is that negative curvature acts as a **superset geometry**, emulating flat attention at small scales while the converse is impossible; establishing that reading would require robustness across systematically varied task structures, and part of the mixture's loss is attributable to a fixable gate-design weakness. What the data do support: the explicit gate added no accuracy over the pure hyperbolic kernel on these tasks.
- **A tie with the curvature-adaptive incumbent on sequences (§5.3).** Against a reimplementation of CAT's central mechanism (arXiv:2510.01634; per-token routing among Euclidean, Poincaré and spherical branches, tuned to avoid a strawman), the pure Fisher–Rao kernel ties on Dyck-2 (88.43% vs. 88.13%, p = 0.53) at ≈70% of its training time; CAT-style itself beats Euclidean attention (+2.34 pp, p = 0.0071).
- **A second negative result against the real incumbents (§5.2).** On the canonical WordNet-mammals embedding benchmark, the Fisher–Rao kernel crushes Euclidean (+0.49 MAP at D=10) but is **beaten by Poincaré embeddings** (0.883 vs 0.941, p = 0.0002; our Poincaré reimplementation reproduces the published MAP ≈ 0.93). The kernel's value is specific to *sequence attention* (§4); it is **not** a better hierarchy embedder and does not compete with dedicated hyperbolic graph methods in the ontology/KG domains where those are deployed. This bounds the applied reading of the work.
- **Two further applied hypotheses tested and refuted (out of scope of the attention mechanism).** The scale parameter $\sigma$ is not a usable predictive-uncertainty signal (AUC-ROC ≈ 0.49, chance level). In few-shot hierarchical classification, the Fisher–Rao geometry is *outperformed* by both Euclidean and Poincaré at low sample counts (p = 0.0003 vs. Poincaré at k = 10/class) — the opposite of what hyperbolic geometry's known embedding data-efficiency would suggest.

## Repository structure

```
.
├── paper_draft.md                  # The preprint (all claims cross-referenced to scripts below)
├── arxiv/                          # LaTeX source of the arXiv version (main.tex), abstract and figures
├── README.md
├── LICENSE                         # MIT (code); paper text © the authors
├── CITATION.cff, .zenodo.json      # citation and archive metadata
├── requirements.txt
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
    ├── exp_eficiencia_dimensional.py  # §4.7 parameter/compute efficiency sweep  (~20 min)
    ├── exp_mecanismo_umbral.py     # §4.7 pre-registered probe of the threshold mechanism  (~3 min)
    ├── exp_ablacion_kernel.py      # §4.5–4.6 KL/Bregman and geodesic-Laplacian ablations  (~15 min)
    ├── exp_cat_incumbente.py       # §5.3 CAT-style reimplementation, pilot tuning, head-to-head  (~20 min)
    │
    │   # --- Extended evidence (§4.8) ---
    ├── exp_respuesta_escala.py     # §4.8a–b 20-seed extension and 19M-parameter scale probe
    ├── exp1_profundidad.py         # §4.8c pre-registered depth-stratified Dyck-4 test
    ├── exp_codigo_python.py        # §4.8d in-distribution character-level Python modeling
    ├── exp_codigo_extrapolativo.py, exp_codigo_extrapolativo_rep.py
    │                               # §4.8d pre-registered code length-extrapolation (original kernel)
    ├── diag_sigma_extrapolacion.py, diag_sigma_ronda2.py
    │                               # §4.8d failure-mode diagnosis (outlier μ norms); output in log_diag_sigma_*.txt
    ├── exp_estabilizado.py         # §4.8d stabilized kernel (μ-norm cap), pre-registered n=20
    ├── manifiesto_corpus_codigo.py/.json  # file list, Python version and SHA-256 of the §4.8d code corpus
    ├── exp_sensibilidad_cap.py     # §4.8d cap-robustness analysis
    ├── exp_rope.py                 # §4.8f pre-registered relative-position (RoPE) test
    ├── PREREGISTRO_ROPE.md, SELLO_PREREGISTRO_ROPE.txt  # its protocol and pre-run SHA-256 seal
    ├── exp_prosa_man.py            # §4.8e pre-registered English-prose length-extrapolation
    │                               #   (corpus rendered from the local system manual)
    ├── manifiesto_corpus_man.py/.json  # page list, OS version and SHA-256 of that corpus
    ├── PREREGISTRO_FASE2_CPU.md    # October 2026 pre-registrations (prose, CAT, threshold probe)
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
    ├── resultados_ablacion_kernel.json, resultados_cat_incumbente.json,
    │   resultados_prosa_man.json, resultados_mecanismo_umbral.json, ...
    ├── log_*.txt                   # console output of the October 2026 runs and of the §4.8d diagnosis
    ├── fig_results.png             # results figure of the paper (plotted from the result JSONs; plotting script not included)
    ├── figuras_geometria.py        # geometry figures and the interactive scene (EN/ES text; same data and seeds)
    ├── fig_geometrias_3d.png, fig_semiespacio_hiperbolico.png, fig_tokens_reales_esfera.png,
    │   geometrias_interactivas.html, tokens_esfera_*.npy   # their outputs and the stored trained-token data (the training/extraction script for tokens_esfera_*.npy is not included)
    │
    │   # --- Applied follow-up (out of scope of the attention mechanism;
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
    └── resultados_*.txt, multi_seed_results.json
```

## Reproducing the benchmark

Requires Python ≥ 3.9. CPU is sufficient (all reported numbers were produced on a laptop CPU). The §4.8d code corpus is read from the standard library of the running interpreter; to reproduce the reported numbers exactly, use CPython 3.9.6 (SHA-256 manifest in `verificacion_claude/manifiesto_corpus_codigo.json`; `python manifiesto_corpus_codigo.py` checks a local interpreter against it).

```bash
pip install -r requirements.txt
cd verificacion_claude

# Main 3-geometry benchmark (5 seeds, paired stats)        ~30 min
python exp_tres_geometrias.py

# 4th arm: Ricci–Mahalanobis approximation                 ~25 min
python exp_ricci_addendum.py

# Isotropic hyperbolic kernel + speed micro-benchmark      ~15 min
python fast_gaussian_test.py bench    # speed table (ms/step; printed only)
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

# KL/Bregman and geodesic-Laplacian kernel ablations (§4.5–4.6)   ~15 min
python exp_ablacion_kernel.py

# CAT-style head-to-head (§5.3)                             ~20 min
python exp_cat_incumbente.py

# Natural data (§4.8d–e): stabilized kernel on code, then on technical prose
python exp_estabilizado.py        # ~25 min
python exp_prosa_man.py           # ~10 min (renders the corpus from local man pages)

# Relative positions (RoPE) in both arms, Dyck-2 and copy (§4.8f)   ~45 min
python exp_rope.py
```

Each script prints per-seed results and summary statistics and, except the speed benchmark (`bench`) and the two `diag_sigma_*` diagnostics, writes a `resultados_*.json` with the raw numbers. Training seeds are fixed in each script — 1–5 for the Dyck-2 and WordNet benchmarks (1–3 for the copy arms of `exp_mixta.py`), 16–35 for `exp_estabilizado.py`, 46–55 for `exp_prosa_man.py` and 6–25 for `exp_rope.py`; the other scripts in the tree state theirs in-line (e.g. 6–20 in `exp_respuesta_escala.py`, 36–40 in `exp_sensibilidad_cap.py`, 6–8 in `exp_mecanismo_umbral.py`). Data-generation seeds and every hyperparameter are also fixed in-line, so runs are deterministic on CPU.

### Using the kernel in your own model

The paper's main mechanism is self-contained in one class — `IsotropicHyperbolicAttention` in [`verificacion_claude/fast_gaussian_test.py`](verificacion_claude/fast_gaussian_test.py) (~40 lines). It is a drop-in replacement for a multi-head attention block: same input/output shapes, causal mask supported, one extra hyperparameter (`eps_sigma`). The exact Fisher–Rao distance between the isotropic Gaussian tokens is

```
d(i,j) = √2 · arccosh( 1 + (‖μᵢ−μⱼ‖²/2 + (σᵢ−σⱼ)²) / (2σᵢσⱼ) )
```

with the pairwise `‖μᵢ−μⱼ‖²` computed by a single matmul.

## Honest limitations

CPU scale throughout. The headline contrast rests on 20 paired seeds and replicates at 19M parameters; natural-data transfer is shown for length extrapolation on real code (n=20) and technical English prose (n=10), but only as character-level CPU pilots — general-domain text, token-level modeling and GPU scale are untested, and in-distribution natural-data modeling consistently favors Euclidean attention. Except for the relative-position test (§4.8f), results use learned absolute positional embeddings; the natural-data experiments have not been re-run with relative positions. Only the isotropic- and anisotropic-vs-Euclidean contrasts survive Bonferroni correction. The efficiency result (§4.7) is threshold-gated: it holds for $d_h \ge 16$ and **reverses** below that (hyperbolic is significantly worse at small head width). Several other pre-specified hypotheses were tested and **failed**: geometry gating is unnecessary (§5.1), σ is not a usable uncertainty estimator (AUC ≈ 0.49), the kernel is beaten by Poincaré embeddings on static hierarchy embedding (§5.2), and it is outperformed by Euclidean and Poincaré alike in few-shot hierarchical classification at low sample counts — opposite to what hyperbolic geometry's embedding-side data efficiency would suggest. The demonstrated value is confined to sequence attention, above a scale threshold. See Section 6 of the paper before extrapolating any claim; this is a bounded workshop-scale contribution, not a general advance.

## Citation

```bibtex
@misc{quinelen2026fisherrao,
  doi    = {10.5281/zenodo.23424909},
  title  = {Fisher--Rao Geodesic Attention: Exact Constant-Curvature Kernels for Sequence Models
            at Near-Matmul Cost},
  author = {Qui{\~n}elen Rojas, Eduardo and Contreras Torrejon, Miguel},
  year   = {2026},
  note   = {Preprint. Code and data: https://github.com/quinelen/fisher-rao-attention},
}
```

## License

Code released under the [MIT License](LICENSE). The paper text (`paper_draft.md`) is © 2026 Eduardo Quiñelen Rojas and Miguel Contreras Torrejon.
