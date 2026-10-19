# ============================================================================
# P2 — PROSA NATURAL, PILOTO CPU (pre-registrado en PREREGISTRO_FASE2_CPU.md)
#
# Corpus NUEVO jamás usado: prosa técnica en inglés de las páginas man locales
# (~3 MB, renderizadas a texto plano, split 90/10 por página, determinista).
# Es prosa en lenguaje natural (no código); el salto a TinyStories/WikiText en
# GPU sigue pendiente como E2 del plan.
#
# Diseño idéntico a exp_codigo_extrapolativo: char-level, train L=128,
# eval {128, 256, 512}, 4000 seq, 3 épocas, d128/2capas/4cabezas.
# Brazos: euclidiano vs FR-ESTABILIZADO (cap mu=12). Semillas 46-55 (frescas).
# ENDPOINT PRIMARIO: bpc zona extrapolativa L=512, FR < Euc, p<0.05 (n=10).
# Secundarios: zona L=256; control IN L=128 (predicción: Euc gana IN);
# estabilidad sd(FR) <= 1.5 sd(Euc). NEGATIVO DECLARADO si primario falla.
# Ejecutar: ~/asreview/venv/bin/python exp_prosa_man.py
# ============================================================================
import torch, numpy as np, json, os, subprocess
from scipy import stats
from exp_codigo_extrapolativo import (seqs_of, run, TRAIN_LEN, EVAL_LENS,
                                       N_TRAIN_SEQ, N_EVAL_SEQ)
from exp_codigo_python import make_vocab, encode
from exp_tres_geometrias import EuclideanAttention
from exp_estabilizado import StableFR

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_TRAIN = os.path.join(HERE, "corpus_man_train.txt")
CACHE_TEST = os.path.join(HERE, "corpus_man_test.txt")
TARGET_BYTES = 3_000_000

def build_corpus_man():
    """Renderiza páginas man1 locales a texto plano (determinista, ordenado);
    split 90/10 por página. Cachea a disco para no re-renderizar."""
    if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
        with open(CACHE_TRAIN) as f1, open(CACHE_TEST) as f2:
            return f1.read(), f2.read()
    mandir = "/usr/share/man/man1"
    names = sorted(os.listdir(mandir))
    train_txt, test_txt, total, pages = [], [], 0, 0
    for i, fn in enumerate(names):
        stem = fn[:-3] if fn.endswith(".gz") else fn
        if stem.endswith(".1"):
            stem = stem[:-2]
        try:
            out = subprocess.run(
                ["sh", "-c", f"MANWIDTH=80 man -P cat {stem} 2>/dev/null | col -b"],
                capture_output=True, text=True, timeout=20).stdout
        except Exception:
            continue
        if len(out) < 1500:
            continue
        (test_txt if pages % 10 == 0 else train_txt).append(out)
        pages += 1; total += len(out)
        if total > TARGET_BYTES:
            break
    tr, te = "".join(train_txt), "".join(test_txt)
    with open(CACHE_TRAIN, "w") as f1, open(CACHE_TEST, "w") as f2:
        f1.write(tr); f2.write(te)
    print(f"Corpus man construido: {pages} páginas, train={len(tr)} test={len(te)} bytes")
    return tr, te

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus_man()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}
    print(f"Prosa man: train={len(train_txt)} test={len(test_txt)} vocab={VOCAB}")
    print(f"Train {Xtr.shape} (len {TRAIN_LEN}); eval {EVAL_LENS}\n")

    ARMS = {"euclidean": EuclideanAttention, "fr_estable": StableFR}
    SEEDS = list(range(46, 56))
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = run(cls, seed, Xtr, evals, VOCAB)
            res[name].append(r)
            print(f"seed {seed} {name:<11}: extra256={r['bpc_extra_L256']:.4f} "
                  f"extra512={r['bpc_extra_L512']:.4f} in128={r['bpc_L128']:.4f} ({r['secs']}s)",
                  flush=True)

    print("\n*** VEREDICTO PRE-REGISTRADO (prosa man, n=10, semillas 46-55) ***")
    for k, rol in [("bpc_extra_L512", "PRIMARIO"), ("bpc_extra_L256", "secundario"),
                   ("bpc_L128", "control IN")]:
        e = np.array([r[k] for r in res["euclidean"]])
        f_ = np.array([r[k] for r in res["fr_estable"]])
        t, p = stats.ttest_rel(f_, e)
        print(f"  [{rol:>10}] {k}: Euc={e.mean():.4f}±{e.std():.4f}  FRs={f_.mean():.4f}±{f_.std():.4f}  "
              f"Δ={f_.mean()-e.mean():+.4f}  t={t:+.2f}  p={p:.4f}")
    e512 = np.array([r["bpc_extra_L512"] for r in res["euclidean"]])
    f512 = np.array([r["bpc_extra_L512"] for r in res["fr_estable"]])
    t, p = stats.ttest_rel(f512, e512)
    exito = f512.mean() < e512.mean() and p < 0.05
    estable = f512.std() <= 1.5 * e512.std()
    print(f"\n  semillas ganadas por FR en extra_L512: {int((f512 < e512).sum())}/10")
    print(f"  estabilidad (sd_FR <= 1.5 sd_Euc): {'SÍ' if estable else 'NO'} "
          f"(sd_FR={f512.std():.4f}, sd_Euc={e512.std():.4f})")
    print(f"  ENDPOINT PRIMARIO: "
          f"{'CUMPLIDO — la transferencia extrapolativa alcanza PROSA natural (piloto CPU)' if exito else 'NO CUMPLIDO — negativo declarado: sin transferencia demostrada a prosa en esta escala'}")

    with open("resultados_prosa_man.json", "w") as fj:
        json.dump({a: res[a] for a in ARMS}, fj, indent=1)
    print("\nGuardado: resultados_prosa_man.json")
