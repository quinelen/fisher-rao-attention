# ============================================================================
# REPLICACIÓN PRE-REGISTRADA del test extrapolativo de código (n=10 fresco)
#
# Motivo: el experimento original (semillas 1-5) mostró el patrón cruzado
# predicho (FR peor IN-distribution, mejor en zona extrapolativa) con
# significancia en L=256 (p=0.027) pero SIN alcanzar el endpoint registrado
# (L=512, p=0.136). Declarar éxito con L=256 sería endpoint-switching.
# La resolución limpia es esta replicación independiente con más poder.
#
# *** PRE-REGISTRO (antes de ejecutar, 10-ago-2026) ***
# Semillas: 6-15 (diez, frescas, ninguna usada en el experimento original).
# ENDPOINT PRIMARIO (idéntico al original): bpc zona extrapolativa L=512,
#     FR < Euclidiano con p<0.05 (t pareado bilateral, n=10).
# SECUNDARIO (reportado, no decisorio): zona extrapolativa L=256.
# El análisis conjunto (n=15) se reportará como complementario, claramente
# etiquetado como agregación post-hoc.
# ============================================================================
import numpy as np, json, torch
from scipy import stats
from exp_codigo_extrapolativo import (build_corpus, make_vocab, encode,
                                       seqs_of, run, TRAIN_LEN, EVAL_LENS,
                                       N_TRAIN_SEQ, N_EVAL_SEQ)
from exp_tres_geometrias import EuclideanAttention
from fast_gaussian_test import IsotropicHyperbolicAttention

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}

    ARMS = {"euclidean": EuclideanAttention, "fisher_rao": IsotropicHyperbolicAttention}
    SEEDS = list(range(6, 16))
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = run(cls, seed, Xtr, evals, VOCAB)
            res[name].append(r)
            print(f"seed {seed:>2} {name:<11}: extra256={r['bpc_extra_L256']:.4f} "
                  f"extra512={r['bpc_extra_L512']:.4f} in128={r['bpc_L128']:.4f} ({r['secs']}s)", flush=True)

    print("\n*** VEREDICTO DE LA REPLICACIÓN (n=10, semillas 6-15) ***")
    for k, rol in [("bpc_extra_L512", "PRIMARIO"), ("bpc_extra_L256", "secundario"),
                   ("bpc_L128", "control IN")]:
        e = np.array([r[k] for r in res["euclidean"]])
        f_ = np.array([r[k] for r in res["fisher_rao"]])
        t, p = stats.ttest_rel(f_, e)
        print(f"  [{rol:>10}] {k}: Euc={e.mean():.4f}±{e.std():.4f}  FR={f_.mean():.4f}±{f_.std():.4f}  "
              f"Δ={f_.mean()-e.mean():+.4f}  t={t:+.2f}  p={p:.4f}")
    e512 = np.array([r["bpc_extra_L512"] for r in res["euclidean"]])
    f512 = np.array([r["bpc_extra_L512"] for r in res["fisher_rao"]])
    t, p = stats.ttest_rel(f512, e512)
    exito = f512.mean() < e512.mean() and p < 0.05
    print(f"\n  ENDPOINT PRIMARIO: {'CUMPLIDO — transferencia extrapolativa a código REPLICADA' if exito else 'NO CUMPLIDO — sin transferencia demostrada'}")

    # agregación post-hoc n=15 (etiquetada como tal)
    orig = json.load(open("resultados_codigo_extrapolativo.json"))
    e15 = np.concatenate([[r["bpc_extra_L512"] for r in orig["euclidean"]], e512])
    f15 = np.concatenate([[r["bpc_extra_L512"] for r in orig["fisher_rao"]], f512])
    t15, p15 = stats.ttest_rel(f15, e15)
    print(f"  [post-hoc n=15 conjunto] extra_L512: Δ={f15.mean()-e15.mean():+.4f}  t={t15:+.2f}  p={p15:.4f}")

    with open("resultados_codigo_extrapolativo_rep.json", "w") as fj:
        json.dump({a: res[a] for a in ARMS}, fj, indent=1)
    print("\nGuardado: resultados_codigo_extrapolativo_rep.json")
