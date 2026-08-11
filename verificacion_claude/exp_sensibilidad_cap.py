# ============================================================================
# ANÁLISIS DE SENSIBILIDAD DEL CAP (respuesta a la crítica del "pecado
# original"): ¿el resultado de §4.8d depende finamente de c=12, o la
# estabilización es robusta a la elección del radio?
# NO es un test confirmatorio nuevo (el claim queda anclado en c=12, n=20);
# es un análisis de robustez descriptivo con c ∈ {8, 16}, 5 semillas frescas
# (36-40) jamás usadas. Si la dirección (FR mejor en zonas extrapolativas)
# se mantiene en ambos caps, c=12 no es un número mágico.
# ============================================================================
import torch, numpy as np, json
from scipy import stats
import exp_estabilizado as E
from exp_codigo_extrapolativo import (build_corpus, make_vocab, encode,
                                       seqs_of, run, TRAIN_LEN, EVAL_LENS,
                                       N_TRAIN_SEQ, N_EVAL_SEQ)
from exp_tres_geometrias import EuclideanAttention

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}

    SEEDS = [36, 37, 38, 39, 40]
    res = {"euclidean": [], "cap8": [], "cap16": []}
    for seed in SEEDS:
        r = run(EuclideanAttention, seed, Xtr, evals, VOCAB)
        res["euclidean"].append(r)
        print(f"seed {seed} euclidean: extra256={r['bpc_extra_L256']:.4f} extra512={r['bpc_extra_L512']:.4f}", flush=True)
        for cap, name in [(8.0, "cap8"), (16.0, "cap16")]:
            E.MU_CAP = cap
            r = run(E.StableFR, seed, Xtr, evals, VOCAB)
            res[name].append(r)
            print(f"seed {seed} FR c={cap:<4}: extra256={r['bpc_extra_L256']:.4f} extra512={r['bpc_extra_L512']:.4f}", flush=True)

    print("\n*** SENSIBILIDAD AL CAP (n=5, semillas 36-40; c=12 de referencia: n=20, Δ512=-0.084) ***")
    for name in ("cap8", "cap16"):
        for k in ("bpc_extra_L512", "bpc_extra_L256"):
            e = np.array([r[k] for r in res["euclidean"]])
            f_ = np.array([r[k] for r in res[name]])
            t, p = stats.ttest_rel(f_, e)
            print(f"  {name:>6} {k}: Δ={f_.mean()-e.mean():+.4f}  t={t:+.2f}  p={p:.4f}  "
                  f"({int((f_<e).sum())}/5 semillas FR mejor)")
    with open("resultados_sensibilidad_cap.json", "w") as fj:
        json.dump({a: [{k: r[k] for k in r} for r in v] for a, v in res.items()}, fj, indent=1)
    print("\nGuardado: resultados_sensibilidad_cap.json")
