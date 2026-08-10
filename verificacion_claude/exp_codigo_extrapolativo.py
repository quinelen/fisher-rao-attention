# ============================================================================
# EL TEST DECISIVO: EXTRAPOLACIÓN DE LONGITUD EN CÓDIGO PYTHON REAL
#
# Contexto: §4.8d mostró que FR es peor que euclidiano en código real
# DENTRO de distribución (train y eval a 256). Pero toda la ventaja
# demostrada de FR es EXTRAPOLATIVA (Dyck: train 20-60, eval 80-120).
# Este es el test justo: entrenar corto, evaluar largo, en datos naturales.
#
# *** PREDICCIÓN PRE-REGISTRADA (antes de ejecutar, 10-ago-2026) ***
# H1: en la ZONA EXTRAPOLATIVA (posiciones > longitud de entrenamiento),
#     FR supera a euclidiano en bpc, aunque dentro de distribución empate
#     o pierda (patrón Dyck).
# UMBRAL DE ÉXITO: bpc(FR) < bpc(Euc) en la zona extrapolativa de L=512
#     con p<0.05 (t pareado, n=5 semillas).
# NEGATIVO DECLARADO SI: FR no es mejor con p<0.05, o es peor.
# CONSECUENCIA PRE-ACORDADA: positivo => se abre la vía comercial/GPU
#     (con HypeCodeNet como comparación obligada); negativo => el paper
#     mantiene su §4.8d y el claim queda confinado a lenguajes formales.
#
# Diseño:
#  - Corpus: stdlib local de Python (mismo de exp_codigo_python).
#  - TRAIN: secuencias de 128 chars.  EVAL: 128 (control IN), 256, 512.
#  - Métricas: bpc total por longitud + bpc SOLO en posiciones >=128
#    (zona extrapolativa pura) para L=256 y L=512.
#  - Ambos brazos sufren el mismo handicap de posiciones no entrenadas
#    (embedding posicional absoluto) — comparación pareada justa, igual
#    que en el protocolo Dyck del paper.
#  - d128, 2 capas, 4 cabezas, 4000 seq, 3 épocas, 5 semillas.
# ============================================================================
import torch, torch.nn as nn, numpy as np, time, json, math
import torch.optim as optim
from scipy import stats
from exp_tres_geometrias import Model, EuclideanAttention
from fast_gaussian_test import IsotropicHyperbolicAttention
from exp_codigo_python import build_corpus, make_vocab, encode

TRAIN_LEN = 128
EVAL_LENS = [128, 256, 512]
MAX_LEN = 512
EPOCHS, BS = 3, 32
N_TRAIN_SEQ, N_EVAL_SEQ = 4000, 400

def seqs_of(arr, n, L, rng):
    starts = rng.randint(0, len(arr) - L - 1, size=n)
    return np.stack([arr[s:s + L] for s in starts])

def bpc_zones(model, X, vocab, from_pos=0, bs=16):
    """bpc sobre posiciones de target >= from_pos (0 = todas)."""
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    tot = ntok = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs])
            logits = model(xb[:, :-1])
            tgt = xb[:, 1:].clone()
            if from_pos > 0:
                tgt[:, :from_pos] = 0        # ignora la zona ya vista en entrenamiento
            tot += crit(logits.reshape(-1, vocab), tgt.reshape(-1)).item()
            ntok += (tgt != 0).sum().item()
    return tot / ntok / math.log(2)

def run(attn_cls, seed, Xtr, evals, vocab):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = Model(attn_cls, vocab=vocab, d_model=128, n_layers=2,
                  n_heads=4, max_len=MAX_LEN)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), BS):
            xb = torch.from_numpy(Xtr[perm[i:i + BS]])
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, vocab), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    out = {}
    for L, X in evals.items():
        out[f"bpc_L{L}"] = bpc_zones(model, X, vocab, from_pos=0)
        if L > TRAIN_LEN:
            out[f"bpc_extra_L{L}"] = bpc_zones(model, X, vocab, from_pos=TRAIN_LEN - 1)
    out["secs"] = round(time.time() - t0)
    return out

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}
    print(f"Corpus: train={len(train_txt)} test={len(test_txt)} vocab={VOCAB}")
    print(f"Train: {Xtr.shape} (len {TRAIN_LEN}); eval lens {EVAL_LENS}\n")

    ARMS = {"euclidean": EuclideanAttention, "fisher_rao": IsotropicHyperbolicAttention}
    SEEDS = [1, 2, 3, 4, 5]
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = run(cls, seed, Xtr, evals, VOCAB)
            res[name].append(r)
            msg = " ".join(f"{k}={v:.4f}" for k, v in r.items() if k.startswith("bpc"))
            print(f"seed {seed} {name:<11}: {msg} ({r['secs']}s)", flush=True)

    print("\n" + "=" * 84)
    keys = [k for k in res["euclidean"][0] if k.startswith("bpc")]
    print(f"{'métrica':>16} {'euclidean':>18} {'fisher_rao':>18} {'Δ(FR-Euc)':>12} {'p':>9}")
    verdict = {}
    for k in keys:
        e = np.array([r[k] for r in res["euclidean"]])
        f_ = np.array([r[k] for r in res["fisher_rao"]])
        t, p = stats.ttest_rel(f_, e)
        verdict[k] = (f_.mean() - e.mean(), p)
        print(f"{k:>16} {e.mean():>10.4f}±{e.std():.4f} {f_.mean():>10.4f}±{f_.std():.4f} "
              f"{f_.mean()-e.mean():>+12.4f} {p:>9.4f}")

    print("\n*** VEREDICTO DE LA PREDICCIÓN PRE-REGISTRADA ***")
    d512, p512 = verdict["bpc_extra_L512"]
    exito = d512 < 0 and p512 < 0.05
    print(f"  Zona extrapolativa L=512: Δ={d512:+.4f} bpc (negativo=FR mejor), p={p512:.4f}")
    print(f"  Umbral (FR mejor, p<0.05): {'CUMPLIDO — LA VENTAJA EXTRAPOLATIVA TRANSFIERE A CÓDIGO' if exito else 'NO CUMPLIDO — sin transferencia demostrada a datos naturales'}")

    with open("resultados_codigo_extrapolativo.json", "w") as f2:
        json.dump({a: res[a] for a in ARMS}, f2, indent=1)
    print("\nGuardado: resultados_codigo_extrapolativo.json")
