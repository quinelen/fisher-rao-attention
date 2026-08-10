# ============================================================================
# DATOS NATURALES: modelado de lenguaje a nivel de caracteres sobre CÓDIGO
# PYTHON REAL (biblioteca estándar local — no sintético).
#
# El código fuente tiene jerarquía genuina: indentación anidada, paréntesis,
# corchetes, llaves, defs dentro de clases dentro de módulos. Es el puente
# natural entre Dyck-2 (sintético) y lenguaje natural pleno.
#
# Protocolo: LM autoregresivo por caracteres, secuencias de 256 chars,
# corpus ~1MB de .py de la stdlib (split determinista por archivo 90/10),
# d128, 2 capas, 4 cabezas (config §4.2), 3 épocas, 3 semillas por brazo.
# Métrica: bits por carácter (bpc) en held-out. Menor = mejor.
#
# Ejecutar: ~/asreview/venv/bin/python exp_codigo_python.py
# ============================================================================
import torch, torch.nn as nn, numpy as np, time, json, os, glob, math
import torch.optim as optim
from scipy import stats
from exp_tres_geometrias import Model, EuclideanAttention
from fast_gaussian_test import IsotropicHyperbolicAttention

SEQ_LEN = 256
D_MODEL, N_LAYERS, N_HEADS = 128, 2, 4
EPOCHS, BS = 3, 32
CORPUS_BYTES = 1_200_000

def build_corpus():
    """Concatena .py de la stdlib local (determinista, ordenado)."""
    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(stdlib, "*.py")))
    train_txt, test_txt = [], []
    total = 0
    for i, fp in enumerate(files):
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                t = f.read()
        except OSError:
            continue
        if len(t) < 2000:
            continue
        (test_txt if i % 10 == 0 else train_txt).append(t)   # split 90/10 por archivo
        total += len(t)
        if total > CORPUS_BYTES:
            break
    return "".join(train_txt), "".join(test_txt)

def make_vocab(text):
    chars = sorted(set(text))
    # 0 reservado para PAD/UNK
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    return stoi, len(stoi) + 1

def encode(text, stoi):
    return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)

def make_sequences(arr, n, rng):
    starts = rng.randint(0, len(arr) - SEQ_LEN - 1, size=n)
    return np.stack([arr[s:s + SEQ_LEN] for s in starts])

def bpc(model, X, vocab, bs=BS):
    model.eval(); crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    tot_loss = tot_tok = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs])
            logits = model(xb[:, :-1])
            tgt = xb[:, 1:]
            tot_loss += crit(logits.reshape(-1, vocab), tgt.reshape(-1)).item()
            tot_tok += (tgt != 0).sum().item()
    return tot_loss / tot_tok / math.log(2)   # nats -> bits

def run(attn_cls, seed, Xtr, Xte, vocab):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = Model(attn_cls, vocab=vocab, d_model=D_MODEL, n_layers=N_LAYERS,
                  n_heads=N_HEADS, max_len=SEQ_LEN)
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
    return {"bpc": bpc(model, Xte, vocab),
            "params": sum(p.numel() for p in model.parameters()),
            "secs": round(time.time() - t0)}

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    print(f"Corpus: train={len(train_txt)} chars, test={len(test_txt)} chars, vocab={VOCAB}")
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = make_sequences(tr_arr, 4000, rng0)
    Xte = make_sequences(te_arr, 800, rng0)
    print(f"Secuencias: train={Xtr.shape}, test={Xte.shape} (len {SEQ_LEN})\n")

    res = {"euclidean": [], "fisher_rao": []}
    for seed in [1, 2, 3]:
        for name, cls in [("euclidean", EuclideanAttention),
                          ("fisher_rao", IsotropicHyperbolicAttention)]:
            r = run(cls, seed, Xtr, Xte, VOCAB)
            res[name].append(r)
            print(f"seed {seed} {name:<11}: bpc={r['bpc']:.4f}  params={r['params']} ({r['secs']}s)", flush=True)

    e = np.array([r["bpc"] for r in res["euclidean"]])
    f_ = np.array([r["bpc"] for r in res["fisher_rao"]])
    t, p = stats.ttest_rel(f_, e)
    print(f"\nCódigo Python (bpc, menor=mejor): euclidiana={e.mean():.4f}±{e.std():.4f}  "
          f"fisher-rao={f_.mean():.4f}±{f_.std():.4f}")
    print(f"Contraste pareado (n=3): Δ={f_.mean()-e.mean():+.4f} bpc  t={t:+.2f}  p={p:.4f}")
    print("(Δ negativo = fisher-rao mejor)")
    with open("resultados_codigo_python.json", "w") as f2:
        json.dump({k: [r["bpc"] for r in v] for k, v in res.items()}, f2, indent=1)
    print("Guardado: resultados_codigo_python.json")
