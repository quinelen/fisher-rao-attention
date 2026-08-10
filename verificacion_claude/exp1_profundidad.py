# ============================================================================
# EXPERIMENTO 1 (plan reordenado): ¿LA VENTAJA HIPERBÓLICA CRECE CON LA
# PROFUNDIDAD JERÁRQUICA?  — Dyck-4, precisión de cierre estratificada por
# profundidad de pila.
#
# *** PREDICCIÓN PRE-REGISTRADA (antes de ejecutar, 10-ago-2026) ***
# H1: la ventaja de Fisher-Rao sobre euclidiano CRECE con la profundidad del
#     stack: Δ(deep) > Δ(mid) > Δ(shallow).
# UMBRAL DE ÉXITO (de DeepSeek, aceptado): FR > Euclidiano en el bin DEEP
#     (profundidad ≥9) por >5 pp con p<0.01 (t pareado, n=5).
# RESULTADO NEGATIVO DECLARADO SI: Δ(deep) ≤ Δ(shallow) o p≥0.01.
# También se reporta el brazo HÍBRIDO (capas 1-2 FR + 3-4 euclidianas),
# motivado por la especialización por capa observada en exp_mixta.
#
# Desviaciones del plan de DeepSeek, con razón:
#  - n_heads=4 (d_h=32) en vez de 8 (d_h=16): mantiene d_h claramente sobre
#    el umbral de §4.7; con d_h=16 estaríamos en el borde del régimen malo.
#  - Sus "resultados esperados simulados" NO se usan: solo el umbral.
#
# Config: Dyck-4 (8 tipos de token, vocab 9), 4 capas, d128, entrenar
# L∈[50,100], extrapolar L∈[200,300]; 6000 muestras, 4 épocas, 5 semillas.
# ============================================================================
import torch, torch.nn as nn, numpy as np, time, json
import torch.optim as optim
from scipy import stats
from exp_tres_geometrias import Layer, EuclideanAttention
from fast_gaussian_test import IsotropicHyperbolicAttention

VOCAB = 9                      # 0 pad; abre: 1,3,5,7; cierra: 2,4,6,8
OPENS, CLOSES = [1, 3, 5, 7], [2, 4, 6, 8]
MATCH = {1: 2, 3: 4, 5: 6, 7: 8}

def gen_dyck4(length, rng):
    stack, seq = [], []
    for i in range(length):
        if not stack or (len(stack) < length - i and rng.rand() < 0.5):
            o = OPENS[rng.randint(4)]
            seq.append(o); stack.append(o)
        else:
            seq.append(MATCH[stack.pop()])
    return seq

def make_set(n, lo, hi, seed):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        L = rng.randint(lo // 2, hi // 2 + 1) * 2
        out.append(torch.tensor(gen_dyck4(L, rng)))
    return out

def collate(batch):
    m = max(len(x) for x in batch)
    out = torch.zeros(len(batch), m, dtype=torch.long)
    for i, s in enumerate(batch):
        out[i, :len(s)] = s
    return out

def batches(seqs, bs, shuffle=True):
    idx = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        yield collate([seqs[j] for j in idx[i:i + bs]])

def depths_of(seq):
    """Profundidad del stack ANTES de procesar cada token (para un cierre en t,
    es el número de brackets abiertos que incluye al que se cierra)."""
    d, out = 0, []
    for t in seq:
        if t in (1, 3, 5, 7):
            out.append(d + 1); d += 1
        elif t in (2, 4, 6, 8):
            out.append(d); d -= 1
        else:
            out.append(0)
    return out

class MultiModel(nn.Module):
    """Transformer con clase de atención POR CAPA (para el brazo híbrido)."""
    def __init__(self, attn_classes, vocab=VOCAB, d_model=128, n_heads=4, max_len=310):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([Layer(c, d_model, n_heads) for c in attn_classes])
        self.out = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, L = x.shape
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        h = self.emb(x) + self.pos(torch.arange(L, device=x.device).unsqueeze(0))
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.out(h)

ARMS = {
    "euclidean":  [EuclideanAttention] * 4,
    "fisher_rao": [IsotropicHyperbolicAttention] * 4,
    "hybrid":     [IsotropicHyperbolicAttention, IsotropicHyperbolicAttention,
                   EuclideanAttention, EuclideanAttention],
}
BINS = [("shallow", 1, 4), ("mid", 5, 8), ("deep", 9, 10**9)]

def eval_by_depth(model, seqs, bs=32):
    model.eval()
    acc = {b[0]: [0, 0] for b in BINS}
    close_set = torch.tensor(CLOSES)
    with torch.no_grad():
        for xb in batches(seqs, bs, shuffle=False):
            logits = model(xb[:, :-1])
            tgt = xb[:, 1:]
            # preferencia entre los 4 cierres
            close_logits = logits[..., CLOSES]              # [B,L,4]
            pref = close_set[close_logits.argmax(-1)]       # [B,L]
            for b in range(xb.size(0)):
                seq = xb[b].tolist()
                dep = depths_of(seq)
                for t in range(len(seq) - 1):
                    y = tgt[b, t].item()
                    if y in CLOSES:
                        d = dep[t + 1]
                        for name, lo, hi in BINS:
                            if lo <= d <= hi:
                                acc[name][0] += int(pref[b, t].item() == y)
                                acc[name][1] += 1
    return {k: (v[0] / v[1] if v[1] else float('nan'), v[1]) for k, v in acc.items()}

def train_arm(arm, seed, epochs=4, bs=32):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = make_set(6000, 50, 100, seed=42)
    tout = make_set(1500, 200, 300, seed=456)
    model = MultiModel(ARMS[arm])
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb in batches(tr, bs):
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    r = eval_by_depth(model, tout)
    r["secs"] = round(time.time() - t0)
    return r

if __name__ == "__main__":
    torch.set_num_threads(8)
    SEEDS = [1, 2, 3, 4, 5]
    res = {a: {b[0]: [] for b in BINS} for a in ARMS}
    counts = {}
    for seed in SEEDS:
        for arm in ARMS:
            r = train_arm(arm, seed)
            msg = " ".join(f"{k}={r[k][0]:.3f}(n={r[k][1]})" for k, _, _ in BINS)
            for k, _, _ in BINS:
                res[arm][k].append(r[k][0])
                counts[k] = r[k][1]
            print(f"seed {seed} {arm:<11}: {msg} ({r['secs']}s)", flush=True)

    print("\n" + "=" * 78)
    print(f"{'bin':>8} {'n_ejemplos':>10} {'euclidean':>14} {'fisher_rao':>14} {'hybrid':>14}")
    for k, _, _ in BINS:
        row = [f"{np.mean(res[a][k]):.4f}±{np.std(res[a][k]):.4f}" for a in ARMS]
        print(f"{k:>8} {counts[k]:>10} {row[0]:>14} {row[1]:>14} {row[2]:>14}")

    print("\n*** VEREDICTO DE LA PREDICCIÓN PRE-REGISTRADA ***")
    deltas = {}
    for k, _, _ in BINS:
        fr = np.array(res["fisher_rao"][k]); eu = np.array(res["euclidean"][k])
        t, p = stats.ttest_rel(fr, eu)
        deltas[k] = fr.mean() - eu.mean()
        print(f"  {k:>8}: Δ(FR-Euc)={deltas[k]:+.4f}  t={t:+.2f}  p={p:.4f}")
    fr_d = np.array(res["fisher_rao"]["deep"]); eu_d = np.array(res["euclidean"]["deep"])
    t_d, p_d = stats.ttest_rel(fr_d, eu_d)
    exito = deltas["deep"] > 0.05 and p_d < 0.01
    creciente = deltas["deep"] > deltas["mid"] > deltas["shallow"]
    print(f"\n  Umbral DeepSeek (Δdeep>+5pp y p<0.01): {'CUMPLIDO' if exito else 'NO CUMPLIDO'}")
    print(f"  H1 (ventaja creciente con profundidad): {'PATRÓN OBSERVADO' if creciente else 'NO OBSERVADO'}")
    # híbrido vs puro
    hy_d = np.array(res["hybrid"]["deep"])
    t_h, p_h = stats.ttest_rel(hy_d, fr_d)
    print(f"  Híbrido vs FR puro en deep: Δ={hy_d.mean()-fr_d.mean():+.4f}  p={p_h:.4f}")

    with open("resultados_exp1_profundidad.json", "w") as f:
        json.dump({a: res[a] for a in ARMS} | {"counts": counts}, f, indent=1)
    print("\nGuardado: resultados_exp1_profundidad.json")
