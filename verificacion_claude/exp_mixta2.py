# ============================================================================
# EXPERIMENTO FINAL: tarea de ESTRUCTURA MIXTA
#   Secuencia = [5] bloque-Dyck [6] bloque-Copy ( [5] dyck [6] copy ... )
#   - bloques Dyck-2: estructura jerárquica (cierre correcto requiere stack)
#   - bloques Copy:   estructura plana (mitad aleatoria + copia exacta)
#   Train: 1 par de bloques (long. total 22-58). OUT: 2 pares (76-116).
#   Vocab: 0 pad, 1-4 tokens, 5 marcador-dyck, 6 marcador-copy.
#
# Hipótesis: la MIX adaptativa debe (a) ganar a las tres geometrías puras en la
# métrica combinada, y (b) oscilar sus gates DENTRO de la secuencia:
# más hiperbólica en bloques Dyck, más euclidiana en bloques Copy.
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np, math, time, json
from scipy import stats
from exp_tres_geometrias import (Model, EuclideanAttention,
                                 SphericalFisherRaoAttention, collate_dyck)
from fast_gaussian_test import IsotropicHyperbolicAttention
from exp_mixta import MixedCurvatureAttention

VOCAB = 7
M_DYCK, M_COPY = 5, 6

def gen_dyck_block(length, rng):
    stack, seq, oc = [], [], 0
    tok = {'(': 1, ')': 2, '[': 3, ']': 4}
    for i in range(length):
        if oc == 0 or (oc < length - i and rng.rand() < 0.5):
            t = rng.choice(['(', '['])
            seq.append(tok[t]); stack.append(t); oc += 1
        else:
            t = stack.pop()
            seq.append(tok[')' if t == '(' else ']']); oc -= 1
    return seq

class MixedStructureDataset(IterableDataset):
    def __init__(self, num_samples, n_pairs, min_block, max_block, seed=None):
        self.n, self.pairs = num_samples, n_pairs
        self.a, self.b, self.seed = min_block, max_block, seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for _ in range(self.n):
            seq = []
            for _ in range(self.pairs):
                dl = rng.randint(self.a // 2, self.b // 2 + 1) * 2
                seq.append(M_DYCK); seq.extend(gen_dyck_block(dl, rng))
                ch = rng.randint(self.a // 2, self.b // 2 + 1)
                first = rng.randint(1, 5, size=ch).tolist()
                seq.append(M_COPY); seq.extend(first + first)
            yield torch.tensor(seq)

def regions(seq):
    """Devuelve lista de (tipo, inicio, fin) escaneando marcadores; índices sobre seq."""
    out, i, cur, start = [], 0, None, None
    for i, t in enumerate(seq):
        if t in (M_DYCK, M_COPY):
            if cur is not None:
                out.append((cur, start, i))
            cur, start = ('dyck' if t == M_DYCK else 'copy'), i + 1
        elif t == 0:
            break
    if cur is not None:
        end = i if seq[i] == 0 else i + 1
        out.append((cur, start, end))
    return out

def eval_mixed(model, loader, collect_gates=False):
    """Métricas por región + (opcional) gates por región para MIX."""
    model.eval()
    dc = dt = cc = ct = 0
    gsum = {"dyck": np.zeros(3), "copy": np.zeros(3)}; gn = {"dyck": 0, "copy": 0}
    with torch.no_grad():
        for seq, _ in loader:
            logits = model(seq[:, :-1]); tgt = seq[:, 1:]
            pred = logits.argmax(-1)
            pref = torch.where(logits[..., 2] >= logits[..., 4],
                               torch.tensor(2), torch.tensor(4))
            if collect_gates:
                g = torch.stack([lyr.attn.last_gate_full for lyr in model.layers]
                                ).mean(dim=(0, 2))          # [B, L, 3]
            for b in range(seq.size(0)):
                s = seq[b].tolist()
                for typ, i0, i1 in regions(s):
                    # posiciones de target dentro de la región: t predice s[t+1]
                    for t in range(max(i0 - 1, 0), i1 - 1):
                        y = tgt[b, t].item()
                        if typ == 'dyck' and y in (2, 4):
                            dc += int(pref[b, t].item() == y); dt += 1
                        if typ == 'copy':
                            h = (i1 - i0) // 2
                            if t + 1 >= i0 + h:              # segunda mitad: predecible
                                cc += int(pred[b, t].item() == y); ct += 1
                    if collect_gates:
                        span = g[b, max(i0 - 1, 0):i1 - 1]
                        if len(span):
                            gsum[typ] += span.mean(0).numpy(); gn[typ] += 1
    r = {"dyck_acc": dc / max(dt, 1), "copy_acc": cc / max(ct, 1)}
    r["combined"] = 0.5 * (r["dyck_acc"] + r["copy_acc"])
    if collect_gates:
        r["gates"] = {k: (gsum[k] / max(gn[k], 1)).round(3).tolist() for k in gsum}
    return r

def train_arm(attn_cls, seed, epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(MixedStructureDataset(10000, 1, 10, 28, seed=42),
                    batch_size=bs, collate_fn=collate_dyck)
    tin = DataLoader(MixedStructureDataset(2000, 1, 10, 28, seed=123),
                     batch_size=bs, collate_fn=collate_dyck)
    tout = DataLoader(MixedStructureDataset(2000, 2, 18, 28, seed=456),
                      batch_size=bs, collate_fn=collate_dyck)
    model = Model(attn_cls, vocab=VOCAB)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for seq, _ in tr:
            logits = model(seq[:, :-1])
            loss = crit(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    is_mix = attn_cls is MixedCurvatureAttention
    rin = eval_mixed(model, tin, collect_gates=is_mix)
    rout = eval_mixed(model, tout, collect_gates=False)
    return {"in": rin, "out": rout, "secs": round(time.time() - t0)}

if __name__ == "__main__":
    torch.set_num_threads(8)
    ARMS = {
        "euclidiana": EuclideanAttention,
        "esferica": SphericalFisherRaoAttention,
        "isotropica": IsotropicHyperbolicAttention,
        "mix": MixedCurvatureAttention,
    }
    SEEDS = [1, 2, 3, 4, 5]
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = train_arm(cls, seed)
            res[name].append(r)
            gmsg = ""
            if "gates" in r["in"]:
                gd, gc = r["in"]["gates"]["dyck"], r["in"]["gates"]["copy"]
                gmsg = (f"  gates dyck e/s/h={gd[0]:.2f}/{gd[1]:.2f}/{gd[2]:.2f}"
                        f"  copy e/s/h={gc[0]:.2f}/{gc[1]:.2f}/{gc[2]:.2f}")
            print(f"seed {seed} | {name:<11}: IN comb={r['in']['combined']:.1%} "
                  f"(dyck {r['in']['dyck_acc']:.1%} / copy {r['in']['copy_acc']:.1%}) | "
                  f"OUT comb={r['out']['combined']:.1%} "
                  f"(dyck {r['out']['dyck_acc']:.1%} / copy {r['out']['copy_acc']:.1%}) "
                  f"({r['secs']}s){gmsg}", flush=True)

    print("\n" + "=" * 90)
    print(f"{'brazo':<12} {'IN combinada':>18} {'OUT combinada':>18} {'OUT dyck':>12} {'OUT copy':>12}")
    print("=" * 90)
    summ = {}
    for name, rs in res.items():
        ic = np.array([r["in"]["combined"] for r in rs])
        oc = np.array([r["out"]["combined"] for r in rs])
        od = np.array([r["out"]["dyck_acc"] for r in rs])
        ocp = np.array([r["out"]["copy_acc"] for r in rs])
        summ[name] = {"in_comb": ic.tolist(), "out_comb": oc.tolist(),
                      "out_dyck": od.tolist(), "out_copy": ocp.tolist()}
        print(f"{name:<12} {ic.mean():>10.1%}±{ic.std():.1%} {oc.mean():>11.1%}±{oc.std():.1%} "
              f"{od.mean():>10.1%} {ocp.mean():>10.1%}")

    print("\nContrastes pareados MIX vs puras (combinada, n=5):")
    for base in ("euclidiana", "esferica", "isotropica"):
        for met, key in [("IN", "in_comb"), ("OUT", "out_comb")]:
            a = np.array(summ["mix"][key]); b = np.array(summ[base][key])
            t, p = stats.ttest_rel(a, b)
            print(f"  MIX vs {base:<11} [{met}]: Δ={np.mean(a-b):+.2%}  t={t:+.2f}  p={p:.4f}")

    # gates por región (promedio de las 5 semillas)
    gd = np.mean([r["in"]["gates"]["dyck"] for r in res["mix"]], axis=0)
    gc = np.mean([r["in"]["gates"]["copy"] for r in res["mix"]], axis=0)
    print(f"\nGates de MIX DENTRO de la misma secuencia (promedio 5 semillas):")
    print(f"  en bloques Dyck (jerárq.): euc={gd[0]:.2f} esf={gd[1]:.2f} hip={gd[2]:.2f}")
    print(f"  en bloques Copy (planos):  euc={gc[0]:.2f} esf={gc[1]:.2f} hip={gc[2]:.2f}")

    with open("resultados_mixta2.json", "w") as f:
        json.dump({"arms": summ,
                   "gates_dyck": gd.tolist(), "gates_copy": gc.tolist()}, f, indent=1)
    print("\nGuardado: resultados_mixta2.json")
