# ============================================================================
# RESPUESTA A LA CRÍTICA DE ESCALA: (A) 20 semillas, (B) modelo grande
#
# (A) VEINTE SEMILLAS para el contraste principal (euclidiana vs hiperbólica
#     isotrópica, config §4.2: d128, 2 capas, 10k muestras, 5 épocas).
#     Corre semillas 6-20 y las agrega a las 5 existentes (JSONs previos).
#
# (B) SONDA DE ESCALA: 6 capas, d_model=512, n_heads=8 (d_head=64, sobre el
#     umbral de §4.7), 10k muestras, 5 épocas, 3 semillas por brazo.
#     Es la escala que pide la crítica (≥6 capas, d_model≥512), CPU-viable.
#
# Ejecutar: ~/asreview/venv/bin/python exp_respuesta_escala.py
# ============================================================================
import torch, torch.nn as nn, numpy as np, time, json
import torch.optim as optim
from scipy import stats
from exp_tres_geometrias import (Model, EuclideanAttention, Dyck2Dataset,
                                 collate_dyck, close_acc, run_one)
from fast_gaussian_test import IsotropicHyperbolicAttention

def make_batchset(n, a, b, seed):
    return list(iter(Dyck2Dataset(n, a, b, seed=seed)))

def _batches(seqs, bs, shuffle=True):
    idx = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        yield collate_dyck([seqs[j] for j in idx[i:i + bs]])

def train_big(attn_cls, seed, d_model=512, n_layers=6, n_heads=8,
              epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = make_batchset(10000, 20, 60, seed=42)
    tin = make_batchset(2000, 20, 60, seed=123)
    tout = make_batchset(2000, 80, 120, seed=456)
    model = Model(attn_cls, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, _l in _batches(tr, bs):
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, 5), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    ia = close_acc(model, list(_batches(tin, bs, shuffle=False)))
    oa = close_acc(model, list(_batches(tout, bs, shuffle=False)))
    return {"in_acc": ia, "out_acc": oa,
           "params": sum(p.numel() for p in model.parameters()),
           "secs": round(time.time() - t0)}

if __name__ == "__main__":
    torch.set_num_threads(8)

    # ---------- (A) 20 semillas del contraste principal ----------
    print("=== (A) EXTENSIÓN A 20 SEMILLAS (config §4.2: d128, 2L, 10k, 5ep) ===")
    # semillas 1-5 existentes, de los JSONs originales
    prev = json.load(open("resultados_tres_geometrias.json"))
    prev_fast = json.load(open("resultados_fast_gaussian.json"))
    euc = list(prev["euclidiana (K=0)"]["out_acc"])
    iso = [r["out_acc"] for r in prev_fast["isotropica"]]
    print(f"  semillas 1-5 recuperadas de JSONs: euc={len(euc)}, iso={len(iso)}")
    for seed in range(6, 21):
        for name, cls, acc_list in [("euclidean", EuclideanAttention, euc),
                                     ("fisher_rao", IsotropicHyperbolicAttention, iso)]:
            r = run_one(cls, seed)
            acc_list.append(r["out_acc"])
            print(f"  seed {seed:>2} {name:<11}: OUT={r['out_acc']:.4f} ({r['secs']}s)", flush=True)
    euc_a, iso_a = np.array(euc), np.array(iso)
    t, p = stats.ttest_rel(iso_a, euc_a)
    print(f"\n  n=20:  euclidiana={euc_a.mean():.4f}±{euc_a.std():.4f}  "
          f"fisher-rao={iso_a.mean():.4f}±{iso_a.std():.4f}")
    print(f"  contraste pareado n=20: Δ={iso_a.mean()-euc_a.mean():+.4f}  t={t:+.2f}  p={p:.2e}")
    wins = int((iso_a > euc_a).sum())
    print(f"  semillas ganadas por fisher-rao: {wins}/20")
    with open("resultados_20semillas.json", "w") as f:
        json.dump({"euclidean": euc_a.tolist(), "fisher_rao": iso_a.tolist(),
                   "t": float(t), "p": float(p)}, f, indent=1)

    # ---------- (B) sonda de escala ----------
    print("\n=== (B) SONDA DE ESCALA (d_model=512, 6 capas, 8 cabezas, 10k, 5ep, 3 semillas) ===")
    res = {"euclidean": [], "fisher_rao": []}
    for seed in [1, 2, 3]:
        for name, cls in [("euclidean", EuclideanAttention),
                          ("fisher_rao", IsotropicHyperbolicAttention)]:
            r = train_big(cls, seed)
            res[name].append(r)
            print(f"  seed {seed} {name:<11}: IN={r['in_acc']:.4f} OUT={r['out_acc']:.4f} "
                  f"params={r['params']} ({r['secs']}s)", flush=True)
    e = np.array([r["out_acc"] for r in res["euclidean"]])
    f_ = np.array([r["out_acc"] for r in res["fisher_rao"]])
    t2, p2 = stats.ttest_rel(f_, e)
    print(f"\n  escala 6L/512: euclidiana={e.mean():.4f}±{e.std():.4f}  "
          f"fisher-rao={f_.mean():.4f}±{f_.std():.4f}  Δ={f_.mean()-e.mean():+.4f}  t={t2:+.2f}  p={p2:.4f}")
    with open("resultados_escala_6L512.json", "w") as f2:
        json.dump({k: [r["out_acc"] for r in v] for k, v in res.items()}, f2, indent=1)
    print("\nGuardado: resultados_20semillas.json, resultados_escala_6L512.json")
