# ============================================================================
# EFICIENCIA DIMENSIONAL: ¿alcanza el kernel hiperbólico la misma precisión
# que el euclidiano con MENOS dimensión (= menos parámetros, menos FLOPs)?
#
# Motivación teórica real (no especulativa): Nickel & Kiela (NeurIPS 2017)
# mostraron que Poincaré en d=2 iguala a Euclidiano en d~50 para reconstruir
# árboles, porque el volumen de una bola hiperbólica crece exponencialmente
# con el radio (vs polinomialmente en euclidiano) -- se necesitan MENOS
# coordenadas para separar exponencialmente muchos nodos jerárquicos.
# Nunca medimos esto para ATENCIÓN (solo comparamos a d_model IGUAL, donde
# el presupuesto de parámetros ya está emparejado). Aquí se barre d_model
# para ver si el kernel hiperbólico "hace más con menos".
#
# Config: Dyck-2, extrapolación 20-60 -> 80-120 (igual que el resto del
# paper), n_heads=4 fijo (así d_head = d_model/4 varía con d_model),
# 2 capas, 5 épocas, 5 semillas por punto.
#
# Ejecutar: ~/asreview/venv/bin/python exp_eficiencia_dimensional.py
# ============================================================================
import torch, torch.nn as nn, numpy as np, time, json
from scipy import stats
from exp_tres_geometrias import (Model, EuclideanAttention, Dyck2Dataset,
                                 collate_dyck, close_acc)
from fast_gaussian_test import IsotropicHyperbolicAttention
import torch.optim as optim

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def make_batchset(n, min_len, max_len, seed):
    """Materializa n secuencias Dyck-2 (misma generación que Dyck2Dataset) como lista."""
    ds = Dyck2Dataset(n, min_len, max_len, seed=seed)
    return list(iter(ds))

def flops_per_token_attn(d_model, n_heads, arm):
    """FLOPs aproximados del bloque de atención por par (i,j) de tokens,
    para comparar costo computacional a igual d_model (no solo params)."""
    d_head = d_model // n_heads
    proj = 4 * d_model * d_model            # Q,K,V,O proyecciones (por token)
    if arm == 'euclidean':
        score = n_heads * d_head            # producto punto por cabeza
    else:  # fisher_rao isotropic: matmul de mu (d_head) + arccosh (O(1))
        score = n_heads * (d_head + 5)      # + sigma y arccosh (costo ~cte extra)
    return proj, score  # proj es por-token; score es por-PAR de tokens

def train_arm(arm_cls, d_model, seed, epochs=5, bs=64, n_heads=4):
    torch.manual_seed(seed); np.random.seed(seed)
    train_seqs = make_batchset(2000, 20, 60, seed=42)
    tin = make_batchset(400, 20, 60, seed=123)
    tout = make_batchset(400, 80, 120, seed=456)
    model = Model(arm_cls, d_model=d_model, n_heads=n_heads)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, _lens in _batches(train_seqs, bs):
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, 5), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    ia = close_acc(model, list(_batches(tin, bs, shuffle=False)))
    oa = close_acc(model, list(_batches(tout, bs, shuffle=False)))
    return {"in_acc": ia, "out_acc": oa, "params": count_params(model),
           "secs": round(time.time() - t0)}

def _batches(seqs, bs, shuffle=True):
    idx = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        yield collate_dyck([seqs[j] for j in idx[i:i + bs]])

if __name__ == "__main__":
    torch.set_num_threads(8)
    D_MODELS = [16, 32, 64, 128, 256]
    N_HEADS = 4
    SEEDS = [1, 2, 3, 4, 5]
    ARMS = {'euclidean': EuclideanAttention, 'fisher_rao': IsotropicHyperbolicAttention}

    res = {d: {a: [] for a in ARMS} for d in D_MODELS}
    for d in D_MODELS:
        if d % N_HEADS != 0:
            continue
        for seed in SEEDS:
            for name, cls in ARMS.items():
                r = train_arm(cls, d, seed, n_heads=N_HEADS)
                res[d][name].append(r)
                print(f"d_model={d:>4} seed={seed} {name:<11}: "
                      f"IN={r['in_acc']:.1%} OUT={r['out_acc']:.1%} "
                      f"params={r['params']:>7} ({r['secs']}s)", flush=True)

    print("\n" + "=" * 88)
    print(f"{'d_model':>8} {'d_head':>7} {'arm':<11} {'OUT acc':>16} {'params':>9}")
    print("=" * 88)
    summary = {}
    for d in D_MODELS:
        if d % N_HEADS != 0: continue
        for name in ARMS:
            rs = res[d][name]
            oa = np.array([r["out_acc"] for r in rs])
            summary.setdefault(d, {})[name] = {
                "out_acc_mean": oa.mean(), "out_acc_std": oa.std(),
                "out_acc_seeds": oa.tolist(), "params": rs[0]["params"]}
            print(f"{d:>8} {d // N_HEADS:>7} {name:<11} {oa.mean():>10.4f}±{oa.std():.4f} {rs[0]['params']:>9}")

    print("\nContraste pareado por d_model (Fisher-Rao vs Euclidiano, n=5):")
    for d in D_MODELS:
        if d % N_HEADS != 0: continue
        fr = np.array(summary[d]['fisher_rao']['out_acc_seeds'])
        eu = np.array(summary[d]['euclidean']['out_acc_seeds'])
        t, p = stats.ttest_rel(fr, eu)
        print(f"  d_model={d:>4}: Δ={fr.mean()-eu.mean():+.4f}  t={t:+.2f}  p={p:.4f}")

    print("\n¿Con cuántas dimensiones MENOS alcanza Fisher-Rao lo que Euclidiano logra en d=256?")
    target = summary[256]['euclidean']['out_acc_mean']
    print(f"  Euclidiano en d=256: {target:.4f}")
    for d in D_MODELS:
        if d % N_HEADS != 0: continue
        fr_d = summary[d]['fisher_rao']['out_acc_mean']
        flag = " <-- iguala o supera con MENOS parámetros" if fr_d >= target and d < 256 else ""
        print(f"  Fisher-Rao en d={d:>4} (params={summary[d]['fisher_rao']['params']}): {fr_d:.4f}{flag}")

    with open("resultados_eficiencia_dimensional.json", "w") as f:
        json.dump({str(d): summary[d] for d in summary}, f, indent=1)
    print("\nGuardado: resultados_eficiencia_dimensional.json")
