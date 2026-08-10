# Comparación multi-semilla (5 seeds x 8 epochs) de 4 mecanismos de atención en Dyck-2
# con extrapolación de longitud (train 20-60 -> test 80-120).
# Modelos: estándar, Ricci (proyecto), Fisher-Rao exacta, Hiperbólica (Poincaré).

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json
from train_compare import (make_batchset, batches, Block, TF,
                           VanillaMHA, RicciMHA, eval_loss, close_accuracy)
from train_fisher_rao import FisherRaoMHA

class PoincareMHA(nn.Module):
    """Atención por distancia geodésica en la bola de Poincaré (curvatura -1).
    q,k se mapean a la bola vía el mapa exponencial en el origen:
        expmap0(v) = tanh(||v||) * v / ||v||
    Distancia:  d(x,y) = arccosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
    Es la geometría hiperbólica que la teoría del proyecto postula para jerarquías.
    """
    def __init__(self, d_model, n_heads, beta=1.0):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.beta = nn.Parameter(torch.tensor(beta))
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def expmap0(v, max_norm=0.999):
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        x = torch.tanh(n) * v / n
        # mantener estrictamente dentro de la bola
        xn = x.norm(dim=-1, keepdim=True)
        scale = torch.where(xn > max_norm, max_norm / xn, torch.ones_like(xn))
        return x * scale

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
        qb, kb = self.expmap0(q), self.expmap0(k)
        # ||q-k||^2 pareado vía descomposición en matmul (O(L^2) memoria)
        q2 = (qb * qb).sum(-1, keepdim=True)                    # [B,H,L,1]
        k2 = (kb * kb).sum(-1, keepdim=True).transpose(-2, -1)  # [B,H,1,L]
        sq_dist = (q2 + k2 - 2.0 * qb @ kb.transpose(-2, -1)).clamp_min(0)
        denom = (1.0 - q2) * (1.0 - k2)                          # [B,H,L,L] por broadcast
        arg = (1.0 + 2.0 * sq_dist / denom.clamp_min(1e-9)).clamp_min(1.0 + 1e-7)
        d_hyp = torch.acosh(arg)
        scores = -self.beta * d_hyp.pow(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


def train_quiet(model, train_seqs, tin, tout, epochs=8, bs=32, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    for ep in range(epochs):
        model.train()
        for xb in batches(train_seqs, bs):
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, 5), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return {"in_loss": eval_loss(model, tin), "out_loss": eval_loss(model, tout),
            "in_acc": close_accuracy(model, tin), "out_acc": close_accuracy(model, tout)}


if __name__ == "__main__":
    train_seqs = make_batchset(2000, 20, 60, seed=42)
    tin = make_batchset(400, 20, 60, seed=123)
    tout = make_batchset(400, 80, 120, seed=456)

    MODELS = {
        "estandar":   lambda: TF(VanillaMHA),
        "ricci":      lambda: TF(RicciMHA),
        "fisher_rao": lambda: TF(FisherRaoMHA),
        "poincare":   lambda: TF(PoincareMHA),
    }
    SEEDS = [1, 2, 3, 4, 5]
    results = {name: [] for name in MODELS}

    for seed in SEEDS:
        for name, build in MODELS.items():
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            r = train_quiet(build(), train_seqs, tin, tout, epochs=8)
            results[name].append(r)
            print(f"seed {seed} | {name:<11}: IN acc={r['in_acc']:.1%} OUT acc={r['out_acc']:.1%} "
                  f"OUT loss={r['out_loss']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print(f"{'modelo':<12} {'IN acc':>16} {'OUT acc (extrap.)':>20} {'OUT loss':>16}")
    print("=" * 78)
    summary = {}
    for name, rs in results.items():
        ia = np.array([r["in_acc"] for r in rs]); oa = np.array([r["out_acc"] for r in rs])
        ol = np.array([r["out_loss"] for r in rs])
        summary[name] = {"in_acc": [ia.mean(), ia.std()], "out_acc": [oa.mean(), oa.std()],
                         "out_loss": [ol.mean(), ol.std()], "out_acc_seeds": oa.tolist()}
        print(f"{name:<12} {ia.mean():>8.1%}±{ia.std():.1%} {oa.mean():>12.1%}±{oa.std():.1%} "
              f"{ol.mean():>10.4f}±{ol.std():.4f}")

    # diferencias pareadas vs estándar (mismas semillas)
    print("\nDiferencia pareada de OUT acc vs estándar (por semilla):")
    base = np.array(summary["estandar"]["out_acc_seeds"])
    for name in ("ricci", "fisher_rao", "poincare"):
        d = np.array(summary[name]["out_acc_seeds"]) - base
        # t pareado simple
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std(ddof=1) > 0 else float('nan')
        print(f"  {name:<11}: Δ = {d.mean():+.2%} ± {d.std():.2%}  (t pareado = {t:+.2f}, n=5)")

    with open("multi_seed_results.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\nGuardado: multi_seed_results.json")
