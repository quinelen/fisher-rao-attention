# ============================================================================
# EXPERIMENTO: Curvatura mixta adaptativa segun la estructura del dato
#
# MixedCurvatureAttention: cada cabeza calcula scores bajo TRES geometrias
# (euclidiana K=0, esferica Fisher-Rao K=+1/4, hiperbolica isotropica K<0)
# y un gate causal dependiente del input las mezcla por posicion:
#     s = a_e*s_euc + a_s*s_esf + a_h*s_hip,  a = softmax(W_g . cummean(x))
# El gate no mira el futuro (media acumulada causal).
#
# Tareas de estructura opuesta:
#   - Dyck-2 : jerarquica (prediccion del cierre correcto; extrapolacion 80-120)
#   - Copy   : plana (primera mitad tokens aleatorios, segunda mitad = copia)
# Prediccion de la hipotesis: gates -> hiperbolica en Dyck, -> euclidiana en Copy.
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np, math, time, json
from exp_tres_geometrias import (Dyck2Dataset, collate_dyck, Model,
                                 EuclideanAttention, close_acc, eval_loss)
from fast_gaussian_test import IsotropicHyperbolicAttention

GEOM_NAMES = ["euclid", "esfer", "hiperb"]

class MixedCurvatureAttention(nn.Module):
    def __init__(self, d_model, n_heads, eps_sigma=0.05):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.sg_q = nn.Linear(d_model, n_heads); self.sg_k = nn.Linear(d_model, n_heads)
        self.gate = nn.Linear(d_model, n_heads * 3)
        self.beta_s = nn.Parameter(torch.ones(n_heads))   # temperatura esferica por cabeza
        self.beta_h = nn.Parameter(torch.ones(n_heads))   # temperatura hiperbolica por cabeza
        self.eps_sigma = eps_sigma
        self.last_gate = None   # [H,3] promedio del ultimo batch (diagnostico)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.h, self.dh
        q = self.q(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v(x).view(B, L, H, dh).transpose(1, 2)

        # --- geometria euclidiana (K=0) ---
        dot = q @ k.transpose(-2, -1)                       # [B,H,L,L] (se reusa abajo)
        s_euc = dot / math.sqrt(dh)

        # --- geometria esferica (K=+1/4): Fisher-Rao categorica exacta ---
        qp = F.softmax(q, -1).clamp(min=1e-6)
        kp = F.softmax(k, -1).clamp(min=1e-6)
        bc = torch.matmul(qp.sqrt(), kp.sqrt().transpose(-2, -1)).clamp(max=1.0 - 1e-4)
        s_sph = -self.beta_s.view(1, H, 1, 1) * (2.0 * torch.acos(bc)).pow(2)

        # --- geometria hiperbolica isotropica (K<0): H^{dh+1} exacta ---
        sq = (F.softplus(self.sg_q(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)  # [B,H,L,1]
        sk = (F.softplus(self.sg_k(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        q2 = (q * q).sum(-1, keepdim=True)
        k2 = (k * k).sum(-1, keepdim=True).transpose(-2, -1)
        dmu2 = (q2 + k2 - 2.0 * dot).clamp_min(0)
        dsg2 = (sq - sk.transpose(-2, -1)).pow(2)
        arg = 1.0 + (0.5 * dmu2 + dsg2) / (2.0 * sq * sk.transpose(-2, -1))
        s_hyp = -self.beta_h.view(1, H, 1, 1) * 2.0 * torch.acosh(arg.clamp(min=1.0 + 1e-7)).pow(2) / dh

        # --- gate causal dependiente del dato: media acumulada (sin ver el futuro) ---
        cm = x.cumsum(1) / torch.arange(1, L + 1, device=x.device).view(1, L, 1)
        g = F.softmax(self.gate(cm).view(B, L, H, 3), dim=-1).permute(0, 2, 1, 3)  # [B,H,L,3]
        self.last_gate = g.mean(dim=(0, 2)).detach()
        self.last_gate_full = g.detach()   # [B,H,L,3] para análisis por región

        s = (g[..., 0:1] * s_euc + g[..., 1:2] * s_sph + g[..., 2:3] * s_hyp)
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(s, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

# ---------------- Tarea plana: copia ----------------
class CopyDataset(IterableDataset):
    """Primera mitad: tokens aleatorios de {1..4}; segunda mitad: copia exacta.
    Estructura plana (lookup posicional), sin jerarquia."""
    def __init__(self, num_samples, min_len, max_len, seed=None):
        self.n, self.a, self.b, self.seed = num_samples, min_len, max_len, seed

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for _ in range(self.n):
            half = np.random.randint(self.a // 2, self.b // 2 + 1)
            first = np.random.randint(1, 5, size=half)
            yield torch.tensor(np.concatenate([first, first]))

def copy_acc(model, loader):
    """Precision en la segunda mitad (posiciones copiables)."""
    model.eval(); c = t = 0
    with torch.no_grad():
        for seq, lens in loader:
            logits = model(seq[:, :-1]); tgt = seq[:, 1:]
            pred = logits.argmax(-1)
            for i, Ln in enumerate(lens.tolist()):
                h = Ln // 2
                # targets en posiciones h-1 .. Ln-2 del tensor desplazado
                c += (pred[i, h - 1:Ln - 1] == tgt[i, h - 1:Ln - 1]).sum().item()
                t += Ln - h
    return c / max(t, 1)

# ---------------- Entrenamiento generico ----------------
def train_task(attn_cls, seed, task, epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    DS = Dyck2Dataset if task == "dyck" else CopyDataset
    metric = close_acc if task == "dyck" else copy_acc
    tr = DataLoader(DS(10000, 20, 60, seed=42), batch_size=bs, collate_fn=collate_dyck)
    tin = DataLoader(DS(2000, 20, 60, seed=123), batch_size=bs, collate_fn=collate_dyck)
    tout = DataLoader(DS(2000, 80, 120, seed=456), batch_size=bs, collate_fn=collate_dyck)
    model = Model(attn_cls)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for seq, _ in tr:
            logits = model(seq[:, :-1])
            loss = crit(logits.reshape(-1, 5), seq[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    r = {"in_acc": metric(model, tin), "out_acc": metric(model, tout),
         "secs": round(time.time() - t0),
         "params": sum(p.numel() for p in model.parameters())}
    # gates promedio sobre el set de evaluacion IN (por capa y cabeza)
    if attn_cls is MixedCurvatureAttention:
        model.eval(); acc = []
        with torch.no_grad():
            for seq, _ in tin:
                model(seq[:, :-1])
                acc.append(torch.stack([lyr.attn.last_gate for lyr in model.layers]))  # [nl,H,3]
        r["gates"] = torch.stack(acc).mean(0).tolist()   # [n_layers, H, 3]
    return r

def fmt_gates(gates):
    out = []
    for li, layer in enumerate(gates):
        heads = "  ".join("[" + " ".join(f"{v:.2f}" for v in h) + "]" for h in layer)
        g = np.array(layer).mean(0)
        out.append(f"    capa {li}: {heads}   promedio (euc/esf/hip) = "
                   f"{g[0]:.2f}/{g[1]:.2f}/{g[2]:.2f}")
    return "\n".join(out)

if __name__ == "__main__":
    torch.set_num_threads(8)
    results = {"dyck": {}, "copy": {}}

    print("=== TAREA JERARQUICA (Dyck-2) — MIX adaptativa, 5 semillas ===")
    rs = []
    for seed in [1, 2, 3, 4, 5]:
        r = train_task(MixedCurvatureAttention, seed, "dyck")
        rs.append(r)
        g = np.array(r["gates"]).mean(axis=(0, 1))
        print(f"seed {seed} | MIX dyck: IN={r['in_acc']:.1%} OUT={r['out_acc']:.1%} ({r['secs']}s) "
              f"gates euc/esf/hip = {g[0]:.2f}/{g[1]:.2f}/{g[2]:.2f}", flush=True)
    results["dyck"]["mix"] = rs
    print("Gates detallados (ultima semilla):")
    print(fmt_gates(rs[-1]["gates"]))

    print("\n=== TAREA PLANA (Copy) — MIX + baselines, 3 semillas ===")
    for name, cls in [("mix", MixedCurvatureAttention),
                      ("euclid", EuclideanAttention),
                      ("isotropica", IsotropicHyperbolicAttention)]:
        rs = []
        for seed in [1, 2, 3]:
            r = train_task(cls, seed, "copy")
            rs.append(r)
            extra = ""
            if "gates" in r:
                g = np.array(r["gates"]).mean(axis=(0, 1))
                extra = f" gates euc/esf/hip = {g[0]:.2f}/{g[1]:.2f}/{g[2]:.2f}"
            print(f"seed {seed} | {name:<11} copy: IN={r['in_acc']:.1%} OUT={r['out_acc']:.1%} "
                  f"({r['secs']}s){extra}", flush=True)
        results["copy"][name] = rs
    if "gates" in results["copy"]["mix"][-1]:
        print("Gates detallados en Copy (ultima semilla):")
        print(fmt_gates(results["copy"]["mix"][-1]["gates"]))

    print("\n=== RESUMEN ===")
    oa = np.array([r["out_acc"] for r in results["dyck"]["mix"]])
    print(f"MIX en Dyck-2 (5 seeds): OUT acc = {oa.mean():.1%} ± {oa.std():.1%}")
    print("  (referencias Dyck-2: euclidiana 85.8%±0.7, esferica 86.6%±0.6, "
          "isotropica 88.4%±0.8)")
    for name in ("mix", "euclid", "isotropica"):
        oa = np.array([r["out_acc"] for r in results["copy"][name]])
        print(f"{name:<11} en Copy (3 seeds): OUT acc = {oa.mean():.1%} ± {oa.std():.1%}")
    gd = np.array([np.array(r["gates"]).mean(axis=(0, 1)) for r in results["dyck"]["mix"]]).mean(0)
    gc = np.array([np.array(r["gates"]).mean(axis=(0, 1)) for r in results["copy"]["mix"]]).mean(0)
    print(f"\nGates promedio MIX  — Dyck (jerarquica): euc={gd[0]:.2f} esf={gd[1]:.2f} hip={gd[2]:.2f}")
    print(f"Gates promedio MIX  — Copy (plana):      euc={gc[0]:.2f} esf={gc[1]:.2f} hip={gc[2]:.2f}")

    with open("resultados_mixta.json", "w") as fjs:
        json.dump(results, fjs, indent=1)
    print("\nGuardado: resultados_mixta.json")
