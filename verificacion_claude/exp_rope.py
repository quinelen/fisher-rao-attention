# ============================================================================
# EXTRAPOLACIÓN CON POSICIONES RELATIVAS (RoPE) — pre-registrado en
# PREREGISTRO_ROPE.md (18-oct-2026, antes de ejecutar).
#
# Ningún brazo usa embedding posicional absoluto. Euclidiano: RoPE sobre q y k.
# Fisher-Rao isotrópico: la misma rotación sobre mu_q y mu_k, isometría exacta
# de H^{d+1} (||R_i mu_q - R_j mu_k|| depende solo de i - j) que conserva la
# descomposición en una sola multiplicación de matrices.
# PRIMARIO: Dyck-2 OUT, FR+RoPE vs Euc+RoPE, t pareado bilateral, n=20
# (semillas 6-25). Secundario: tarea de copia, mismos brazos y semillas.
# Ejecutar: python exp_rope.py
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, math
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader
from exp_tres_geometrias import (Model, EuclideanAttention, Dyck2Dataset,
                                 collate_dyck, close_acc)
from fast_gaussian_test import IsotropicHyperbolicAttention
from exp_mixta import CopyDataset, copy_acc


def rope(t):
    """Rotación posicional estándar sobre pares de coordenadas. t: [B,H,L,dh]."""
    B, H, L, dh = t.shape
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
    ang = torch.arange(L, dtype=torch.float32)[:, None] * inv[None]
    c, s = ang.cos(), ang.sin()
    t1, t2 = t[..., 0::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., 0::2] = t1 * c - t2 * s
    out[..., 1::2] = t1 * s + t2 * c
    return out


class EucRoPE(EuclideanAttention):
    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = rope(self.q(x).view(B, L, self.h, self.dh).transpose(1, 2))
        k = rope(self.k(x).view(B, L, self.h, self.dh).transpose(1, 2))
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        return self.o((F.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B, L, D))


class FRRoPE(IsotropicHyperbolicAttention):
    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.h, self.dh
        mq = rope(self.mu_q(x).view(B, L, H, dh).transpose(1, 2))
        mk = rope(self.mu_k(x).view(B, L, H, dh).transpose(1, 2))
        sq = (F.softplus(self.sg_q(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        sk = (F.softplus(self.sg_k(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        v = self.v(x).view(B, L, H, dh).transpose(1, 2)
        q2 = (mq * mq).sum(-1, keepdim=True)
        k2 = (mk * mk).sum(-1, keepdim=True).transpose(-2, -1)
        dmu2 = (q2 + k2 - 2.0 * mq @ mk.transpose(-2, -1)).clamp_min(0)
        skT = sk.transpose(-2, -1)
        arg = 1.0 + (0.5 * dmu2 + (sq - skT) ** 2) / (2.0 * sq * skT)
        d2 = 2.0 * torch.acosh(arg.clamp(min=1.0 + 1e-7)) ** 2
        s = -self.beta * d2 / dh
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        return self.o((F.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B, L, D))


class ModelNoPos(Model):
    """Mismo transformer sin embedding posicional absoluto."""
    def forward(self, x):
        B, L = x.shape
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        h = self.emb(x)
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.out(h)


def train_task(attn_cls, seed, task, epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    DS = Dyck2Dataset if task == "dyck" else CopyDataset
    metric = close_acc if task == "dyck" else copy_acc
    tr = DataLoader(DS(10000, 20, 60, seed=42), batch_size=bs, collate_fn=collate_dyck)
    tin = DataLoader(DS(2000, 20, 60, seed=123), batch_size=bs, collate_fn=collate_dyck)
    tout = DataLoader(DS(2000, 80, 120, seed=456), batch_size=bs, collate_fn=collate_dyck)
    model = ModelNoPos(attn_cls)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for seq, _ in tr:
            loss = crit(model(seq[:, :-1]).reshape(-1, 5), seq[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return {"in_acc": metric(model, tin), "out_acc": metric(model, tout),
            "params": sum(p.numel() for p in model.parameters()),
            "secs": round(time.time() - t0)}


def paired(a, b):
    a, b = np.array(a), np.array(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {"delta_pp": 100 * d.mean(), "ci95_pp": [100 * (d.mean() - half), 100 * (d.mean() + half)],
            "t": float(t), "p": float(p), "wins": int((d > 0).sum()), "n": len(d)}


if __name__ == "__main__":
    torch.set_num_threads(8)
    SEEDS = list(range(6, 26))
    out = {}
    for task in ["dyck", "copy"]:
        res = {"euc_rope": [], "fr_rope": []}
        for s in SEEDS:
            for name, cls in [("euc_rope", EucRoPE), ("fr_rope", FRRoPE)]:
                r = train_task(cls, s, task)
                res[name].append(r)
                print(f"{task} seed {s:>2} {name:<8}: IN={r['in_acc']:.2%} OUT={r['out_acc']:.2%} "
                      f"({r['secs']}s, {r['params']} params)", flush=True)
        fr = [r["out_acc"] for r in res["fr_rope"]]
        eu = [r["out_acc"] for r in res["euc_rope"]]
        cmp_ = paired(fr, eu)
        out[task] = {**res, "fr_minus_euc": cmp_,
                     "fr_mean_out": 100 * np.mean(fr), "euc_mean_out": 100 * np.mean(eu),
                     "fr_sd_out": 100 * np.std(fr), "euc_sd_out": 100 * np.std(eu)}
        print(f"\n{task.upper()}: Euc+RoPE {100*np.mean(eu):.2f}% ± {100*np.std(eu):.2f} | "
              f"FR+RoPE {100*np.mean(fr):.2f}% ± {100*np.std(fr):.2f} | FR-Euc {cmp_}\n", flush=True)
        if task == "dyck":
            ok = cmp_["delta_pp"] > 0 and cmp_["p"] < 0.05
            print("*** ENDPOINT PRIMARIO: " + ("CUMPLIDO — la ventaja de Fisher-Rao sobrevive con posiciones relativas"
                  if ok else "NO CUMPLIDO — ventaja no demostrada con posiciones relativas a esta escala") + "\n", flush=True)
            out["primary_met"] = bool(ok)
    with open("resultados_rope.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Guardado: resultados_rope.json")
