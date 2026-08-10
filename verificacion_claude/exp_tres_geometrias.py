# ============================================================================
# EXPERIMENTO: Tres geometrías Fisher-Rao para atención — Dyck-2 extrapolativo
#   1. Euclidiana estándar (producto punto, K = 0)
#   2. Esférica categórica  (Fisher-Rao exacta en el símplex, K = +1/4)
#   3. Hiperbólica gaussiana (Fisher-Rao de N(mu,sigma), K = -1/2, §5.2 del paper)
# Config idéntica a la v2: d_model=128, 2 capas, 4 cabezas, 10k muestras,
# batch 64, AdamW lr=5e-4, 5 épocas. 5 semillas. Test t pareado.
# Ejecutar: ~/asreview/venv/bin/python exp_tres_geometrias.py
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np, math, time, json

# ---------------- Datos Dyck-2 ----------------
class Dyck2Dataset(IterableDataset):
    def __init__(self, num_samples, min_len, max_len, seed=None):
        self.num_samples, self.min_len, self.max_len, self.seed = num_samples, min_len, max_len, seed
        self.token_map = {'(': 1, ')': 2, '[': 3, ']': 4}

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for _ in range(self.num_samples):
            L = np.random.randint(self.min_len // 2, self.max_len // 2 + 1) * 2
            yield torch.tensor(self._gen(L))

    def _gen(self, length):
        stack, seq, oc = [], [], 0
        for _ in range(length):
            if oc == 0 or (oc < length - len(seq) and np.random.rand() < 0.5):
                t = np.random.choice(['(', '['])
                seq.append(self.token_map[t]); stack.append(t); oc += 1
            else:
                t = stack.pop()
                seq.append(self.token_map[')' if t == '(' else ']']); oc -= 1
        return seq

def collate_dyck(batch):
    m = max(len(x) for x in batch)
    out = torch.zeros(len(batch), m, dtype=torch.long)
    for i, s in enumerate(batch):
        out[i, :len(s)] = s
    return out, torch.tensor([len(x) for x in batch])

# ---------------- Las tres atenciones ----------------
class EuclideanAttention(nn.Module):
    """K = 0: producto punto escalado estándar."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(s, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

class SphericalFisherRaoAttention(nn.Module):
    """K = +1/4: distancia geodésica exacta en el símplex categórico
    (esfera de radio 2 vía p -> 2*sqrt(p)); un solo matmul."""
    def __init__(self, d_model, n_heads, eps=1e-6):
        super().__init__()
        self.h, self.dh, self.eps = n_heads, d_model // n_heads, eps
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        qp = F.softmax(q, -1).clamp(min=self.eps)
        kp = F.softmax(k, -1).clamp(min=self.eps)
        bc = torch.matmul(qp.sqrt(), kp.sqrt().transpose(-2, -1)).clamp(max=1.0 - 1e-4)
        d = 2.0 * torch.acos(bc)
        s = -(d ** 2)
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(s, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

class HyperbolicGaussianAttention(nn.Module):
    """K = -1/2: cada cabeza proyecta el token a d_head gaussianas univariadas
    N(mu_r, sigma_r^2). La variedad Fisher-Rao de N(mu,sigma) es el semiplano
    de Poincaré con ds^2 = (dmu^2 + 2 dsigma^2)/sigma^2 (curvatura -1/2).
    Distancia geodésica exacta (Atkinson & Mitchell 1981; Costa et al. 2015):
        d_FR = sqrt(2) * arccosh( 1 + (0.5*(mu1-mu2)^2 + (s1-s2)^2) / (2 s1 s2) )
    d^2 total = media sobre las d_head coordenadas del producto de semiplanos
    (la media, en vez de la suma, normaliza la escala de los logits; beta aprendible).
    Coste: O(L^2 * d_head) elementwise (sin descomposición en matmul por el arccosh).
    """
    def __init__(self, d_model, n_heads, eps_sigma=0.05, beta_init=1.0):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.mu_q = nn.Linear(d_model, d_model); self.sg_q = nn.Linear(d_model, d_model)
        self.mu_k = nn.Linear(d_model, d_model); self.sg_k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.eps_sigma = eps_sigma
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def _heads(self, t, B, L):
        return t.view(B, L, self.h, self.dh).transpose(1, 2)  # [B,H,L,dh]

    def forward(self, x, mask=None):
        B, L, D = x.shape
        mq = self._heads(self.mu_q(x), B, L)
        sq = F.softplus(self._heads(self.sg_q(x), B, L)) + self.eps_sigma
        mk = self._heads(self.mu_k(x), B, L)
        sk = F.softplus(self._heads(self.sg_k(x), B, L)) + self.eps_sigma
        v = self._heads(self.v(x), B, L)
        # pareado por coordenada del producto de semiplanos: [B,H,L,L,dh]
        dmu2 = (mq.unsqueeze(3) - mk.unsqueeze(2)).pow(2)
        dsg2 = (sq.unsqueeze(3) - sk.unsqueeze(2)).pow(2)
        prod = sq.unsqueeze(3) * sk.unsqueeze(2)
        arg = 1.0 + (0.5 * dmu2 + dsg2) / (2.0 * prod)
        # d_FR^2 = 2 * arccosh(arg)^2 ; clamp: arccosh'(1) diverge (d^2 sí es suave ahí)
        d2 = 2.0 * torch.acosh(arg.clamp(min=1.0 + 1e-7)).pow(2)
        d2 = d2.mean(-1)                      # media sobre coordenadas
        s = -self.beta * d2
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(s, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

# ---------------- Transformer común ----------------
class Layer(nn.Module):
    def __init__(self, attn_cls, d_model, n_heads):
        super().__init__()
        self.attn = attn_cls(d_model, n_heads)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask=None):
        x = self.n1(x + self.attn(x, mask))
        return self.n2(x + self.ffn(x))

class Model(nn.Module):
    def __init__(self, attn_cls, vocab=5, d_model=128, n_layers=2, n_heads=4, max_len=120):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([Layer(attn_cls, d_model, n_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, L = x.shape
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        h = self.emb(x) + self.pos(torch.arange(L, device=x.device).unsqueeze(0))
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.out(h)

# ---------------- Métricas ----------------
def close_acc(model, loader):
    model.eval(); c = t = 0
    with torch.no_grad():
        for seq, _ in loader:
            logits = model(seq[:, :-1]); tgt = seq[:, 1:]
            m = (tgt == 2) | (tgt == 4)
            pref = torch.where(logits[..., 2] >= logits[..., 4], torch.tensor(2), torch.tensor(4))
            c += (pref[m] == tgt[m]).sum().item(); t += m.sum().item()
    return c / max(t, 1)

def eval_loss(model, loader, crit, vocab=5):
    model.eval(); tot = n = 0
    with torch.no_grad():
        for seq, _ in loader:
            logits = model(seq[:, :-1])
            tot += crit(logits.reshape(-1, vocab), seq[:, 1:].reshape(-1)).item() * len(seq)
            n += len(seq)
    return tot / n

# ---------------- Loop experimental ----------------
def run_one(attn_cls, seed, epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(Dyck2Dataset(10000, 20, 60, seed=42), batch_size=bs, collate_fn=collate_dyck)
    tin = DataLoader(Dyck2Dataset(2000, 20, 60, seed=123), batch_size=bs, collate_fn=collate_dyck)
    tout = DataLoader(Dyck2Dataset(2000, 80, 120, seed=456), batch_size=bs, collate_fn=collate_dyck)
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
    return {"in_loss": eval_loss(model, tin, crit), "out_loss": eval_loss(model, tout, crit),
            "in_acc": close_acc(model, tin), "out_acc": close_acc(model, tout),
            "params": sum(p.numel() for p in model.parameters()),
            "secs": round(time.time() - t0)}

def paired_t(a, b):
    d = np.array(a) - np.array(b)
    sd = d.std(ddof=1)
    return d.mean(), sd, (d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else float('nan'))

if __name__ == "__main__":
    torch.set_num_threads(8)
    GEOMS = {
        "euclidiana (K=0)":        EuclideanAttention,
        "esferica (K=+1/4)":       SphericalFisherRaoAttention,
        "hiperbolica (K=-1/2)":    HyperbolicGaussianAttention,
    }
    SEEDS = [1, 2, 3, 4, 5]
    res = {g: [] for g in GEOMS}
    for seed in SEEDS:
        for name, cls in GEOMS.items():
            r = run_one(cls, seed)
            res[name].append(r)
            print(f"seed {seed} | {name:<22}: IN acc={r['in_acc']:.1%}  OUT acc={r['out_acc']:.1%}  "
                  f"OUT loss={r['out_loss']:.4f}  ({r['secs']}s, {r['params']} params)", flush=True)

    print("\n" + "=" * 84)
    print(f"{'geometría':<24} {'IN acc':>14} {'OUT acc (extrap.)':>20} {'OUT loss':>16}")
    print("=" * 84)
    summ = {}
    for name, rs in res.items():
        oa = np.array([r["out_acc"] for r in rs]); ia = np.array([r["in_acc"] for r in rs])
        ol = np.array([r["out_loss"] for r in rs])
        summ[name] = {"out_acc": oa.tolist(), "in_acc": ia.tolist(), "out_loss": ol.tolist(),
                      "params": rs[0]["params"]}
        print(f"{name:<24} {ia.mean():>7.1%}±{ia.std():.1%} {oa.mean():>12.1%}±{oa.std():.1%} "
              f"{ol.mean():>10.4f}±{ol.std():.4f}")

    print("\nContrastes pareados de OUT acc (n=5, t crítico bilateral 5% = 2.776):")
    names = list(GEOMS)
    for a, b in [(names[1], names[0]), (names[2], names[0]), (names[2], names[1])]:
        m, sdev, t = paired_t(summ[a]["out_acc"], summ[b]["out_acc"])
        print(f"  {a}  vs  {b}:  Δ = {m:+.2%} ± {sdev:.2%}   t = {t:+.2f}")

    with open("resultados_tres_geometrias.json", "w") as f:
        json.dump(summ, f, indent=1)
    print("\nGuardado: resultados_tres_geometrias.json")
