# Experimento comparativo: Ricci geodesic attention vs atención estándar en Dyck-2
# - Verifica el crash del DataLoader original (longitudes variables sin collate)
# - Verifica la fuga de información sin máscara causal
# - Compara extrapolación de longitud (train 20-60, test 80-120)
# - Registra la evolución de self.g

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json, sys

torch.manual_seed(42)
DEV = 'cpu'

# ---------- Datos Dyck-2 (generador del proyecto) ----------
TOKEN = {'(': 1, ')': 2, '[': 3, ']': 4}
def gen_seq(length, rng):
    stack, seq, open_count = [], [], 0
    for i in range(length):
        if open_count == 0 or (open_count < length - i and rng.random() < 0.5):
            t = rng.choice(['(', '['])
            seq.append(TOKEN[t]); stack.append(t); open_count += 1
        else:
            t = stack.pop()
            seq.append(TOKEN[')' if t == '(' else ']']); open_count -= 1
    return seq

def make_batchset(n, min_len, max_len, seed):
    rng = np.random.RandomState(seed)
    seqs = []
    for _ in range(n):
        L = rng.randint(min_len // 2, max_len // 2 + 1) * 2
        seqs.append(torch.tensor(gen_seq(L, rng), dtype=torch.long))
    return seqs

def collate_pad(seqs):
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out

def batches(seqs, bs, shuffle=True):
    idx = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        yield collate_pad([seqs[j] for j in idx[i:i+bs]])

def test_dataloader_crash():
    print("TEST 0: DataLoader original (sin collate de padding, longitudes variables)")
    try:
        from torch.utils.data import DataLoader, IterableDataset
        class Raw(IterableDataset):
            def __iter__(self):
                rng = np.random.RandomState(0)
                for _ in range(10):
                    L = rng.randint(10, 31) * 2
                    yield torch.tensor(gen_seq(L, rng)), L
        next(iter(DataLoader(Raw(), batch_size=4)))
        print("  -> no crasheo (inesperado)")
    except RuntimeError as e:
        print(f"  -> RuntimeError confirmado: {str(e)[:110]}")
    print()

# ---------- Modelos ----------
class RicciMHA(nn.Module):
    """Copia fiel del proyecto (con soporte de mask)."""
    def __init__(self, d_model, n_heads, beta=1.0, kappa=0.5, dt=0.01):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.beta, self.kappa, self.dt = beta, kappa, dt
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer('g', torch.eye(self.d_head).repeat(n_heads, 1, 1))
        self.g_log = []
    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
        q_p, k_p = F.softmax(q, -1), F.softmax(k, -1)
        g_q, g_k = 1.0 / (q_p + 1e-12), 1.0 / (k_p + 1e-12)
        g_mid = 0.5 * (g_q.unsqueeze(3) + g_k.unsqueeze(2))
        diff = q_p.unsqueeze(3) - k_p.unsqueeze(2)
        dist_sq = (diff * diff * g_mid).sum(-1)
        scores = -self.beta * dist_sq
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, L, D)
        with torch.no_grad():
            avg_g = g_q.mean(dim=(0, 2)); avg_p = q_p.mean(dim=(0, 2))
            S = -(avg_p * torch.log(avg_p + 1e-12)).sum(-1, keepdim=True)
            I_diag = -S * avg_g - 1.0 / (avg_p + 1e-12)
            dg = -2.0 * (-0.5 * avg_g) + self.kappa * I_diag
            self.g = torch.clamp(avg_g + self.dt * dg, 0.1, 10.0)  # shapes coherentes [H, dh]
        return self.out_proj(out)

class VanillaMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class Block(nn.Module):
    def __init__(self, attn, d_model):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = attn
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class TF(nn.Module):
    def __init__(self, attn_cls, vocab=5, d_model=64, n_layers=2, n_heads=4, max_len=130, causal=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([Block(attn_cls(d_model, n_heads), d_model) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab)
        self.causal = causal
    def forward(self, x):
        B, L = x.shape
        h = self.embed(x) + self.pos(torch.arange(L, device=x.device)).unsqueeze(0)
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L) if self.causal else None
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.out(h)

# ---------- Entrenamiento y evaluación ----------
def close_accuracy(model, seqs, bs=32):
    """Precisión del TIPO de cierre: en posiciones donde el target es ) o ],
    ¿el modelo prefiere el cierre correcto entre {), ]}?  (mide seguimiento del stack)"""
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for xb in batches(seqs, bs, shuffle=False):
            logits = model(xb[:, :-1]); tgt = xb[:, 1:]
            m = (tgt == 2) | (tgt == 4)
            pref = torch.where(logits[..., 2] >= logits[..., 4],
                               torch.tensor(2), torch.tensor(4))
            correct += (pref[m] == tgt[m]).sum().item(); tot += m.sum().item()
    return correct / max(tot, 1)

def eval_loss(model, seqs, bs=32):
    model.eval(); tot = n = 0
    crit = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for xb in batches(seqs, bs, shuffle=False):
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, 5), xb[:, 1:].reshape(-1))
            tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / n

def train_model(name, model, train_seqs, tin, tout, epochs=4, bs=32, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    print(f"--- {name} ({sum(p.numel() for p in model.parameters())} params) ---")
    for ep in range(epochs):
        model.train(); tl = nb = 0
        t0 = time.time()
        for xb in batches(train_seqs, bs):
            logits = model(xb[:, :-1])
            loss = crit(logits.reshape(-1, 5), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        lin, lout = eval_loss(model, tin), eval_loss(model, tout)
        ain, aout = close_accuracy(model, tin), close_accuracy(model, tout)
        gmsg = ""
        a0 = model.layers[0].attn
        if hasattr(a0, 'g') and a0.g.dim() == 2:
            gv = a0.g
            gmsg = f" | g: min={gv.min():.2f} mean={gv.mean():.2f} max={gv.max():.2f} %clamp0.1={(gv<=0.1001).float().mean():.0%}"
        print(f"  ep{ep+1}: train={tl/nb:.4f} | IN loss={lin:.4f} acc_cierre={ain:.1%} | "
              f"OUT loss={lout:.4f} acc_cierre={aout:.1%} | {time.time()-t0:.0f}s{gmsg}")
    return {"in_loss": lin, "out_loss": lout, "in_acc": ain, "out_acc": aout}

if __name__ == "__main__":
    test_dataloader_crash()
    train_seqs = make_batchset(2000, 20, 60, seed=42)
    tin = make_batchset(400, 20, 60, seed=123)
    tout = make_batchset(400, 80, 120, seed=456)

    print("TEST 1: FUGA — Ricci SIN máscara causal (como el código original) — 2 epochs")
    torch.manual_seed(1)
    m_leak = TF(RicciMHA, causal=False)
    train_model("Ricci sin máscara (código original)", m_leak, train_seqs, tin, tout, epochs=2)
    print()

    print("TEST 2: comparación justa CON máscara causal — 4 epochs")
    torch.manual_seed(1)
    res_v = train_model("Transformer estándar (causal)", TF(VanillaMHA), train_seqs, tin, tout)
    torch.manual_seed(1)
    res_r = train_model("Ricci-Transformer (causal)", TF(RicciMHA), train_seqs, tin, tout)

    print()
    print("RESUMEN:", json.dumps({"vanilla": res_v, "ricci": res_r}, indent=1))
