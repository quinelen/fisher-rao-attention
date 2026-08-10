# Verificación de la v2 del usuario (FisherRaoExactAttention) + variante con beta aprendible.
# Config fiel a su script: d_model=128, 2 capas, 4 cabezas, 10k muestras, batch 64, 5 epochs.
# Se agrega: eval IN-distribution y precisión de tipo de cierre (métrica discriminativa).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np, time, json

# ---------- Código del usuario (verbatim en lo esencial) ----------
class Dyck2Dataset(IterableDataset):
    def __init__(self, num_samples, min_len, max_len, seed=None):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.token_map = {'(': 1, ')': 2, '[': 3, ']': 4}

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for _ in range(self.num_samples):
            L = np.random.randint(self.min_len // 2, self.max_len // 2 + 1) * 2
            yield torch.tensor(self._generate_sequence(L))

    def _generate_sequence(self, length):
        stack, seq, open_count = [], [], 0
        for _ in range(length):
            if open_count == 0 or (open_count < length - len(seq) and np.random.rand() < 0.5):
                typ = np.random.choice(['(', '['])
                seq.append(self.token_map[typ]); stack.append(typ); open_count += 1
            else:
                typ = stack.pop()
                seq.append(self.token_map[')' if typ == '(' else ']']); open_count -= 1
        return seq

def collate_dyck(batch):
    lens = [len(x) for x in batch]
    max_len = max(lens)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, :len(s)] = s
    return padded, torch.tensor(lens)

class FisherRaoExactAttention(nn.Module):
    def __init__(self, d_model, n_heads, eps=1e-6, learnable_beta=False, beta_init=8.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta_init))
        else:
            self.beta = 1.0  # versión del usuario: scores = -d^2 sin temperatura

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        q_prob = F.softmax(q, dim=-1).clamp(min=self.eps)
        k_prob = F.softmax(k, dim=-1).clamp(min=self.eps)
        inner = torch.matmul(q_prob.sqrt(), k_prob.sqrt().transpose(-2, -1)).clamp(max=1.0 - self.eps)
        dist = 2.0 * torch.acos(inner)
        scores = -self.beta * (dist ** 2) if isinstance(self.beta, nn.Parameter) else -(dist ** 2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, dim=-1)
        ctx = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(ctx), w

class Layer(nn.Module):
    def __init__(self, d_model, n_heads, **kw):
        super().__init__()
        self.attn = FisherRaoExactAttention(d_model, n_heads, **kw)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        a, _ = self.attn(x, mask=mask)
        x = self.norm1(x + a)
        return self.norm2(x + self.ffn(x))

class Model(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, max_len, **kw):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([Layer(d_model, n_heads, **kw) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, L = x.shape
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        h = self.embed(x) + self.pos(torch.arange(L, device=x.device).unsqueeze(0))
        for lyr in self.layers:
            h = lyr(h, mask=mask)
        return self.out(h)

# ---------- Métricas ----------
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

def run(name, **kw):
    torch.manual_seed(1); np.random.seed(1)
    VOCAB, D, NL, NH, MAXLEN, BS, EPOCHS = 5, 128, 2, 4, 120, 64, 5
    tr = DataLoader(Dyck2Dataset(10000, 20, 60, seed=42), batch_size=BS, collate_fn=collate_dyck)
    tin = DataLoader(Dyck2Dataset(2000, 20, 60, seed=123), batch_size=BS, collate_fn=collate_dyck)
    tout = DataLoader(Dyck2Dataset(2000, 80, 120, seed=456), batch_size=BS, collate_fn=collate_dyck)
    model = Model(VOCAB, D, NL, NH, MAXLEN, **kw)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"--- {name} ---")
    for ep in range(EPOCHS):
        model.train(); el = nb = 0; t0 = time.time()
        for seq, _ in tr:
            logits = model(seq[:, :-1])
            loss = crit(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        li, lo = eval_loss(model, tin, crit), eval_loss(model, tout, crit)
        ai, ao = close_acc(model, tin), close_acc(model, tout)
        betas = [f"{l.attn.beta.item():.2f}" for l in model.layers if isinstance(l.attn.beta, nn.Parameter)]
        bmsg = f" | betas={betas}" if betas else ""
        print(f"  ep{ep+1}: train={el/nb:.4f} | IN loss={li:.4f} acc={ai:.1%} | "
              f"OUT loss={lo:.4f} acc={ao:.1%} | {time.time()-t0:.0f}s{bmsg}", flush=True)
    return {"in_loss": li, "out_loss": lo, "in_acc": ai, "out_acc": ao}

if __name__ == "__main__":
    r1 = run("v2 del usuario (scores = -d^2, sin temperatura)")
    r2 = run("v2 + beta aprendible (init 8.0)", learnable_beta=True)
    print(json.dumps({"v2_usuario": r1, "v2_beta": r2}, indent=1))
