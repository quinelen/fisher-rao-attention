# Variante mejorada: atención con distancia Fisher-Rao EXACTA (geodésica esférica)
# d_FR(q,k) = 2*arccos( sum_i sqrt(q_i * k_i) )  — forma cerrada, un solo matmul
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json
import sys
sys.path.insert(0, '.')
from train_compare import make_batchset, batches, Block, TF, train_model, VanillaMHA

class FisherRaoMHA(nn.Module):
    """Atención con distancia geodésica exacta de Fisher-Rao (ángulo de Bhattacharyya).
    Coste: un matmul de sqrt(q_p) @ sqrt(k_p)^T — igual de barato que la atención estándar."""
    def __init__(self, d_model, n_heads, beta=8.0):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.beta = nn.Parameter(torch.tensor(beta))  # temperatura aprendible
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
        sq = F.softmax(q, -1).clamp_min(1e-12).sqrt()   # puntos en la esfera unitaria
        sk = F.softmax(k, -1).clamp_min(1e-12).sqrt()
        bc = (sq @ sk.transpose(-2, -1)).clamp(-1 + 1e-7, 1 - 1e-7)  # coef. Bhattacharyya
        d_fr = 2.0 * torch.arccos(bc)                    # distancia geodésica exacta
        scores = -self.beta * d_fr.pow(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

if __name__ == "__main__":
    train_seqs = make_batchset(2000, 20, 60, seed=42)
    tin = make_batchset(400, 20, 60, seed=123)
    tout = make_batchset(400, 80, 120, seed=456)
    print("Fisher-Rao exacta (geodésica esférica, causal) — 4 epochs")
    torch.manual_seed(1)
    res = train_model("FisherRao-Transformer (causal)", TF(FisherRaoMHA), train_seqs, tin, tout)
    print(json.dumps(res, indent=1))
