# Baselines con la MISMA config de la v2 del usuario (d128, 10k, bs64, lr5e-4, 5 ep, seed 1):
# atención estándar y atención Ricci (Mahalanobis-Fisher del proyecto original).
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, json
from v2_test import run, Layer, FisherRaoExactAttention

class StdAttention(nn.Module):
    def __init__(self, d_model, n_heads, eps=1e-6):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)
        self.beta = 1.0
    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, dim=-1)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(ctx), w

class RicciAttention(nn.Module):
    """Mecanismo original del proyecto: distancia (q-k)^2 * g_mid con g = 1/softmax."""
    def __init__(self, d_model, n_heads, eps=1e-6):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)
        self.beta = 1.0
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
        scores = -(diff * diff * g_mid).sum(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, dim=-1)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(ctx), w

if __name__ == "__main__":
    import v2_test
    results = {}
    for name, cls in [("estandar", StdAttention), ("ricci", RicciAttention)]:
        # inyectar la clase de atención en Layer
        orig = v2_test.FisherRaoExactAttention
        v2_test.FisherRaoExactAttention = cls
        try:
            results[name] = run(f"{name} (config v2: d128, 10k, bs64, 5ep, seed1)")
        finally:
            v2_test.FisherRaoExactAttention = orig
    print(json.dumps(results, indent=1))
