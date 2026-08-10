# ============================================================================
# Test de la FastGaussianFisherRaoAttention del usuario (log-Bhattacharyya)
# y de la alternativa propuesta: HIPERBOLICA ISOTROPICA EXACTA
#   - N(mu, sigma^2 I) con sigma escalar por token/cabeza
#   - variedad Fisher-Rao = espacio hiperbolico H^{d+1} (curvatura constante)
#   - distancia exacta: sqrt(2)*arccosh(1 + (||dmu||^2/2 + dsig^2)/(2 s_i s_j))
#   - ||dmu||^2 pareado via UN matmul -> arccosh solo sobre [L,L]
# Modos:
#   bench : micro-benchmark forward+backward de las 4 atenciones (rapido)
#   acc   : espera a que terminen los experimentos en curso y entrena
#           (FastBC seed 1; Isotropica seeds 1-5) con la config v2
# ============================================================================
import sys, os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from exp_tres_geometrias import (EuclideanAttention, HyperbolicGaussianAttention,
                                 run_one, Model)

class FastGaussianFisherRaoAttention(nn.Module):
    """Version del usuario: score = media del log-coeficiente de Bhattacharyya
    gaussiano por coordenada. NOTA: sigue materializando tensores 5D [B,H,L,L,dh];
    y usa una sola proyeccion mu/sigma para query y key (log_BC(i,i)=0 = maximo)."""
    def __init__(self, d_model, n_heads, eps=1e-5):
        super().__init__()
        self.h, self.dh, self.eps = n_heads, d_model // n_heads, eps
        self.mu_proj = nn.Linear(d_model, d_model)
        self.sigma_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        mu = self.mu_proj(x).view(B, L, self.h, self.dh).transpose(1, 2)
        sigma = F.softplus(self.sigma_proj(x).view(B, L, self.h, self.dh).transpose(1, 2)) + self.eps
        v = self.v_proj(x).view(B, L, self.h, self.dh).transpose(1, 2)
        mu_q, mu_k = mu.unsqueeze(3), mu.unsqueeze(2)
        sig_q, sig_k = sigma.unsqueeze(3), sigma.unsqueeze(2)
        sigma_sum2 = sig_q**2 + sig_k**2
        exponent = -0.25 * ((mu_q - mu_k)**2 / sigma_sum2)
        scale_term = 0.5 * torch.log((2.0 * sig_q * sig_k / sigma_sum2) + self.eps)
        scores = (exponent + scale_term).mean(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        return self.out_proj(torch.matmul(w, v).transpose(1, 2).contiguous().view(B, L, D))

class IsotropicHyperbolicAttention(nn.Module):
    """Gaussiana isotropica N(mu, sigma^2 I), sigma ESCALAR por token/cabeza.
    Variedad Fisher-Rao: H^{dh+1} con ds^2 = (||dmu||^2 + 2 dsigma^2)/sigma^2.
    Distancia geodesica EXACTA con ||dmu||^2 descompuesto en un matmul:
        ||mu_i - mu_j||^2 = ||mu_i||^2 + ||mu_j||^2 - 2 mu_i . mu_j
    -> arccosh opera sobre [B,H,L,L], no sobre [B,H,L,L,dh]."""
    def __init__(self, d_model, n_heads, eps_sigma=0.05, beta_init=1.0):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.mu_q = nn.Linear(d_model, d_model); self.mu_k = nn.Linear(d_model, d_model)
        self.sg_q = nn.Linear(d_model, n_heads); self.sg_k = nn.Linear(d_model, n_heads)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.eps_sigma = eps_sigma
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x, mask=None):
        B, L, D = x.shape
        mq = self.mu_q(x).view(B, L, self.h, self.dh).transpose(1, 2)   # [B,H,L,dh]
        mk = self.mu_k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        sq = (F.softplus(self.sg_q(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)  # [B,H,L,1]
        sk = (F.softplus(self.sg_k(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        # ||dmu||^2 pareado con un matmul
        q2 = (mq * mq).sum(-1, keepdim=True)                    # [B,H,L,1]
        k2 = (mk * mk).sum(-1, keepdim=True).transpose(-2, -1)  # [B,H,1,L]
        dmu2 = (q2 + k2 - 2.0 * mq @ mk.transpose(-2, -1)).clamp_min(0)  # [B,H,L,L]
        dsg2 = (sq - sk.transpose(-2, -1)).pow(2)               # [B,H,L,L]
        prod = sq * sk.transpose(-2, -1)
        arg = 1.0 + (0.5 * dmu2 + dsg2) / (2.0 * prod)
        d2 = 2.0 * torch.acosh(arg.clamp(min=1.0 + 1e-7)).pow(2)  # [B,H,L,L]
        scores = -self.beta * d2 / self.dh
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

def bench():
    torch.set_num_threads(1)  # 1 hilo: los experimentos principales siguen corriendo
    B, L, D, H = 32, 60, 128, 4
    x = torch.randn(B, L, D)
    mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)
    mods = {
        "euclidiana (referencia)":        EuclideanAttention(D, H),
        "gaussiana exacta (arccosh 5D)":  HyperbolicGaussianAttention(D, H),
        "FastBC del usuario (log-BC 5D)": FastGaussianFisherRaoAttention(D, H),
        "isotropica exacta (matmul)":     IsotropicHyperbolicAttention(D, H),
    }
    print(f"Micro-benchmark fwd+bwd, B={B} L={L} d={D} H={H}, 1 hilo (bajo contención):")
    for name, m in mods.items():
        out = m(x, mask); loss = out.sum(); loss.backward()  # warmup
        t0 = time.time()
        for _ in range(5):
            m.zero_grad()
            out = m(x, mask)
            out.sum().backward()
        ms = (time.time() - t0) / 5 * 1000
        print(f"  {name:<34} {ms:8.1f} ms/paso", flush=True)

def acc():
    # esperar a que terminen los dos experimentos en curso (liberan la CPU)
    here = os.path.dirname(os.path.abspath(__file__))
    while not (os.path.exists(os.path.join(here, "resultados_tres_geometrias.json"))
               and os.path.exists(os.path.join(here, "resultados_ricci_addendum.json"))):
        time.sleep(30)
    torch.set_num_threads(8)
    out = {}
    r = run_one(FastGaussianFisherRaoAttention, seed=1)
    out["fast_bc_seed1"] = r
    print(f"FastBC (usuario)  seed 1: IN acc={r['in_acc']:.1%}  OUT acc={r['out_acc']:.1%}  "
          f"OUT loss={r['out_loss']:.4f}  ({r['secs']}s, {r['params']} params)", flush=True)
    iso = []
    for seed in [1, 2, 3, 4, 5]:
        r = run_one(IsotropicHyperbolicAttention, seed=seed)
        iso.append(r)
        print(f"Isotropica exacta seed {seed}: IN acc={r['in_acc']:.1%}  OUT acc={r['out_acc']:.1%}  "
              f"OUT loss={r['out_loss']:.4f}  ({r['secs']}s, {r['params']} params)", flush=True)
    oa = np.array([r["out_acc"] for r in iso])
    print(f"\nIsotropica exacta (5 seeds): OUT acc = {oa.mean():.1%} ± {oa.std():.1%}")
    out["isotropica"] = iso
    with open(os.path.join(here, "resultados_fast_gaussian.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Guardado: resultados_fast_gaussian.json")

if __name__ == "__main__":
    {"bench": bench, "acc": acc}[sys.argv[1] if len(sys.argv) > 1 else "bench"]()
