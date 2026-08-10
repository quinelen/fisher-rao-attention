# ============================================================================
# PRUEBA DE NOVEDAD: ¿es la geometría hiperbólica (Fisher-Rao) la que ayuda,
# o basta con usar tokens gaussianos?
#
# Control de la prior-art más cercana (Stochastic ViT, Wasserstein-aware attention,
# arXiv 2311.18645): MISMOS tokens gaussianos N(mu, sigma^2 I), MISMA arquitectura,
# solo cambia la DISTANCIA entre distribuciones:
#   - Fisher-Rao (nuestra): geometría HIPERBÓLICA, curvatura K<0
#       d^2 = 2 arccosh^2(1 + (||dmu||^2/2 + dsig^2)/(2 s_i s_j))
#   - Wasserstein-2 / Bures (Stochastic ViT): geometría de transporte óptimo,
#       NO hiperbólica.  Para isotrópicas:  W2^2 = ||dmu||^2 + d_h (s_i - s_j)^2
# Ambas se reducen a un matmul (||dmu||^2 = ||mu_i||^2+||mu_j||^2-2 mu_i.mu_j).
# Si Fisher-Rao > Wasserstein en Dyck-2 extrapolativo, la ventaja proviene de la
# CURVATURA, no de la representación gaussiana -> distingue empíricamente de la
# prior art y refuerza la tesis "curvatura para jerarquías".
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, json
from scipy import stats
from exp_tres_geometrias import run_one, EuclideanAttention
from fast_gaussian_test import IsotropicHyperbolicAttention

class WassersteinIsotropicAttention(nn.Module):
    """Tokens gaussianos isotrópicos N(mu, sigma^2 I), distancia Wasserstein-2 (Bures).
    Geometría de transporte óptimo (NO hiperbólica). Control de Stochastic ViT."""
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
        mq = self.mu_q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        mk = self.mu_k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        sq = (F.softplus(self.sg_q(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        sk = (F.softplus(self.sg_k(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        q2 = (mq * mq).sum(-1, keepdim=True)
        k2 = (mk * mk).sum(-1, keepdim=True).transpose(-2, -1)
        dmu2 = (q2 + k2 - 2.0 * mq @ mk.transpose(-2, -1)).clamp_min(0)
        dsg2 = (sq - sk.transpose(-2, -1)).pow(2)
        # Wasserstein-2 entre isotrópicas: ||dmu||^2 + d_h (s_i - s_j)^2
        w2 = dmu2 + self.dh * dsg2
        scores = -self.beta * w2 / self.dh
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

if __name__ == "__main__":
    torch.set_num_threads(8)
    ARMS = {
        "euclidiana":        EuclideanAttention,             # referencia plana, sin gaussianas
        "wasserstein (Bures)": WassersteinIsotropicAttention, # gaussianas, geometría NO hiperbólica
        "fisher-rao (hiperb)": IsotropicHyperbolicAttention,  # gaussianas, geometría hiperbólica (nuestra)
    }
    SEEDS = [1, 2, 3, 4, 5]
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = run_one(cls, seed)
            res[name].append(r)
            print(f"seed {seed} | {name:<20}: IN acc={r['in_acc']:.1%}  OUT acc={r['out_acc']:.1%}  "
                  f"OUT loss={r['out_loss']:.4f}  ({r['secs']}s, {r['params']} params)", flush=True)

    print("\n" + "=" * 76)
    summ = {}
    for name, rs in res.items():
        oa = np.array([r["out_acc"] for r in rs])
        summ[name] = oa.tolist()
        print(f"{name:<22} OUT acc = {oa.mean():.2%} ± {oa.std():.2%}")

    print("\nContraste decisivo (n=5, df=4):")
    a = np.array(summ["fisher-rao (hiperb)"]); b = np.array(summ["wasserstein (Bures)"])
    t, p = stats.ttest_rel(a, b)
    print(f"  Fisher-Rao vs Wasserstein (mismos tokens gaussianos): "
          f"Δ={np.mean(a-b):+.2%}  t={t:+.2f}  p={p:.4f}")
    c = np.array(summ["euclidiana"])
    tw, pw = stats.ttest_rel(b, c)
    print(f"  Wasserstein vs Euclidiana (¿aportan las gaussianas per se?): "
          f"Δ={np.mean(b-c):+.2%}  t={tw:+.2f}  p={pw:.4f}")

    with open("resultados_wasserstein_vs_fr.json", "w") as f:
        json.dump(summ, f, indent=1)
    print("\nGuardado: resultados_wasserstein_vs_fr.json")
