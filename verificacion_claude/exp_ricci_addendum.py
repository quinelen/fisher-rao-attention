# Cuarto brazo del benchmark multi-semilla: Ricci-Mahalanobis original del proyecto
# (distancia (q-k)^2 * g_mid con g = 1/softmax), misma config v2 y mismas 5 semillas
# que exp_tres_geometrias.py. Corre con 2 hilos para no interferir con el run principal
# -> los tiempos (secs) NO son comparables; la precisión sí.
# Ejecutar: ~/asreview/venv/bin/python exp_ricci_addendum.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, json
from exp_tres_geometrias import run_one

class RicciMahalanobisAttention(nn.Module):
    """Mecanismo original del proyecto: aproximación cuadrática de la distancia
    Fisher-Rao con métrica del punto medio, g = 1/p sobre softmax(q), softmax(k)."""
    def __init__(self, d_model, n_heads, eps=1e-12):
        super().__init__()
        self.h, self.dh, self.eps = n_heads, d_model // n_heads, eps
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        qp, kp = F.softmax(q, -1), F.softmax(k, -1)
        g_mid = 0.5 * (1.0 / (qp + self.eps)).unsqueeze(3) + 0.5 * (1.0 / (kp + self.eps)).unsqueeze(2)
        diff = qp.unsqueeze(3) - kp.unsqueeze(2)
        s = -(diff * diff * g_mid).sum(-1)
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(s, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

if __name__ == "__main__":
    torch.set_num_threads(2)  # convivir con el experimento principal
    SEEDS = [1, 2, 3, 4, 5]
    rs = []
    for seed in SEEDS:
        r = run_one(RicciMahalanobisAttention, seed)
        rs.append(r)
        print(f"seed {seed} | ricci-mahalanobis     : IN acc={r['in_acc']:.1%}  OUT acc={r['out_acc']:.1%}  "
              f"OUT loss={r['out_loss']:.4f}  ({r['secs']}s*, {r['params']} params)", flush=True)
    oa = np.array([r["out_acc"] for r in rs]); ol = np.array([r["out_loss"] for r in rs])
    ia = np.array([r["in_acc"] for r in rs])
    print(f"\nricci-mahalanobis: IN {ia.mean():.1%}±{ia.std():.1%} | "
          f"OUT {oa.mean():.1%}±{oa.std():.1%} | loss {ol.mean():.4f}±{ol.std():.4f}")
    with open("resultados_ricci_addendum.json", "w") as f:
        json.dump({"out_acc": oa.tolist(), "in_acc": ia.tolist(), "out_loss": ol.tolist(),
                   "params": rs[0]["params"]}, f, indent=1)
    print("Guardado: resultados_ricci_addendum.json  (*tiempos bajo contención de CPU, no comparables)")
