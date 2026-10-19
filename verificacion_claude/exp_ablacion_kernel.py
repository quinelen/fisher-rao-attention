# ============================================================================
# ABLACIONES DE KERNEL CON LA MISMA REPRESENTACIÓN GAUSSIANA (§4.6 y §4.5)
#
# Mismos tokens N(mu, sigma^2 I), misma arquitectura y configuración §4.1;
# solo cambia la función de puntaje:
#   - KL / Bregman (§4.6): s = -beta * KL(q_i || k_j) / d_h.  KL entre
#     gaussianas es la divergencia de Bregman de la log-partición y coincide
#     con la métrica de Fisher a segundo orden (misma geometría LOCAL de la
#     información), pero crece cuadráticamente con la separación de medias,
#     mientras la geodésica hiperbólica de Fisher-Rao crece logarítmicamente
#     (curvatura negativa GLOBAL). Aísla lo global de lo local.
#   - Laplaciano geodésico (§4.5): s = -beta * d_FR / sqrt(d_h). El kernel
#     gaussiano geodésico exp(-beta d^2) no es definido positivo en espacios
#     curvos para todo beta, mientras exp(-beta d) sí lo es en espacios
#     hiperbólicos (Feragen, Lauze & Hauberg, CVPR 2015) -> única forma del
#     kernel compatible con linealización por features aleatorias.
# Comparación pareada (semillas 1-5) contra los brazos euclidiano y
# Fisher-Rao isotrópico de resultados_tres_geometrias.json y
# resultados_fast_gaussian.json (mismas semillas, misma configuración).
# Ejecutar: python exp_ablacion_kernel.py
# ============================================================================
import torch
import torch.nn.functional as F
import numpy as np, json, math
from scipy import stats
from exp_tres_geometrias import run_one
from fast_gaussian_test import IsotropicHyperbolicAttention


def _gauss_tokens(self, x):
    B, L, D = x.shape
    H, dh = self.h, self.dh
    mq = self.mu_q(x).view(B, L, H, dh).transpose(1, 2)
    mk = self.mu_k(x).view(B, L, H, dh).transpose(1, 2)
    sq = (F.softplus(self.sg_q(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
    sk = (F.softplus(self.sg_k(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
    v = self.v(x).view(B, L, H, dh).transpose(1, 2)
    q2 = (mq * mq).sum(-1, keepdim=True)
    k2 = (mk * mk).sum(-1, keepdim=True).transpose(-2, -1)
    dmu2 = (q2 + k2 - 2.0 * mq @ mk.transpose(-2, -1)).clamp_min(0)
    return dmu2, sq, sk.transpose(-2, -1), v


def _attend(self, s, v, mask, x):
    B, L, D = x.shape
    if mask is not None:
        s = s.masked_fill(mask == 0, float('-inf'))
    return self.o((F.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B, L, D))


class KLGaussAttention(IsotropicHyperbolicAttention):
    """s = -beta * KL(N(mu_q, s_q^2 I) || N(mu_k, s_k^2 I)) / d_h."""
    def forward(self, x, mask=None):
        dmu2, sq, skT, v = _gauss_tokens(self, x)
        dh = self.dh
        r2 = (sq / skT) ** 2
        kl = 0.5 * (dh * r2 + dmu2 / skT ** 2 - dh - dh * torch.log(r2))
        return _attend(self, -self.beta * kl / dh, v, mask, x)


class LaplaceFRAttention(IsotropicHyperbolicAttention):
    """s = -beta * d_FR / sqrt(d_h): kernel laplaciano geodésico."""
    def forward(self, x, mask=None):
        dmu2, sq, skT, v = _gauss_tokens(self, x)
        arg = 1.0 + (0.5 * dmu2 + (sq - skT) ** 2) / (2.0 * sq * skT)
        d = math.sqrt(2.0) * torch.acosh(arg.clamp(min=1.0 + 1e-7))
        return _attend(self, -self.beta * d / math.sqrt(self.dh), v, mask, x)


def paired(a, b):
    a, b = np.array(a), np.array(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {"delta_pp": 100 * d.mean(), "ci95_pp": [100 * (d.mean() - half), 100 * (d.mean() + half)],
            "t": float(t), "p": float(p), "wins": int((d > 0).sum())}


if __name__ == "__main__":
    torch.set_num_threads(8)
    SEEDS = [1, 2, 3, 4, 5]
    eu = json.load(open("resultados_tres_geometrias.json"))["euclidiana (K=0)"]["out_acc"]
    fr =[r["out_acc"] for r in json.load(open("resultados_fast_gaussian.json"))["isotropica"]]
    out = {}
    for name, cls in [("kl_bregman", KLGaussAttention), ("laplaciano", LaplaceFRAttention)]:
        rs = []
        for s in SEEDS:
            r = run_one(cls, s)
            rs.append(r)
            print(f"{name} seed {s}: IN={r['in_acc']:.2%} OUT={r['out_acc']:.2%} "
                  f"({r['secs']}s, {r['params']} params)", flush=True)
        oa = [r["out_acc"] for r in rs]
        out[name] = {"runs": rs, "vs_fisher_rao": paired(fr, oa), "vs_euclidean": paired(oa, eu)}
        print(f"  {name}: OUT {100*np.mean(oa):.2f}% | Fisher-Rao - {name}: {out[name]['vs_fisher_rao']} "
              f"| {name} - Euclidean: {out[name]['vs_euclidean']}\n", flush=True)
    with open("resultados_ablacion_kernel.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Guardado: resultados_ablacion_kernel.json")
