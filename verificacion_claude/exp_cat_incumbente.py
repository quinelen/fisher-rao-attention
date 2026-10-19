# ============================================================================
# P1 — CAT-STYLE vs FISHER-RAO EN SECUENCIAS (pre-registrado, 17-oct-2026)
#
# CAT (Lin, Ojha & Bai, arXiv:2510.01634; workshop NeurIPS 2025) enruta POR
# TOKEN entre ramas de atención en espacios modelo (euclidiano / hiperbólico /
# esférico) con compuerta diferenciable (el artículo no describe curvatura
# aprendible). Sin código público; se reimplementa su mecanismo central a
# nuestra escala ("CAT-style", así se rotula). Diferencias con el diseño
# publicado:
#  (1) mezcla los PUNTAJES de las tres ramas antes de un único softmax; CAT
#      mezcla las SALIDAS de atenciones separadas por rama, cada una con su
#      propia agregación en la variedad, mapa logarítmico y capa feed-forward;
#  (2) puntajes = distancias cuadráticas negativas entre consulta y clave
#      (geodésica en la bola, angular en la esfera) con temperaturas
#      aprendibles por rama y por cabeza; CAT puntúa la rama hiperbólica con
#      exp(-d) entre consultas proyectadas y la esférica con un producto
#      interno exponenciado;
#  (3) la curvatura c es aprendible (CAT no la describe como aprendible);
#  (4) proyecciones q/k/v/o COMPARTIDAS entre ramas para igualar parámetros;
#  (5) sin la regularización por entropía de la compuerta que usa CAT.
#
# ENDPOINT PRIMARIO: FR-isotrópico vs CAT-style en Dyck-2 OUT acc (n=5 pareado).
# Predicción (superconjunto, §5.1): FR >= CAT. Si CAT gana con p<0.05, la
# lectura superconjunto queda refutada contra el incumbente y así se reporta.
# Equidad: piloto de lr y c_init SOLO para el brazo CAT (semilla 1); euclid y
# FR usan la config publicada sin tocar.
# Ejecutar: ~/asreview/venv/bin/python exp_cat_incumbente.py
# ============================================================================
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, math
import torch.optim as optim
from scipy import stats
from exp_tres_geometrias import (Model, EuclideanAttention, Dyck2Dataset,
                                 collate_dyck, close_acc, eval_loss)
from fast_gaussian_test import IsotropicHyperbolicAttention
from torch.utils.data import DataLoader

def softplus_inv(y):
    return math.log(math.expm1(y))

class CATStyleAttention(nn.Module):
    """Reimplementación a pequeña escala del mecanismo central de CAT (difiere del
    diseño publicado en que mezcla PUNTAJES antes de un softmax en vez de salidas
    de ramas con agregación, mapa logarítmico y feed-forward propios; usa
    distancias cuadráticas q-k, geodésica en la bola y angular en la esfera,
    con temperaturas aprendibles; hace aprendible la curvatura, algo que CAT no
    describe; comparte las proyecciones entre ramas; y no regulariza la
    compuerta por entropía; ver cabecera): tres ramas de atención en ESPACIOS
    MODELO (no Fisher-Rao) mezcladas por una compuerta diferenciable POR TOKEN.
      - euclidiana: producto punto escalado
      - hiperbólica: mapa exponencial al origen de la bola de Poincaré
        (curvatura c aprendible) + distancia geodésica exacta de la bola
      - esférica: proyección a la esfera unitaria + distancia angular
    Proyecciones compartidas entre ramas => parámetros ~iguales al baseline."""
    def __init__(self, d_model, n_heads, c_init=1.0, lr_tag=""):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 3)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)  # α=1/3 al inicio
        self.c_raw = nn.Parameter(torch.tensor(softplus_inv(c_init)))     # curvatura aprendible
        self.beta = nn.Parameter(torch.ones(3, n_heads))                  # temp por rama y cabeza

    def _heads(self, t, B, L):
        return t.view(B, L, self.h, self.dh).transpose(1, 2)              # [B,H,L,dh]

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self._heads(self.q(x), B, L); k = self._heads(self.k(x), B, L)
        v = self._heads(self.v(x), B, L)
        bet = self.beta.clamp(0.05, 50.0)
        # --- rama euclidiana ---
        s_e = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s_e = bet[0].view(1, self.h, 1, 1) * s_e
        # --- rama hiperbólica (bola de Poincaré, curvatura c aprendible) ---
        c = F.softplus(self.c_raw) + 1e-4
        sc = torch.sqrt(c)
        def expmap0(t):
            n = t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            return torch.tanh((sc * n).clamp(max=15.0)) * t / (sc * n)
        u, w_ = expmap0(q), expmap0(k)
        u2 = (u * u).sum(-1, keepdim=True); w2 = (w_ * w_).sum(-1, keepdim=True)
        duw2 = (u2 + w2.transpose(-2, -1) - 2.0 * u @ w_.transpose(-2, -1)).clamp_min(0)
        den = ((1 - c * u2) * (1 - c * w2).transpose(-2, -1)).clamp_min(1e-6)
        argp = 1.0 + 2.0 * c * duw2 / den
        d_p = torch.acosh(argp.clamp(min=1.0 + 1e-7)) / sc
        s_h = -bet[1].view(1, self.h, 1, 1) * d_p.pow(2)
        # --- rama esférica (esfera unitaria, distancia angular) ---
        qs = F.normalize(q, dim=-1); ks = F.normalize(k, dim=-1)
        cosqk = (qs @ ks.transpose(-2, -1)).clamp(-1 + 1e-6, 1 - 1e-6)
        s_s = -bet[2].view(1, self.h, 1, 1) * torch.acos(cosqk).pow(2)
        # --- compuerta por token (enrutamiento CAT) ---
        a = F.softmax(self.gate(x), dim=-1)                                # [B,L,3]
        a = a.transpose(1, 2).unsqueeze(1).unsqueeze(-1)                   # [B,1,3,L,1]
        s = a[:, :, 0] * s_e + a[:, :, 1] * s_h + a[:, :, 2] * s_s
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(s, -1)
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

def run_one_cfg(attn_factory, seed, lr=5e-4, epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(Dyck2Dataset(10000, 20, 60, seed=42), batch_size=bs, collate_fn=collate_dyck)
    tin = DataLoader(Dyck2Dataset(2000, 20, 60, seed=123), batch_size=bs, collate_fn=collate_dyck)
    tout = DataLoader(Dyck2Dataset(2000, 80, 120, seed=456), batch_size=bs, collate_fn=collate_dyck)
    model = Model(attn_factory)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for seq, _ in tr:
            loss = crit(model(seq[:, :-1]).reshape(-1, 5), seq[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return {"in_acc": close_acc(model, tin), "out_acc": close_acc(model, tout),
            "params": sum(p.numel() for p in model.parameters()),
            "secs": round(time.time() - t0)}

if __name__ == "__main__":
    torch.set_num_threads(8)

    # ---------- FASE PILOTO (solo brazo CAT, solo semilla 1) ----------
    print("PILOTO CAT-style (semilla 1): lr x c_init")
    pilot = {}
    for lr in [5e-4, 1e-3]:
        for ci in [0.5, 1.0]:
            fac = (lambda ci_: lambda d, h: CATStyleAttention(d, h, c_init=ci_))(ci)
            r = run_one_cfg(fac, seed=1, lr=lr)
            pilot[(lr, ci)] = r["out_acc"]
            print(f"  lr={lr:.0e} c_init={ci}: OUT={r['out_acc']:.1%} IN={r['in_acc']:.1%} "
                  f"({r['secs']}s, {r['params']} params)", flush=True)
    (best_lr, best_ci) = max(pilot, key=pilot.get)
    print(f"Config elegida para confirmatorio: lr={best_lr:.0e}, c_init={best_ci}\n")

    # ---------- FASE CONFIRMATORIA (3 brazos x 5 semillas pareadas) ----------
    cat_fac = (lambda ci_: lambda d, h: CATStyleAttention(d, h, c_init=ci_))(best_ci)
    ARMS = {
        "euclid":   (EuclideanAttention, 5e-4),
        "fr_iso":   (IsotropicHyperbolicAttention, 5e-4),
        "cat_style": (cat_fac, best_lr),
    }
    SEEDS = [1, 2, 3, 4, 5]
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, (fac, lr) in ARMS.items():
            r = run_one_cfg(fac, seed, lr=lr)
            res[name].append(r)
            print(f"seed {seed} {name:<9}: IN={r['in_acc']:.1%} OUT={r['out_acc']:.1%} "
                  f"({r['secs']}s, {r['params']} params)", flush=True)

    print("\n" + "=" * 70)
    for name in ARMS:
        oa = np.array([r["out_acc"] for r in res[name]])
        print(f"{name:<10} OUT acc = {oa.mean():.2%} ± {oa.std():.2%}  (params {res[name][0]['params']})")

    print("\n*** VEREDICTO PRE-REGISTRADO ***")
    fr = np.array([r["out_acc"] for r in res["fr_iso"]])
    ct = np.array([r["out_acc"] for r in res["cat_style"]])
    eu = np.array([r["out_acc"] for r in res["euclid"]])
    for tag, a, b in [("PRIMARIO  FR vs CAT", fr, ct),
                      ("secundario CAT vs Euc", ct, eu),
                      ("sanidad    FR vs Euc", fr, eu)]:
        t, p = stats.ttest_rel(a, b)
        print(f"  [{tag}]: Δ={a.mean()-b.mean():+.2%}  t={t:+.2f}  p={p:.4f}")
    t, p = stats.ttest_rel(fr, ct)
    if fr.mean() >= ct.mean():
        v = ("FR >= CAT-style: la lectura superconjunto SOBREVIVE al incumbente"
             + (" (diferencia significativa a favor de FR)" if p < 0.05 else " (empate estadístico)"))
    else:
        v = ("CAT-style > FR" + (" con p<0.05: lectura superconjunto REFUTADA contra el incumbente"
             if p < 0.05 else " (no significativo: empate estadístico)"))
    print(f"  => {v}")

    with open("resultados_cat_incumbente.json", "w") as f:
        json.dump({"pilot": {f"lr{lr}_c{ci}": v for (lr, ci), v in pilot.items()},
                   "best": {"lr": best_lr, "c_init": best_ci},
                   **{a: res[a] for a in ARMS}}, f, indent=1)
    print("\nGuardado: resultados_cat_incumbente.json")
