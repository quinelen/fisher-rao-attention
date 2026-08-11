# ============================================================================
# TEST PRE-REGISTRADO DEL KERNEL ESTABILIZADO (extrapolación en código real)
#
# Diagnóstico previo (diag_sigma_ronda2.py, semillas 4,9 buenas / 2,7 malas):
# el modo de falla en L=512 son NORMAS DE MU OUTLIER en posiciones no
# entrenadas (max 15-17 en malas vs 11-13 en buenas; sigma idéntica),
# amplificadas por beta ligeramente mayor. Fix mínimo: acotar ||mu|| <= 12
# (cap elegido del diagnóstico ANTES de este test; no toca semillas buenas).
#
# *** PRE-REGISTRO (antes de ejecutar, 10-ago-2026) ***
# Semillas: 16-35 (veinte, frescas; ninguna usada en experimentos previos).
# Brazos: euclidiano vs FR-estabilizado (mu-cap 12, único cambio).
# ENDPOINT PRIMARIO: bpc zona extrapolativa L=512, FR-estable < Euclidiano,
#   p<0.05 (t pareado bilateral, n=20).
# SECUNDARIOS (reportados, no decisorios): (i) L=256 extrapolativa;
#   (ii) estabilidad: sd(FR-estable) <= 1.5*sd(Euc) en extra L=512;
#   (iii) control IN L=128.
# NEGATIVO DECLARADO SI el primario no se cumple. Tercer y último intento
# del endpoint L=512: sea cual sea el resultado, va al paper.
# ============================================================================
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json
from scipy import stats
from exp_codigo_extrapolativo import (build_corpus, make_vocab, encode,
                                       seqs_of, run, TRAIN_LEN, EVAL_LENS,
                                       N_TRAIN_SEQ, N_EVAL_SEQ)
from exp_tres_geometrias import EuclideanAttention
from fast_gaussian_test import IsotropicHyperbolicAttention

MU_CAP = 12.0

class StableFR(IsotropicHyperbolicAttention):
    """Kernel FR isotrópico + acotación de ||mu|| a una bola de radio MU_CAP.
    Análogo directo del clamp-a-la-bola de las implementaciones de Poincaré."""
    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dh = self.h, self.dh
        mq = self.mu_q(x).view(B, L, H, dh).transpose(1, 2)
        mk = self.mu_k(x).view(B, L, H, dh).transpose(1, 2)
        # --- estabilización: cap de norma (única diferencia con el original) ---
        def cap(m):
            n = m.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            return m * torch.clamp(MU_CAP / n, max=1.0)
        mq, mk = cap(mq), cap(mk)
        sq = (F.softplus(self.sg_q(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        sk = (F.softplus(self.sg_k(x)) + self.eps_sigma).transpose(1, 2).unsqueeze(-1)
        v = self.v(x).view(B, L, H, dh).transpose(1, 2)
        q2 = (mq * mq).sum(-1, keepdim=True)
        k2 = (mk * mk).sum(-1, keepdim=True).transpose(-2, -1)
        dmu2 = (q2 + k2 - 2.0 * mq @ mk.transpose(-2, -1)).clamp_min(0)
        dsg2 = (sq - sk.transpose(-2, -1)).pow(2)
        prod = sq * sk.transpose(-2, -1)
        arg = 1.0 + (0.5 * dmu2 + dsg2) / (2.0 * prod)
        d2 = 2.0 * torch.acosh(arg.clamp(min=1.0 + 1e-7)).pow(2)
        scores = -self.beta * d2 / dh
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}

    ARMS = {"euclidean": EuclideanAttention, "fr_estable": StableFR}
    SEEDS = list(range(16, 36))
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = run(cls, seed, Xtr, evals, VOCAB)
            res[name].append(r)
            print(f"seed {seed:>2} {name:<11}: extra256={r['bpc_extra_L256']:.4f} "
                  f"extra512={r['bpc_extra_L512']:.4f} in128={r['bpc_L128']:.4f} ({r['secs']}s)", flush=True)

    print("\n*** VEREDICTO PRE-REGISTRADO (n=20, semillas 16-35, kernel estabilizado) ***")
    for k, rol in [("bpc_extra_L512", "PRIMARIO"), ("bpc_extra_L256", "secundario"),
                   ("bpc_L128", "control IN")]:
        e = np.array([r[k] for r in res["euclidean"]])
        f_ = np.array([r[k] for r in res["fr_estable"]])
        t, p = stats.ttest_rel(f_, e)
        print(f"  [{rol:>10}] {k}: Euc={e.mean():.4f}±{e.std():.4f}  FRs={f_.mean():.4f}±{f_.std():.4f}  "
              f"Δ={f_.mean()-e.mean():+.4f}  t={t:+.2f}  p={p:.4f}")
    e512 = np.array([r["bpc_extra_L512"] for r in res["euclidean"]])
    f512 = np.array([r["bpc_extra_L512"] for r in res["fr_estable"]])
    t, p = stats.ttest_rel(f512, e512)
    exito = f512.mean() < e512.mean() and p < 0.05
    estable = f512.std() <= 1.5 * e512.std()
    wins = int((f512 < e512).sum())
    print(f"\n  semillas ganadas por FR-estable en extra_L512: {wins}/20")
    print(f"  estabilidad (sd_FR <= 1.5*sd_Euc): {'SÍ' if estable else 'NO'} "
          f"(sd_FR={f512.std():.4f}, sd_Euc={e512.std():.4f})")
    print(f"  ENDPOINT PRIMARIO: {'CUMPLIDO — transferencia extrapolativa a código real DEMOSTRADA con kernel estabilizado' if exito else 'NO CUMPLIDO — tercer intento fallido; el claim queda confinado a lenguajes formales'}")

    with open("resultados_estabilizado.json", "w") as fj:
        json.dump({a: res[a] for a in ARMS}, fj, indent=1)
    print("\nGuardado: resultados_estabilizado.json")
