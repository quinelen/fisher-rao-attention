# ============================================================================
# P3 — MECANISMO DEL UMBRAL d_h ≈ 16 (pre-registrado en PREREGISTRO_FASE2_CPU.md)
#
# H-c (régimen de curvatura): el kernel FR solo rinde cuando los tokens operan
# a distancias geodésicas comparables al radio de curvatura R = sqrt(2); con
# d_h pequeño la configuración aprendida queda en régimen casi-plano (d << R).
#
# MÉTRICA PRIMARIA: media de d/R ponderada por la atención (lo que el softmax
# integra). CRITERIOS (escritos antes): (i) crece monótona con d_h en las 3
# semillas; (ii) cruza 1.0 entre d_h=8 y d_h=16 (donde §4.7 ubicó el cruce).
# Ejecutar: ~/asreview/venv/bin/python exp_mecanismo_umbral.py
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, math
import torch.optim as optim
from exp_tres_geometrias import Model, Dyck2Dataset, collate_dyck
from fast_gaussian_test import IsotropicHyperbolicAttention

R_CURV = math.sqrt(2.0)   # radio de curvatura de la métrica (1'): K = -1/2

class InstrumentedFR(IsotropicHyperbolicAttention):
    """Kernel FR isotrópico idéntico; si self.rec no es None, registra en eval
    la distribución de d/R y la d/R ponderada por la atención."""
    def forward(self, x, mask=None):
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
        dsg2 = (sq - sk.transpose(-2, -1)).pow(2)
        prod = sq * sk.transpose(-2, -1)
        arg = 1.0 + (0.5 * dmu2 + dsg2) / (2.0 * prod)
        d = math.sqrt(2.0) * torch.acosh(arg.clamp(min=1.0 + 1e-7))   # distancia geodésica
        d2 = d.pow(2)
        scores = -self.beta * d2 / dh
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        if self.rec is not None and not self.training:
            with torch.no_grad():
                dr = d / R_CURV                                        # [B,H,L,L]
                # solo tokens reales: query i < longitud real (sin relleno)
                pos = torch.arange(L)
                real_q = pos[None, :] < self.valid_len[:, None]        # [B,L]
                pair = (torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)[None]
                        & real_q[:, :, None])                          # [B,L,L], j<i<len
                pair = pair[:, None].expand(B, H, L, L)
                vals = dr[pair]
                if len(vals) > 20000:
                    vals = vals[torch.randperm(len(vals))[:20000]]
                watt = (w * dr).sum(-1)[real_q[:, None].expand(B, H, L)]   # E_w[d/R] por query real
                self.rec["d_over_R"].append(vals.cpu().numpy())
                self.rec["att_weighted"].append(watt.cpu().numpy())
        return self.o((w @ v).transpose(1, 2).contiguous().view(B, L, D))

InstrumentedFR.rec = None
InstrumentedFR.valid_len = None

def batches(seqs, bs, rng=None):
    idx = rng.permutation(len(seqs)) if rng is not None else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        yield collate_dyck([seqs[j] for j in idx[i:i + bs]])

def run_point(d_model, seed, n_heads=4, epochs=5, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = list(iter(Dyck2Dataset(2000, 20, 60, seed=42)))
    tin = list(iter(Dyck2Dataset(400, 20, 60, seed=123)))
    model = Model(InstrumentedFR, d_model=d_model, n_heads=n_heads)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, _ in batches(tr, bs, rng):
            loss = crit(model(xb[:, :-1]).reshape(-1, 5), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    # --- instrumentación en eval IN ---
    recs = []
    for lyr in model.layers:
        lyr.attn.rec = {"d_over_R": [], "att_weighted": []}
        recs.append(lyr.attn.rec)
    model.eval()
    with torch.no_grad():
        for xb, lens in batches(tin, bs):
            InstrumentedFR.valid_len = lens - 1                # longitud real de la entrada xb[:, :-1]
            model(xb[:, :-1])
    out = {"secs": round(time.time() - t0), "beta": [float(l.attn.beta) for l in model.layers]}
    aw_all, dv_all = [], []
    for li, rec in enumerate(recs):
        dv = np.concatenate(rec["d_over_R"]); aw = np.concatenate(rec["att_weighted"])
        out[f"L{li}_med_dR"] = float(np.median(dv))
        out[f"L{li}_attw_dR"] = float(aw.mean())
        aw_all.append(aw); dv_all.append(dv)
        lyr_obj = model.layers[li].attn; lyr_obj.rec = None
    dv, aw = np.concatenate(dv_all), np.concatenate(aw_all)
    out["med_dR"] = float(np.median(dv)); out["p90_dR"] = float(np.percentile(dv, 90))
    out["attw_dR"] = float(aw.mean())
    out["frac_curvo"] = float((dv > 1.0).mean())   # fracción de pares más allá de R
    return out

if __name__ == "__main__":
    torch.set_num_threads(8)
    D_MODELS = [16, 32, 64, 128]          # d_h = 4, 8, 16, 32
    SEEDS = [6, 7, 8]
    res = {d: [] for d in D_MODELS}
    for d in D_MODELS:
        for s in SEEDS:
            r = run_point(d, s)
            res[d].append(r)
            print(f"d_model={d:>4} (d_h={d//4:>2}) seed {s}: attw_d/R={r['attw_dR']:.3f}  "
                  f"med={r['med_dR']:.3f}  p90={r['p90_dR']:.3f}  frac>R={r['frac_curvo']:.2f}  "
                  f"beta={['%.2f' % b for b in r['beta']]}  ({r['secs']}s)", flush=True)

    print("\n" + "=" * 78)
    print(f"{'d_h':>5} {'attw d/R (PRIMARIA)':>22} {'mediana d/R':>14} {'frac pares >R':>15}")
    summ = {}
    for d in D_MODELS:
        aw = np.array([r["attw_dR"] for r in res[d]])
        md = np.array([r["med_dR"] for r in res[d]])
        fc = np.array([r["frac_curvo"] for r in res[d]])
        summ[d] = {"attw": aw.tolist(), "med": md.tolist(), "frac": fc.tolist()}
        print(f"{d//4:>5} {aw.mean():>12.3f}±{aw.std():.3f} {md.mean():>10.3f}±{md.std():.3f} "
              f"{fc.mean():>11.2f}±{fc.std():.2f}")

    # --- veredicto pre-registrado ---
    aw_by_seed = np.array([[res[d][i]["attw_dR"] for d in D_MODELS] for i in range(len(SEEDS))])
    mono = all(all(np.diff(row) > 0) for row in aw_by_seed)
    m8 = np.mean([r["attw_dR"] for r in res[32]])    # d_h=8
    m16 = np.mean([r["attw_dR"] for r in res[64]])   # d_h=16
    cruza = (m8 < 1.0) and (m16 >= 1.0)
    print("\n*** VEREDICTO PRE-REGISTRADO (H-c, régimen de curvatura) ***")
    print(f"  (i)  monotonía de attw d/R con d_h en las 3 semillas: {'SÍ' if mono else 'NO'}")
    print(f"  (ii) cruce de 1.0 entre d_h=8 ({m8:.3f}) y d_h=16 ({m16:.3f}): {'SÍ' if cruza else 'NO'}")
    if mono and cruza:
        v = "H-c APOYADA: el umbral coincide con la entrada al régimen curvo"
    elif mono:
        v = "APOYO PARCIAL: el régimen escala con d_h pero el cruce de R no coincide con el umbral"
    else:
        v = "NEGATIVO: H-c no explica el umbral; el mecanismo sigue abierto"
    print(f"  => {v}")

    with open("resultados_mecanismo_umbral.json", "w") as f:
        json.dump({str(d): {"runs": res[d], **summ[d]} for d in D_MODELS}, f, indent=1)
    print("\nGuardado: resultados_mecanismo_umbral.json")
