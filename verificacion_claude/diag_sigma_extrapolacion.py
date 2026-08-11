# ============================================================================
# DIAGNÓSTICO: ¿por qué algunas semillas de FR fallan en extrapolación L=512?
# Hipótesis: sigma fuera de rango en posiciones no entrenadas (>=128), porque
# la arquitectura post-norm deja pasar embeddings posicionales aleatorios
# crudos a la proyección de sigma, y sigma divide dentro del arccosh.
# Método: reentrenar semillas buenas (4, 9) y malas (2, 7) del experimento
# extrapolativo, extraer estadísticas de sigma por zona (pos<128 vs >=128)
# en input L=512, y correlacionar con el bpc extrapolativo de cada semilla.
# ============================================================================
import torch, numpy as np, json
from exp_codigo_extrapolativo import (build_corpus, make_vocab, encode,
                                       seqs_of, run, TRAIN_LEN, EVAL_LENS,
                                       N_TRAIN_SEQ, N_EVAL_SEQ)
from exp_tres_geometrias import Model
from fast_gaussian_test import IsotropicHyperbolicAttention
import torch.nn.functional as F

class InstrumentedFR(IsotropicHyperbolicAttention):
    """Igual al kernel, pero guarda las sigmas del último forward."""
    def forward(self, x, mask=None):
        B, L, D = x.shape
        sq = (F.softplus(self.sg_q(x)) + self.eps_sigma)   # [B, L, H]
        self.last_sigma = sq.detach()
        return super().forward(x, mask)

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}

    # semillas buenas (bpc extra512 bajo) y malas (alto) según los dos experimentos
    SEEDS = {"buena": [4, 9], "mala": [2, 7]}
    print(f"{'semilla':>8} {'tipo':>6} {'bpc_extra512':>13} "
          f"{'σ zona<128':>22} {'σ zona>=128':>22}")
    rows = []
    for tipo, seeds in SEEDS.items():
        for seed in seeds:
            r = run(InstrumentedFR, seed, Xtr, evals, VOCAB)
            # forward instrumentado sobre un batch L=512 para leer sigmas
            model = None  # el modelo vive dentro de run(); repetimos manualmente
            # rehacemos un mini-entrenamiento no: mejor instrumentar via run
            rows.append((seed, tipo, r["bpc_extra_L512"]))
            print(f"{seed:>8} {tipo:>6} {r['bpc_extra_L512']:>13.4f}   (ver siguiente fase)")
    print("\nNOTA: para sigma por zona se requiere acceso al modelo; ver fase B abajo.")

    # ---- Fase B: entrenar UNA vez por semilla reteniendo el modelo ----
    import torch.nn as nn, torch.optim as optim
    from exp_codigo_extrapolativo import EPOCHS, BS
    def train_keep(seed):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        model = Model(InstrumentedFR, vocab=VOCAB, d_model=128, n_layers=2,
                      n_heads=4, max_len=512)
        crit = nn.CrossEntropyLoss(ignore_index=0)
        opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        for _ in range(EPOCHS):
            model.train()
            perm = rng.permutation(len(Xtr))
            for i in range(0, len(Xtr), BS):
                xb = torch.from_numpy(Xtr[perm[i:i + BS]])
                logits = model(xb[:, :-1])
                loss = crit(logits.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    print(f"\n{'semilla':>8} {'capa':>5} {'σ mediana <128':>15} {'σ mediana >=128':>16} "
          f"{'σ max <128':>11} {'σ max >=128':>12}")
    for tipo, seeds in SEEDS.items():
        for seed in seeds:
            model = train_keep(seed)
            model.eval()
            xb = torch.from_numpy(evals[512][:32])
            with torch.no_grad():
                model(xb[:, :-1])
            for li, lyr in enumerate(model.layers):
                s = lyr.attn.last_sigma          # [B, L, H]
                s_in = s[:, :TRAIN_LEN].flatten()
                s_out = s[:, TRAIN_LEN:].flatten()
                print(f"{seed:>8}({tipo[0]}) {li:>4} {s_in.median():>15.3f} {s_out.median():>16.3f} "
                      f"{s_in.max():>11.2f} {s_out.max():>12.2f}")
