# ============================================================================
# DIAGNÓSTICO RONDA 2: sigma quedó descartada (ronda 1: idéntica en semillas
# buenas y malas). Sospechosos restantes: (a) norma de mu por zona (numerador
# del arccosh), (b) temperatura beta aprendida, (c) entropía de la atención
# por zona de query, (d) magnitud de los scores.
# Mismas 4 semillas: buenas 4, 9 / malas 2, 7.
# ============================================================================
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import torch.optim as optim
from exp_codigo_extrapolativo import (build_corpus, make_vocab, encode,
                                       seqs_of, TRAIN_LEN, EVAL_LENS,
                                       N_TRAIN_SEQ, N_EVAL_SEQ, EPOCHS, BS)
from exp_tres_geometrias import Model
from fast_gaussian_test import IsotropicHyperbolicAttention

class Probe(IsotropicHyperbolicAttention):
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
        d2 = 2.0 * torch.acosh(arg.clamp(min=1.0 + 1e-7)).pow(2)
        scores = -self.beta * d2 / dh
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, -1)
        # --- sondas ---
        self.p_beta = float(self.beta)
        mu_norm = mq.norm(dim=-1)                       # [B,H,L]
        self.p_mu_in = mu_norm[:, :, :TRAIN_LEN].median().item()
        self.p_mu_out = mu_norm[:, :, TRAIN_LEN:].median().item() if L > TRAIN_LEN else float('nan')
        self.p_mu_max_out = mu_norm[:, :, TRAIN_LEN:].max().item() if L > TRAIN_LEN else float('nan')
        ent = -(w * (w + 1e-12).log()).sum(-1)          # [B,H,L]
        self.p_ent_in = ent[:, :, :TRAIN_LEN].mean().item()
        self.p_ent_out = ent[:, :, TRAIN_LEN:].mean().item() if L > TRAIN_LEN else float('nan')
        self.p_d2_med_out = d2[:, :, TRAIN_LEN:, :].median().item() if L > TRAIN_LEN else float('nan')
        out = (w @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)

if __name__ == "__main__":
    torch.set_num_threads(8)
    train_txt, test_txt = build_corpus()
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    X512 = seqs_of(te_arr, N_EVAL_SEQ, 512, rng0)

    def train_keep(seed):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        model = Model(Probe, vocab=VOCAB, d_model=128, n_layers=2, n_heads=4, max_len=512)
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

    print(f"{'seed':>6} {'capa':>5} {'beta':>7} {'|mu| med IN':>12} {'|mu| med OUT':>13} "
          f"{'|mu| max OUT':>13} {'H(attn) IN':>11} {'H(attn) OUT':>12} {'d2 med OUT':>11}")
    for tipo, seeds in [("buena", [4, 9]), ("mala", [2, 7])]:
        for seed in seeds:
            model = train_keep(seed)
            model.eval()
            with torch.no_grad():
                model(torch.from_numpy(X512[:32])[:, :-1])
            for li, lyr in enumerate(model.layers):
                a = lyr.attn
                print(f"{seed:>4}({tipo[0]}) {li:>4} {a.p_beta:>7.3f} {a.p_mu_in:>12.3f} "
                      f"{a.p_mu_out:>13.3f} {a.p_mu_max_out:>13.2f} {a.p_ent_in:>11.3f} "
                      f"{a.p_ent_out:>12.3f} {a.p_d2_med_out:>11.3f}", flush=True)
