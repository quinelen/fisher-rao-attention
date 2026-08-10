# Comparación final WordNet mamíferos: Fisher-Rao vs incumbentes (Poincaré, Lorentz)
# + control euclidiano. Cada brazo con SU mejor lr (barrido previo), 400 épocas,
# bs=20, 50 negativos, 5 semillas. Incumbentes reproducen su MAP publicado (~0.93).
import exp_wordnet_incumbentes as X
import torch, numpy as np, time, json
from scipy import stats

idx, E = X.build_mammal_closure(); N = len(idx)

def train2(model, D, seed, epochs, n_neg, lr, burnin, bs):
    gen = torch.Generator().manual_seed(seed)
    emb = X.init_emb(model, N, D, gen); dist = X.DIST[model]
    Et = torch.from_numpy(E)
    for ep in range(epochs):
        cur = lr / 10 if ep < burnin else lr
        perm = torch.randperm(len(E), generator=gen)
        for i in range(0, len(E), bs):
            b = Et[perm[i:i + bs]]; u, v = b[:, 0], b[:, 1]
            neg = torch.randint(0, N, (len(b), n_neg), generator=gen)
            Vc = emb[torch.cat([v.unsqueeze(1), neg], 1)]
            d = dist(emb[u].unsqueeze(1).expand_as(Vc), Vc)
            torch.nn.functional.cross_entropy(-d, torch.zeros(len(b), dtype=torch.long)).backward()
            X.rsgd_step(model, emb, cur)
    return emb.detach()

# mejor lr por brazo (del barrido)
BEST_LR = {'euclidean': 3.0, 'poincare': 1.0, 'lorentz': 1.0, 'fisher_rao': 3.0}
ARMS = ['euclidean', 'poincare', 'lorentz', 'fisher_rao']
SEEDS = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    torch.set_num_threads(8)
    print(f"WordNet mammal: N={N} nodos, {len(E)} aristas (cierre transitivo)")
    print("D=10, bs=20, 50 neg, 400 épocas, mejor lr por brazo\n")
    res = {a: {'map': [], 'mr': []} for a in ARMS}
    for seed in SEEDS:
        for a in ARMS:
            t0 = time.time()
            emb = train2(a, 10, seed, 400, 50, BEST_LR[a], 10, 20)
            mp, mr = X.evaluate(a, emb, E, N)
            res[a]['map'].append(mp); res[a]['mr'].append(mr)
            print(f"seed {seed} {a:<11}: MAP={mp:.4f} MeanRank={mr:6.2f} (lr={BEST_LR[a]}, {time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 64)
    print(f"{'modelo':<12} {'MAP (media±sd)':>20} {'MeanRank':>18}")
    print("=" * 64)
    for a in ARMS:
        m = np.array(res[a]['map']); r = np.array(res[a]['mr'])
        tag = "  <- NUESTRO" if a == 'fisher_rao' else ("  <- INCUMBENTE" if a in ('poincare','lorentz') else "")
        print(f"{a:<12} {m.mean():>12.4f}±{m.std():.4f} {r.mean():>11.2f}±{r.std():.2f}{tag}")

    print("\nFisher-Rao vs incumbentes (MAP, pareado, n=5):")
    fr = np.array(res['fisher_rao']['map'])
    for a in ['poincare', 'lorentz', 'euclidean']:
        b = np.array(res[a]['map']); t, p = stats.ttest_rel(fr, b)
        v = "FR GANA" if fr.mean() > b.mean() else "FR PIERDE"
        print(f"  vs {a:<10}: Δ={fr.mean()-b.mean():+.4f}  t={t:+.2f}  p={p:.4f}  -> {v}")

    with open("resultados_wordnet_final.json", "w") as f:
        json.dump(res, f, indent=1)
    print("\nGuardado: resultados_wordnet_final.json")
