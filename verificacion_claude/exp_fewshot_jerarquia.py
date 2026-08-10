# ============================================================================
# VERIFICACIÓN DEL "CROSSOVER DE POCOS DATOS" REPORTADO POR ANTIGRAVITY
#
# Afirmación a verificar: con pocas muestras/clase (~<30), la geometría
# hiperbólica gana en clasificación jerárquica; con más datos, pierde.
# Esa cifra NUNCA fue comparada contra el incumbente real (Poincaré) —
# antigravity solo comparó contra euclidiano. Aquí se agrega Poincaré.
#
# Diseño (mismo rigor que exp_wordnet_incumbentes.py):
#   - Árbol sintético (branching=4, depth=3 -> 64 hojas), embeddings
#     "verdaderos" generados recursivamente (hijo = padre + ruido decreciente
#     por nivel), de forma que hojas del mismo padre están cerca en el
#     espacio de rasgos crudo -- jerarquía genuina, no etiquetas arbitrarias.
#   - Encoder MLP COMPARTIDO (misma arquitectura, mismo optimizador Adam)
#     con 3 cabezas de salida: Euclidiana / Poincaré (vía expmap0) /
#     Fisher-Rao (mu + log-sigma). Presupuesto de parámetros de embedding
#     igualado: D dims para Euclid/Poincaré, (D-1)+1 para Fisher-Rao.
#   - Clasificador: softmax sobre -distancia^2 a prototipos de clase
#     APRENDIBLES en la misma geometría (misma receta para los 3 brazos).
#   - Barrido k = muestras/clase en {5,10,20,30,50,100}, 5 semillas.
#   - Métrica primaria: accuracy exacta de hoja. Métrica secundaria:
#     distancia en el árbol de la predicción al verdadero (mide "qué tan
#     lejos" se equivoca, el argumento que antigravity destacó).
#
# Ejecutar: ~/asreview/venv/bin/python exp_fewshot_jerarquia.py
# ============================================================================
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json
from scipy import stats

EPS = 1e-7

# ---------------------------------------------------------- árbol sintético
def build_tree(branching, depth):
    """Árbol balanceado. Devuelve: lista de hojas, padre de cada nodo,
    profundidad de cada nodo, y una función de distancia en el árbol."""
    nodes = ['root']; parent = {'root': None}; node_depth = {'root': 0}
    frontier = ['root']
    for d in range(depth):
        new_frontier = []
        for p in frontier:
            for i in range(branching):
                n = f'{p}.{i}'
                nodes.append(n); parent[n] = p; node_depth[n] = d + 1
                new_frontier.append(n)
        frontier = new_frontier
    leaves = frontier
    def ancestors(n):
        out = [n]
        while parent[out[-1]] is not None:
            out.append(parent[out[-1]])
        return out
    anc = {n: ancestors(n) for n in nodes}
    def tree_dist(a, b):
        Aa, Ab = set(anc[a]), set(anc[b])
        common = Aa & Ab
        lca_depth = max(node_depth[c] for c in common)
        return (node_depth[a] - lca_depth) + (node_depth[b] - lca_depth)
    return leaves, parent, node_depth, tree_dist

def generate_latents(leaves, parent, node_depth, d_true, seed, decay=0.6, scale0=2.0):
    """Embeddings 'verdaderos' recursivos: hijo = padre + ruido ~ N(0, (scale0*decay^nivel)^2)."""
    rng = np.random.RandomState(seed)
    lat = {'root': np.zeros(d_true)}
    all_nodes = set(parent.keys())
    order = sorted(all_nodes, key=lambda n: node_depth[n])
    for n in order:
        if n == 'root': continue
        lvl = node_depth[n]
        noise = rng.randn(d_true) * (scale0 * (decay ** lvl))
        lat[n] = lat[parent[n]] + noise
    return {l: lat[l] for l in leaves}

def make_dataset(leaves, leaf_latent, d_true, n_pool, obs_noise, seed):
    rng = np.random.RandomState(seed + 999)
    X, y = [], []
    for ci, l in enumerate(leaves):
        pts = leaf_latent[l][None, :] + rng.randn(n_pool, d_true) * obs_noise
        X.append(pts); y.extend([ci] * n_pool)
    return np.concatenate(X).astype(np.float32), np.array(y, dtype=np.int64)

# ---------------------------------------------------------------- encoder
class Encoder(nn.Module):
    def __init__(self, d_in, D, model):
        super().__init__()
        self.model = model
        self.body = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU())
        self.head = nn.Linear(32, D)

    def forward(self, x):
        z = self.head(self.body(x))
        if self.model == 'euclidean':
            return z
        if self.model == 'poincare':
            return expmap0(z)
        if self.model == 'fisher_rao':
            mu, log_s = z[..., :-1], z[..., -1:]
            sigma = F.softplus(log_s) + 0.05
            return torch.cat([mu, sigma], dim=-1)   # convención: última col = sigma (no log)

def expmap0(v, max_norm=0.999):
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x = torch.tanh(n) * v / n
    xn = x.norm(dim=-1, keepdim=True)
    scale = torch.where(xn > max_norm, max_norm / xn, torch.ones_like(xn))
    return x * scale

# ------------------------------------------------------------- distancias
def d2_euclidean(U, V):
    return (U - V).pow(2).sum(-1)

def d2_poincare(U, V):
    uu = U.pow(2).sum(-1).clamp(max=1 - 1e-5)
    vv = V.pow(2).sum(-1).clamp(max=1 - 1e-5)
    uv = (U - V).pow(2).sum(-1)
    x = 1 + 2 * uv / ((1 - uu) * (1 - vv)).clamp_min(EPS)
    return torch.acosh(x.clamp_min(1 + EPS)).pow(2)

def d2_fisher_rao(U, V):
    mu_u, s_u = U[..., :-1], U[..., -1].clamp_min(1e-4)
    mu_v, s_v = V[..., :-1], V[..., -1].clamp_min(1e-4)
    dmu2 = (mu_u - mu_v).pow(2).sum(-1)
    dsg2 = (s_u - s_v).pow(2)
    arg = 1 + (0.5 * dmu2 + dsg2) / (2 * s_u * s_v).clamp_min(EPS)
    return 2.0 * torch.acosh(arg.clamp_min(1 + EPS)).pow(2)

D2 = {'euclidean': d2_euclidean, 'poincare': d2_poincare, 'fisher_rao': d2_fisher_rao}

def init_protos(model, C, D, gen):
    if model == 'fisher_rao':
        p = torch.zeros(C, D)
        p[:, :-1] = torch.randn(C, D - 1, generator=gen) * 0.1
        p[:, -1] = 1.0
        return p.requires_grad_(True)
    p = (torch.randn(C, D, generator=gen) * 0.1).requires_grad_(True)
    return p

def proto_forward(model, raw_protos):
    if model == 'poincare':
        return expmap0(raw_protos)
    if model == 'fisher_rao':
        mu, s = raw_protos[..., :-1], raw_protos[..., -1:]
        return torch.cat([mu, F.softplus(s) + 0.05], -1)
    return raw_protos

# ---------------------------------------------------------------- train/eval
def train_eval(model, X_tr, y_tr, X_te, y_te, C, d_true, D, seed,
               epochs=300, bs=128, lr=1e-2):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = Encoder(d_true, D if model != 'fisher_rao' else D, model)
    protos_raw = init_protos(model, C, D, gen)
    opt = torch.optim.Adam(list(enc.parameters()) + [protos_raw], lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X_tr); yt = torch.from_numpy(y_tr)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            emb = enc(Xt[idx])
            protos = proto_forward(model, protos_raw)
            d2 = D2[model](emb.unsqueeze(1), protos.unsqueeze(0))   # [B,C]
            loss = F.cross_entropy(-d2, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if model == 'fisher_rao':
                with torch.no_grad():
                    protos_raw[:, -1].clamp_(min=-5, max=5)
    enc.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te)
        emb = enc(Xte)
        protos = proto_forward(model, protos_raw)
        d2 = D2[model](emb.unsqueeze(1), protos.unsqueeze(0))
        pred = d2.argmin(-1).numpy()
    acc = float((pred == y_te).mean())
    return acc, pred

# --------------------------------------------------------------------- main
if __name__ == "__main__":
    torch.set_num_threads(8)
    BRANCHING, DEPTH = 4, 3          # 64 hojas -- el "punto dulce" que reportó antigravity
    D_TRUE = 16
    D_EMB = 10                        # mismo presupuesto que WordNet (§5.2)
    K_VALUES = [5, 10, 20, 30, 50, 100]
    N_TEST_PER_CLASS = 20
    SEEDS = [1, 2, 3, 4, 5]
    ARMS = ['euclidean', 'poincare', 'fisher_rao']

    leaves, parent, node_depth, tree_dist = build_tree(BRANCHING, DEPTH)
    C = len(leaves)
    print(f"Árbol: branching={BRANCHING} depth={DEPTH} -> {C} hojas")
    leaf_idx = {l: i for i, l in enumerate(leaves)}
    # matriz de distancias en el árbol entre hojas (para métrica secundaria)
    TD = np.zeros((C, C), dtype=np.int32)
    for i, a in enumerate(leaves):
        for j, b in enumerate(leaves):
            TD[i, j] = tree_dist(a, b)

    results = {k: {a: [] for a in ARMS} for k in K_VALUES}
    tree_err = {k: {a: [] for a in ARMS} for k in K_VALUES}

    for seed in SEEDS:
        leaf_lat = generate_latents(leaves, parent, node_depth, D_TRUE, seed)
        X_pool, y_pool = make_dataset(leaves, leaf_lat, D_TRUE, n_pool=100 + N_TEST_PER_CLASS,
                                      obs_noise=1.0, seed=seed)
        # split fijo train-pool / test, nested para todos los k
        rng = np.random.RandomState(seed + 7)
        train_idx, test_idx = [], []
        for ci in range(C):
            idx = np.where(y_pool == ci)[0]
            rng.shuffle(idx)
            test_idx.extend(idx[:N_TEST_PER_CLASS])
            train_idx.extend(idx[N_TEST_PER_CLASS:])
        X_test, y_test = X_pool[test_idx], y_pool[test_idx]
        X_pool_tr, y_pool_tr = X_pool[train_idx], y_pool[train_idx]

        for k in K_VALUES:
            sel = []
            for ci in range(C):
                idx = np.where(y_pool_tr == ci)[0][:k]
                sel.extend(idx)
            X_tr, y_tr = X_pool_tr[sel], y_pool_tr[sel]
            for a in ARMS:
                t0 = time.time()
                acc, pred = train_eval(a, X_tr, y_tr, X_test, y_test, C, D_TRUE, D_EMB, seed)
                mean_td = float(TD[pred, y_test].mean())
                results[k][a].append(acc)
                tree_err[k][a].append(mean_td)
                print(f"seed={seed} k={k:>3} {a:<11}: acc={acc:.4f}  treeDist_err={mean_td:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print(f"{'k':>4} {'euclidean':>16} {'poincare':>16} {'fisher_rao':>16}")
    print("=" * 80)
    for k in K_VALUES:
        row = [k]
        for a in ARMS:
            m = np.array(results[k][a])
            row.append(f"{m.mean():.4f}±{m.std():.4f}")
        print(f"{row[0]:>4} {row[1]:>16} {row[2]:>16} {row[3]:>16}")

    print("\nContrastes pareados (accuracy, n=5, df=4) -- Bonferroni: 12 tests, alpha_corr=0.0042")
    for k in K_VALUES:
        fr = np.array(results[k]['fisher_rao'])
        for a in ['euclidean', 'poincare']:
            b = np.array(results[k][a])
            t, p = stats.ttest_rel(fr, b)
            sig = "  **sobrevive Bonferroni**" if p < 0.05 / 12 else ""
            verdict = "FR gana" if fr.mean() > b.mean() else "FR pierde"
            print(f"  k={k:>3}  FR vs {a:<10}: Δ={fr.mean()-b.mean():+.4f}  t={t:+.2f}  p={p:.4f}  -> {verdict}{sig}")

    print("\nDistancia en el árbol del error (menor = se equivoca 'más cerca'; media±sd):")
    for k in K_VALUES:
        row = [f"{np.mean(tree_err[k][a]):.3f}±{np.std(tree_err[k][a]):.3f}" for a in ARMS]
        print(f"  k={k:>3}  euclid={row[0]}  poincare={row[1]}  fisher_rao={row[2]}")

    out = {"branching": BRANCHING, "depth": DEPTH, "n_leaves": C, "d_true": D_TRUE, "D_emb": D_EMB,
          "k_values": K_VALUES, "accuracy": results, "tree_dist_error": tree_err}
    with open("resultados_fewshot_jerarquia.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nGuardado: resultados_fewshot_jerarquia.json")
