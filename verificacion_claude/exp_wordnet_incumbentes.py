# ============================================================================
# COMPARACIÓN CONTRA LOS INCUMBENTES  (no contra euclidiano de juguete)
#
# Benchmark canónico del campo: subárbol de MAMÍFEROS de WordNet, cierre
# transitivo de la relación de hiperonimia. Es la tarea exacta con la que
# Nickel & Kiela (NeurIPS 2017) establecieron los Poincaré embeddings.
#
# Brazos (todos: mismo grafo, misma pérdida, mismos negativos, mismas épocas,
#         mismo presupuesto de parámetros por nodo):
#   1. Euclidean      — d(u,v) = ||u-v||                        (control plano)
#   2. Poincaré       — Nickel & Kiela 2017, RSGD correcto      (INCUMBENTE)
#   3. Lorentz        — Nickel & Kiela 2018, hiperboloide       (INCUMBENTE)
#   4. Fisher-Rao ISO — N(mu, sigma^2 I), d = sqrt(2) arccosh() (NUESTRO)
#
# Presupuesto igualado: dim D por nodo en todos.
#   Poincaré/Lorentz/Euclidean: D coords.  Fisher-Rao: (D-1) para mu + 1 para sigma.
#
# Métricas estándar de la literatura: MAP y Mean Rank de reconstrucción.
# Ejecutar: ~/asreview/venv/bin/python exp_wordnet_incumbentes.py
# ============================================================================
import torch, numpy as np, time, json, math
from collections import defaultdict
from scipy import stats

DEV = 'cpu'
EPS = 1e-7

# ---------------------------------------------------------------- datos
def build_mammal_closure():
    """Cierre transitivo de hiperonimia bajo mammal.n.01 (benchmark N&K 2017)."""
    from nltk.corpus import wordnet as wn
    root = wn.synset('mammal.n.01')
    nodes, edges = set(), set()
    stack = [root]
    children = defaultdict(list)
    while stack:
        s = stack.pop()
        if s.name() in nodes:
            continue
        nodes.add(s.name())
        for h in s.hyponyms():
            children[s.name()].append(h.name())
            stack.append(h)
    # cierre transitivo: (descendiente, ancestro) para todo par
    def descendants(n, memo={}):
        if n in memo: return memo[n]
        out = set()
        for c in children[n]:
            out.add(c); out |= descendants(c)
        memo[n] = out
        return out
    for n in nodes:
        for d in descendants(n):
            edges.add((d, n))          # (hijo/descendiente, padre/ancestro)
    idx = {n: i for i, n in enumerate(sorted(nodes))}
    E = np.array([[idx[a], idx[b]] for a, b in sorted(edges)], dtype=np.int64)
    return idx, E

# ------------------------------------------------------- distancias por modelo
def d_euclidean(U, V):
    return (U - V).pow(2).sum(-1).clamp_min(EPS).sqrt()

def d_poincare(U, V):
    uu = U.pow(2).sum(-1).clamp(max=1 - 1e-5)
    vv = V.pow(2).sum(-1).clamp(max=1 - 1e-5)
    uv = (U - V).pow(2).sum(-1)
    x = 1 + 2 * uv / ((1 - uu) * (1 - vv)).clamp_min(EPS)
    return torch.acosh(x.clamp_min(1 + EPS))

def lorentz_inner(U, V):
    # <u,v>_L = -u0 v0 + sum ui vi
    return -U[..., 0] * V[..., 0] + (U[..., 1:] * V[..., 1:]).sum(-1)

def d_lorentz(U, V):
    return torch.acosh((-lorentz_inner(U, V)).clamp_min(1 + EPS))

def d_fisher_rao(U, V):
    """U,V = [..., D] con [:-1]=mu, [-1]=log sigma.  Métrica (1'), K=-1/2."""
    mu_u, s_u = U[..., :-1], U[..., -1].exp().clamp_min(1e-4)
    mu_v, s_v = V[..., :-1], V[..., -1].exp().clamp_min(1e-4)
    dmu2 = (mu_u - mu_v).pow(2).sum(-1)
    dsg2 = (s_u - s_v).pow(2)
    arg = 1 + (0.5 * dmu2 + dsg2) / (2 * s_u * s_v).clamp_min(EPS)
    return math.sqrt(2) * torch.acosh(arg.clamp_min(1 + EPS))

# ------------------------------------------------------------ optimización
def rsgd_step(model, emb, lr):
    """Paso de descenso: riemanniano donde la variedad lo exige."""
    g = emb.grad
    if g is None: return
    with torch.no_grad():
        if model == 'poincare':
            # grad riemanniano = ((1-||x||^2)^2 / 4) * grad euclidiano
            n2 = emb.pow(2).sum(-1, keepdim=True)
            emb -= lr * g * ((1 - n2).clamp_min(EPS).pow(2) / 4.0)
            # retracción a la bola
            nrm = emb.norm(dim=-1, keepdim=True)
            over = nrm >= 1 - 1e-5
            emb[over.squeeze(-1)] = (emb / nrm * (1 - 1e-5))[over.squeeze(-1)]
        elif model == 'lorentz':
            # proyectar grad al espacio tangente y hacer exp-map
            gl = g.clone(); gl[..., 0] = -gl[..., 0]           # subir índice
            tang = gl + lorentz_inner(emb, gl).unsqueeze(-1) * emb
            tang = -lr * tang
            nrm = torch.sqrt(lorentz_inner(tang, tang).clamp_min(EPS)).unsqueeze(-1)
            nrm = nrm.clamp(min=EPS, max=10.0)
            new = torch.cosh(nrm) * emb + torch.sinh(nrm) * tang / nrm
            # re-proyectar al hiperboloide
            new[..., 0] = torch.sqrt(1 + new[..., 1:].pow(2).sum(-1))
            emb.copy_(new)
        elif model == 'fisher_rao':
            # métrica del semiespacio: grad_R = sigma^2 * diag(1,..,1,1/2) * grad_E
            s2 = (2 * emb[..., -1]).exp().unsqueeze(-1)
            scale = torch.ones_like(emb); scale[..., -1] = 0.5
            emb -= lr * g * s2 * scale
            emb[..., -1].clamp_(min=math.log(1e-3), max=math.log(1e3))
        else:  # euclidean
            emb -= lr * g
        emb.grad = None

DIST = {'euclidean': d_euclidean, 'poincare': d_poincare,
        'lorentz': d_lorentz, 'fisher_rao': d_fisher_rao}

def init_emb(model, N, D, gen):
    if model == 'lorentz':
        x = torch.zeros(N, D)
        x[:, 1:] = torch.randn(N, D - 1, generator=gen) * 1e-3
        x[:, 0] = torch.sqrt(1 + x[:, 1:].pow(2).sum(-1))
    elif model == 'fisher_rao':
        x = torch.zeros(N, D)
        x[:, :-1] = torch.randn(N, D - 1, generator=gen) * 1e-3
        x[:, -1] = 0.0                      # log sigma = 0 -> sigma = 1
    else:
        x = torch.randn(N, D, generator=gen) * 1e-3
    return x.requires_grad_(True)

# ------------------------------------------------------------- entrenamiento
# Learning rate óptimo por brazo, hallado con barrido independiente (D=10, seed=1):
#   euclidean 3.0 | poincare 3.0 | lorentz 1.0 (3.0 diverge) | fisher_rao 3.0
# Se le da a cada método su mejor configuración para no handicapear a ninguno.
BEST_LR = {'euclidean': 3.0, 'poincare': 3.0, 'lorentz': 1.0, 'fisher_rao': 3.0}

def train(model, N, E, D, seed, epochs=400, n_neg=50, lr=None, burnin=10, bs=20):
    if lr is None:
        lr = BEST_LR[model]
    gen = torch.Generator().manual_seed(seed)
    emb = init_emb(model, N, D, gen)
    dist = DIST[model]
    n_edges = len(E)
    Et = torch.from_numpy(E)
    for ep in range(epochs):
        cur_lr = lr / 10 if ep < burnin else lr
        perm = torch.randperm(n_edges, generator=gen)
        for i in range(0, n_edges, bs):
            b = Et[perm[i:i + bs]]
            u, v = b[:, 0], b[:, 1]
            neg = torch.randint(0, N, (len(b), n_neg), generator=gen)
            U = emb[u].unsqueeze(1)                       # [B,1,D]
            cand = torch.cat([v.unsqueeze(1), neg], 1)    # [B,1+n_neg]
            Vc = emb[cand]                                # [B,1+n_neg,D]
            d = dist(U.expand_as(Vc), Vc)                 # [B,1+n_neg]
            loss = torch.nn.functional.cross_entropy(-d, torch.zeros(len(b), dtype=torch.long))
            loss.backward()
            rsgd_step(model, emb, cur_lr)
    return emb.detach()

# ------------------------------------------------------------- evaluación
@torch.no_grad()
def evaluate(model, emb, E, N):
    """MAP y Mean Rank de reconstrucción (protocolo Nickel & Kiela)."""
    dist = DIST[model]
    parents = defaultdict(set)
    for a, b in E:
        parents[a].add(b)
    ranks, aps = [], []
    for u, ps in parents.items():
        d = dist(emb[u].unsqueeze(0).expand(N, -1), emb)      # [N]
        d[u] = float('inf')
        order = torch.argsort(d)
        pos = torch.zeros(N, dtype=torch.bool); pos[list(ps)] = True
        ranked = pos[order]
        hit_positions = torch.nonzero(ranked).squeeze(-1) + 1  # 1-indexed
        if len(hit_positions) == 0: continue
        # rank de cada positivo descontando otros positivos mejor rankeados
        adj = hit_positions - torch.arange(len(hit_positions))
        ranks.extend(adj.tolist())
        prec = torch.arange(1, len(hit_positions) + 1).float() / hit_positions.float()
        aps.append(prec.mean().item())
    return float(np.mean(aps)), float(np.mean(ranks))

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    torch.set_num_threads(8)
    print("Construyendo subárbol de mamíferos de WordNet (cierre transitivo)...")
    idx, E = build_mammal_closure()
    N = len(idx)
    print(f"  nodos={N}  aristas(cierre transitivo)={len(E)}")
    print("  (referencia Nickel & Kiela 2017: 1180 nodos, 6540 aristas)\n")

    ARMS = ['euclidean', 'poincare', 'lorentz', 'fisher_rao']
    DIMS = [5, 10]
    SEEDS = [1, 2, 3, 4, 5]
    res = {D: {a: {'map': [], 'mr': []} for a in ARMS} for D in DIMS}

    for D in DIMS:
        print(f"===== dimensión por nodo D={D} (presupuesto igualado) =====")
        for seed in SEEDS:
            for a in ARMS:
                t0 = time.time()
                emb = train(a, N, E, D, seed)
                mp, mr = evaluate(a, emb, E, N)
                res[D][a]['map'].append(mp); res[D][a]['mr'].append(mr)
                print(f"  D={D} seed={seed} {a:<11}: MAP={mp:.4f}  MeanRank={mr:7.2f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 74)
    print(f"{'dim':>4} {'modelo':<12} {'MAP (media±sd)':>20} {'MeanRank':>18}")
    print("=" * 74)
    for D in DIMS:
        for a in ARMS:
            m = np.array(res[D][a]['map']); r = np.array(res[D][a]['mr'])
            print(f"{D:>4} {a:<12} {m.mean():>12.4f}±{m.std():.4f} {r.mean():>12.2f}±{r.std():.2f}")

    print("\nContrastes pareados (Fisher-Rao vs cada incumbente, MAP, n=5):")
    for D in DIMS:
        fr = np.array(res[D]['fisher_rao']['map'])
        for a in ['euclidean', 'poincare', 'lorentz']:
            b = np.array(res[D][a]['map'])
            t, p = stats.ttest_rel(fr, b)
            verdict = "FR gana" if fr.mean() > b.mean() else "FR pierde"
            print(f"  D={D}  FR vs {a:<10}: Δ={fr.mean()-b.mean():+.4f}  t={t:+.2f}  p={p:.4f}  -> {verdict}")

    with open("resultados_wordnet_incumbentes.json", "w") as f:
        json.dump({str(D): {a: res[D][a] for a in ARMS} for D in DIMS}, f, indent=1)
    print("\nGuardado: resultados_wordnet_incumbentes.json")
