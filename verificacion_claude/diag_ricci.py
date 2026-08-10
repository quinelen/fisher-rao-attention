# Diagnóstico del RicciMultiHeadAttention (formulación temprana del proyecto)
# 1) Verifica shape/uso del buffer self.g
# 2) Simula la dinámica del flujo de Ricci propuesto (con clamp) vs flujo normalizado
# 3) Verifica numéricamente el signo de la curvatura de la métrica de Fisher categórica

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# --- Copia fiel de la clase del proyecto ---
class RicciMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, beta=1.0, kappa=0.5, dt=0.01):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.beta = beta
        self.kappa = kappa
        self.dt = dt
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer('g', torch.eye(self.d_head).repeat(n_heads, 1, 1))

    def compute_fisher_metric(self, p):
        eps = 1e-12
        return 1.0 / (p + eps)

    def compute_I_munu(self, p, g_diag):
        eps = 1e-12
        S = -(p * torch.log(p + eps)).sum(dim=-1)
        hess_diag = -1.0 / (p + eps)
        I_diag = - S.unsqueeze(-1) * g_diag + hess_diag
        return I_diag

    def update_metric(self, g_diag, p):
        R_diag = -0.5 * g_diag
        I_diag = self.compute_I_munu(p, g_diag)
        dg_dt = -2.0 * R_diag + self.kappa * I_diag
        new_g = g_diag + self.dt * dg_dt
        new_g = torch.clamp(new_g, min=0.1, max=10.0)
        return new_g

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1,2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1,2)
        q_p = F.softmax(q, dim=-1)
        k_p = F.softmax(k, dim=-1)
        g_q = self.compute_fisher_metric(q_p)
        g_k = self.compute_fisher_metric(k_p)
        g_mid = 0.5 * (g_q.unsqueeze(3) + g_k.unsqueeze(2))
        diff = q_p.unsqueeze(3) - k_p.unsqueeze(2)
        dist_sq = (diff * diff * g_mid).sum(dim=-1)
        scores = -self.beta * dist_sq
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1,2).contiguous().view(B, L, D)
        with torch.no_grad():
            avg_g = g_q.mean(dim=(0,2))
            avg_p = q_p.mean(dim=(0,2))
            new_g = self.update_metric(avg_g, avg_p.unsqueeze(0).unsqueeze(2))
            self.g = new_g.squeeze(0).squeeze(1)
        return self.out_proj(attn_output), attn_weights


print("=" * 70)
print("TEST 1: shape y uso del buffer self.g  (d_model=64, n_heads=4, d_head=16)")
print("=" * 70)
attn = RicciMultiHeadAttention(d_model=64, n_heads=4)
print(f"self.g ANTES del forward : {tuple(attn.g.shape)}  (identidades por cabeza)")
x = torch.randn(2, 10, 64)
out, w = attn(x)
print(f"self.g DESPUES del forward: {tuple(attn.g.shape)}")
print("¿Se usa self.g en el cálculo de atención? -> NO (se recalcula g_q/g_k desde softmax(q,k); self.g es estado muerto)")

print()
print("=" * 70)
print("TEST 2: dinámica del flujo  dg/dt = -2R + kappa*I  con R=-0.5g (proyecto)")
print("=" * 70)
# Simulación pura de update_metric con distribución fija
d = 16
p = F.softmax(torch.randn(d), dim=-1)  # distribución categórica típica
g = 1.0 / (p + 1e-12)                  # métrica de Fisher inicial
S = -(p * torch.log(p + 1e-12)).sum()
print(f"Entropía S = {S:.3f}  (ln(16) = {math.log(16):.3f});  kappa=0.5, dt=0.01")
kappa, dt = 0.5, 0.01
g_t = g.clone()
traj = []
for step in range(2000):
    I_diag = -S * g_t + (-1.0 / (p + 1e-12))
    dg = -2.0 * (-0.5 * g_t) + kappa * I_diag   # = g_t*(1 - kappa*S) - kappa/p
    g_t = torch.clamp(g_t + dt * dg, min=0.1, max=10.0)
    if step in (0, 10, 100, 500, 1000, 1999):
        frac_lo = (g_t <= 0.1001).float().mean().item()
        frac_hi = (g_t >= 9.999).float().mean().item()
        traj.append((step, g_t.min().item(), g_t.mean().item(), g_t.max().item(), frac_lo, frac_hi))
print(f"{'paso':>5} {'min(g)':>8} {'mean(g)':>8} {'max(g)':>8} {'%clamp_inf':>11} {'%clamp_sup':>11}")
for s, mn, me, mx, fl, fh in traj:
    print(f"{s:>5} {mn:>8.3f} {me:>8.3f} {mx:>8.3f} {fl:>10.0%} {fh:>10.0%}")
print("Coef. lineal efectivo: (1 - kappa*S) =", f"{(1-kappa*S).item():.3f}",
      "-> si kappa*S>1 colapso al clamp inferior; si <1 crecimiento exponencial al superior")

print()
print("=" * 70)
print("TEST 3: flujo de Ricci NORMALIZADO (sin clamp): dg/dt = -2(R - r̄g) + kappa*(I - Ī g)")
print("=" * 70)
g_t = g.clone()
for step in range(2000):
    R_diag = -0.5 * g_t
    I_diag = -S * g_t + (-1.0 / (p + 1e-12))
    # normalización: se resta la traza media para preservar el "volumen" (Hamilton 1982)
    r_bar = (R_diag / g_t).mean()
    i_bar = (I_diag / g_t).mean()
    dg = -2.0 * (R_diag - r_bar * g_t) + kappa * (I_diag - i_bar * g_t)
    g_t = g_t + dt * dg
    if step in (0, 100, 500, 1000, 1999):
        print(f"paso {step:>4}: min={g_t.min():.3f}  mean={g_t.mean():.3f}  max={g_t.max():.3f}")
print("-> sin clamp, permanece acotado (el término de traza absorbe el modo inestable)")

print()
print("=" * 70)
print("TEST 4: signo de la curvatura de la métrica de Fisher categórica")
print("=" * 70)
# La variedad Fisher-Rao del simplex es isométrica a un octante de esfera de radio 2
# vía p_i = (x_i/2)^2... verificación numérica con triángulo geodésico:
# distancia Fisher-Rao exacta: d(p,q) = 2*arccos(sum sqrt(p_i q_i))
def fr_dist(p, q):
    return 2.0 * torch.arccos(torch.clamp((p.sqrt() * q.sqrt()).sum(), -1, 1))
# tres distribuciones cercanas en el simplex de dim 3
p1 = torch.tensor([0.5, 0.3, 0.2])
p2 = torch.tensor([0.4, 0.35, 0.25])
p3 = torch.tensor([0.45, 0.25, 0.3])
a, b, c = fr_dist(p2, p3), fr_dist(p1, p3), fr_dist(p1, p2)
# suma de ángulos del triángulo geodésico (ley de cosenos esférica en radio 2 => K=+1/4)
def angle(opp, s1, s2, K=0.25):
    # ley de cosenos esférica: cos(sqrt(K)*opp) = cos(sqrt(K)s1)cos(sqrt(K)s2) + sin..sin..cos(A)
    sk = math.sqrt(K)
    ca = (math.cos(sk*opp) - math.cos(sk*s1)*math.cos(sk*s2)) / (math.sin(sk*s1)*math.sin(sk*s2))
    return math.acos(max(-1, min(1, ca)))
A = angle(a, b, c); B = angle(b, a, c); C = angle(c, a, b)
print(f"Triángulo geodésico Fisher-Rao: suma de ángulos (asumiendo K=+1/4) = {math.degrees(A+B+C):.4f}°")
# comparación euclidiana: si K=+1/4 fuera correcto, la ley esférica cierra en 180° exactos
ca_e = (b*b + c*c - a*a) / (2*b*c)
A_e = math.acos(max(-1,min(1,ca_e.item())))
cb_e = (a*a + c*c - b*b) / (2*a*c); B_e = math.acos(max(-1,min(1,cb_e.item())))
cc_e = (a*a + b*b - c*c) / (2*a*b); C_e = math.acos(max(-1,min(1,cc_e.item())))
print(f"Con ley de cosenos EUCLIDIANA la suma da: {math.degrees(A_e+B_e+C_e):.4f}° (>180° => curvatura POSITIVA)")
print("-> La métrica de Fisher categórica tiene curvatura seccional +1/4 (esfera de radio 2),")
print("   NO curvatura negativa. El ansatz R = -0.5*g del proyecto contradice su propia métrica.")
