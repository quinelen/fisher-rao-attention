# Figuras de geometría del paper (geometrías de curvatura constante, semiespacio hiperbólico,
# tokens reales) y escena interactiva (geometrias_interactivas.html). El idioma solo cambia el
# texto (EN: paper; ES: traducción, con coma decimal).
# La figura de tokens reales usa las distribuciones reales guardadas en tokens_esfera_qp.npy / tokens_esfera_ids.npy.
# Uso (desde verificacion_claude/): python figuras_geometria.py en . .
import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

lang, out, src = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
T = {
 'en': dict(
   p1='SPHERICAL  K = +1/4\ncategorical Fisher–Rao: octant of a sphere of radius 2\n(φ(p)=2√p; geodesic distance = 2·arccos Σ√(pᵢqᵢ))',
   p2='EUCLIDEAN  K = 0\nlocal flat regime (fixed σ, ‖Δμ‖ ≪ σ)\ngeodesic = straight line',
   p3='HYPERBOLIC  K < 0\npseudosphere: constant negative curvature (local model)\nexponential volume growth → low-distortion hierarchies',
   sup='The three constant-curvature geometries of the Fisher–Rao kernel: one distributional family → one geometry (Chentsov)',
   b_title="The kernel's manifold: hyperbolic half-space H³ (K = −1/2)\n"
           'd = √2·arccosh(1 + (‖Δμ‖²/2 + Δσ²)/(2σᵢσⱼ)) — the boundary σ→0 is at infinite distance',
   b_spec='specific tokens (small σ → near the boundary)', b_gen='generic tokens (large σ → interior)',
   b_geo='Fisher–Rao geodesics (arcs ⊥ to the boundary)', b_z='σ  (generality radius)',
   c_title='REAL tokens of a trained model, on their exact manifold\n'
           'Spherical Fisher–Rao attention, d_head = 3: each token is a distribution on Δ²,\n'
           'drawn on the octant via the isometry φ(p)=2√p (gray line = sequence order)',
   h_t1='SPHERICAL K=+1/4 — real trained tokens', h_t2='HYPERBOLIC K<0 — pseudosphere (local model)',
   h_t3='Half-space H³ of the kernel — geodesics', h_seq='sequence', h_spec='specific (σ↓)', h_gen='generic (σ↑)',
   h_title='Fisher–Rao Geodesic Attention — spherical and hyperbolic manifolds of the kernel (drag to rotate; double-click to reset)'),
 'es': dict(
   p1='ESFÉRICA  K = +1/4\nFisher–Rao categórica: octante de esfera de radio 2\n(φ(p)=2√p; distancia geodésica = 2·arccos Σ√(pᵢqᵢ))',
   p2='EUCLIDIANA  K = 0\nrégimen plano local (σ fijo, ‖Δμ‖ ≪ σ)\ngeodésica = línea recta',
   p3='HIPERBÓLICA  K < 0\npseudoesfera: curvatura negativa constante (modelo local)\nvolumen exponencial → jerarquías con baja distorsión',
   sup='Las tres geometrías de curvatura constante del núcleo de Fisher–Rao: una familia de distribuciones → una geometría (Chentsov)',
   b_title='La variedad del núcleo: semiespacio hiperbólico H³ (K = −1/2)\n'
           'd = √2·arccosh(1 + (‖Δμ‖²/2 + Δσ²)/(2σᵢσⱼ)) — el borde σ→0 está a distancia infinita',
   b_spec='tokens específicos (σ pequeño → borde)', b_gen='tokens genéricos (σ grande → interior)',
   b_geo='geodésicas Fisher–Rao (arcos ⊥ al borde)', b_z='σ  (radio de generalidad)',
   c_title='Tokens REALES de un modelo entrenado, sobre su variedad exacta\n'
           'Atención esférica Fisher–Rao, d_head=3: cada token es una distribución en Δ²,\n'
           'dibujada en el octante vía la isometría φ(p)=2√p (línea gris = orden de la secuencia)',
   h_t1='ESFÉRICA K=+1/4 — tokens reales entrenados', h_t2='HIPERBÓLICA K<0 — pseudoesfera (modelo local)',
   h_t3='SEMIESPACIO H³ del núcleo — geodésicas', h_seq='secuencia', h_spec='específicos (σ↓)', h_gen='genéricos (σ↑)',
   h_title='Fisher–Rao Geodesic Attention — variedades esférica e hiperbólica del núcleo (arrastra para rotar; doble clic para restablecer)'),
}[lang]

if lang == 'es':  # coma decimal en las marcas de los ejes
    import matplotlib.ticker as mticker
    class _Coma(mticker.ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace('.', ',')
    _savefig = plt.savefig
    def _savefig_es(*a, **k):
        for _ax in plt.gcf().axes:
            for _axis in (_ax.xaxis, _ax.yaxis, getattr(_ax, 'zaxis', None)):
                if _axis is not None: _axis.set_major_formatter(_Coma())
        return _savefig(*a, **k)
    plt.savefig = _savefig_es

# ===================== Figura 1: tres geometrías =====================
fig = plt.figure(figsize=(16, 5.5))
ax = fig.add_subplot(131, projection='3d')
u = np.linspace(0, np.pi/2, 40); v = np.linspace(0, np.pi/2, 40)
U, V = np.meshgrid(u, v); R = 2.0
X = R*np.sin(U)*np.cos(V); Y = R*np.sin(U)*np.sin(V); Z = R*np.cos(U)
ax.plot_surface(X, Y, Z, alpha=0.35, cmap='Blues', edgecolor='steelblue', linewidth=0.15)
p = np.array([0.10, 0.10, 0.80]); q = np.array([0.10, 0.80, 0.10])
a, b = np.sqrt(p), np.sqrt(q)
ang = np.arccos(np.clip(a@b, -1, 1))
t = np.linspace(0, 1, 60)
arc = np.array([(np.sin((1-ti)*ang)*a + np.sin(ti*ang)*b)/np.sin(ang) for ti in t]) * R
ax.plot(arc[:,0], arc[:,1], arc[:,2], color='crimson', lw=3, zorder=5)
for pt, lab in [(2*a, 'p'), (2*b, 'q')]:
    ax.scatter(*pt, color='crimson', s=60, zorder=6)
    ax.text(pt[0]*1.1, pt[1]*1.1, pt[2]*1.05, lab, fontsize=13, color='crimson')
ax.set_title(T['p1'], fontsize=10)
ax.set_box_aspect([1,1,1]); ax.view_init(20, 35)

ax = fig.add_subplot(132, projection='3d')
g = np.linspace(-2, 2, 20)
Xg, Yg = np.meshgrid(g, g)
ax.plot_surface(Xg, Yg, np.zeros_like(Xg), alpha=0.35, cmap='Greys', edgecolor='gray', linewidth=0.15)
ax.plot([-1.5, 1.4], [-1.2, 1.3], [0, 0], color='crimson', lw=3)
ax.scatter([-1.5, 1.4], [-1.2, 1.3], [0, 0], color='crimson', s=60)
ax.text(-1.5, -1.2, 0.15, 'p', fontsize=13, color='crimson'); ax.text(1.4, 1.3, 0.15, 'q', fontsize=13, color='crimson')
ax.set_title(T['p2'], fontsize=10)
ax.set_box_aspect([1,1,0.5]); ax.view_init(28, -60); ax.set_zlim(-1, 1)

ax = fig.add_subplot(133, projection='3d')
u = np.linspace(0.35, 3.2, 60); v = np.linspace(0, 2*np.pi, 60)
U, V = np.meshgrid(u, v)
Xp = 1/np.cosh(U)*np.cos(V); Yp = 1/np.cosh(U)*np.sin(V); Zp = U - np.tanh(U)
ax.plot_surface(Xp, Yp, Zp, alpha=0.4, cmap='Reds', edgecolor='indianred', linewidth=0.1)
ax.plot_surface(Xp, Yp, -Zp, alpha=0.4, cmap='Reds', edgecolor='indianred', linewidth=0.1)
uu = np.linspace(0.4, 2.8, 60); vv = np.full_like(uu, np.radians(-30))  # meridiano (tractriz): geodésica de la pseudoesfera
ax.plot(1/np.cosh(uu)*np.cos(vv), 1/np.cosh(uu)*np.sin(vv), uu-np.tanh(uu), color='darkred', lw=3)
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.set_title(T['p3'], fontsize=10)
ax.set_box_aspect([1,1,1.2]); ax.view_init(15, 40)
plt.suptitle(T['sup'], fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(out, 'fig_geometrias_3d.png'), dpi=170, bbox_inches='tight')
plt.close(fig)

# ===================== Figura 3: semiespacio =====================
fig2 = plt.figure(figsize=(11, 8.6))
ax = fig2.add_subplot(111, projection='3d')
g = np.linspace(-4, 4, 12)
Xg, Yg = np.meshgrid(g, g)
ax.plot_surface(Xg, Yg, np.zeros_like(Xg), alpha=0.15, color='gray')

def geodesic(mu1, s1, mu2, s2, n=60):
    mu1, mu2 = np.array(mu1, float), np.array(mu2, float)
    d = mu2 - mu1; L = np.linalg.norm(d)
    e = d/L if L > 1e-9 else np.array([1.0, 0.0])
    t1, t2 = 0.0, L
    s1_, s2_ = np.sqrt(2)*s1, np.sqrt(2)*s2
    if L < 1e-9:
        ss = np.linspace(s1_, s2_, n)
        return np.array([[*mu1, s/np.sqrt(2)] for s in ss])
    c = (t2**2 + s2_**2 - t1**2 - s1_**2) / (2*(t2 - t1))
    r = np.sqrt((t1-c)**2 + s1_**2)
    th1, th2 = np.arctan2(s1_, t1-c), np.arctan2(s2_, t2-c)
    th = np.linspace(th1, th2, n)
    tt, ss = c + r*np.cos(th), r*np.sin(th)
    return np.array([[*(mu1 + ti*e), si/np.sqrt(2)] for ti, si in zip(tt, ss)])

rng = np.random.RandomState(3)
espec = [(rng.uniform(-3.5,3.5,2), rng.uniform(0.15,0.45)) for _ in range(9)]
gener = [(rng.uniform(-1.5,1.5,2), rng.uniform(1.8,3.0)) for _ in range(4)]
for mu, s in espec: ax.scatter(*mu, s, color='crimson', s=45)
for mu, s in gener: ax.scatter(*mu, s, color='royalblue', s=90)
pares = [(espec[0], gener[0]), (espec[3], espec[6]), (gener[1], gener[3]), (espec[8], gener[2])]
for (m1,s1),(m2,s2) in pares:
    gpts = geodesic(m1, s1, m2, s2)
    ax.plot(gpts[:,0], gpts[:,1], gpts[:,2], color='darkgreen', lw=2, alpha=0.8)
ax.scatter([],[],[], color='crimson', s=45, label=T['b_spec'])
ax.scatter([],[],[], color='royalblue', s=90, label=T['b_gen'])
ax.plot([],[], color='darkgreen', lw=2, label=T['b_geo'])
ax.set_xlabel('μ₁'); ax.set_ylabel('μ₂'); ax.set_zlabel(T['b_z'], labelpad=10)
ax.set_title(T['b_title'], fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.view_init(18, -50); ax.set_zlim(0, 3.4)
ax.set_position([0.0, 0.02, 0.86, 0.86])
_f3 = os.path.join(out, 'fig_semiespacio_hiperbolico.png')
plt.savefig(_f3, dpi=170)
plt.close(fig2)
from PIL import Image  # recorte al contenido con margen blanco
_im = Image.open(_f3); _ys, _xs = np.where(np.asarray(_im.convert('L')) < 250)
_im.crop((max(_xs.min()-25, 0), max(_ys.min()-25, 0), min(_xs.max()+25, _im.size[0]), min(_ys.max()+25, _im.size[1]))).save(_f3, dpi=(170, 170))

# ===================== Figura 2: tokens reales (datos guardados) =====================
qp = np.load(os.path.join(src, 'tokens_esfera_qp.npy')); toks = np.load(os.path.join(src, 'tokens_esfera_ids.npy'))
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, np.pi/2, 40); v = np.linspace(0, np.pi/2, 40)
U, V = np.meshgrid(u, v); R = 2.0
ax.plot_surface(R*np.sin(U)*np.cos(V), R*np.sin(U)*np.sin(V), R*np.cos(U),
                alpha=0.18, cmap='Blues', edgecolor='steelblue', linewidth=0.1)
pts = 2*np.sqrt(qp)
colors = {1:'#C44E52', 2:'#F2A65A', 3:'#4C72B0', 4:'#55A868'}
names = {1:'(', 2:')', 3:'[', 4:']'}
ax.plot(pts[:,0], pts[:,1], pts[:,2], color='gray', lw=0.7, alpha=0.6, zorder=3)
for tk in [1,2,3,4]:
    m = toks == tk
    ax.scatter(pts[m,0], pts[m,1], pts[m,2], color=colors[tk], s=55, label=f"token {names[tk]}", zorder=5)
ax.set_title(T['c_title'], fontsize=10)
ax.legend(loc='upper left'); ax.set_box_aspect([1,1,1]); ax.view_init(22, 40)
plt.tight_layout(); plt.savefig(os.path.join(out, 'fig_tokens_reales_esfera.png'), dpi=170, bbox_inches='tight')
plt.close(fig)

# ===================== Escena interactiva =====================
import plotly.graph_objects as go
from plotly.subplots import make_subplots
figh = make_subplots(rows=1, cols=3, specs=[[{'type':'scene'}]*3], subplot_titles=(T['h_t1'], T['h_t2'], T['h_t3']))
u = np.linspace(0, np.pi/2, 35); v = np.linspace(0, np.pi/2, 35)
U, V = np.meshgrid(u, v); R = 2.0
figh.add_trace(go.Surface(x=R*np.sin(U)*np.cos(V), y=R*np.sin(U)*np.sin(V), z=R*np.cos(U), opacity=0.3, colorscale='Blues', showscale=False), 1, 1)
figh.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines', line=dict(color='gray', width=2), opacity=0.5, name=T['h_seq'], showlegend=False), 1, 1)
for tk in [1,2,3,4]:
    m = toks == tk
    figh.add_trace(go.Scatter3d(x=pts[m,0], y=pts[m,1], z=pts[m,2], mode='markers', marker=dict(size=5, color=colors[tk]), name=f'token {names[tk]}'), 1, 1)
u = np.linspace(0.35, 3.2, 50); v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)
for sign in (1, -1):
    figh.add_trace(go.Surface(x=1/np.cosh(U)*np.cos(V), y=1/np.cosh(U)*np.sin(V), z=sign*(U-np.tanh(U)), opacity=0.45, colorscale='Reds', showscale=False), 1, 2)
g = np.linspace(-4, 4, 8)
Xg, Yg = np.meshgrid(g, g)
figh.add_trace(go.Surface(x=Xg, y=Yg, z=np.zeros_like(Xg), opacity=0.12, colorscale=[[0,'gray'],[1,'gray']], showscale=False), 1, 3)
def geodesic_h(mu1, s1, mu2, s2, n=50):
    mu1, mu2 = np.array(mu1,float), np.array(mu2,float)
    d = mu2-mu1; L = np.linalg.norm(d)
    e = d/L if L>1e-9 else np.array([1.,0.])
    s1_, s2_ = np.sqrt(2)*s1, np.sqrt(2)*s2
    if L<1e-9:
        ss=np.linspace(s1_,s2_,n); return np.array([[*mu1, s/np.sqrt(2)] for s in ss])
    c=(L**2+s2_**2-s1_**2)/(2*L); r=np.sqrt(c**2+s1_**2)
    th=np.linspace(np.arctan2(s1_,-c), np.arctan2(s2_,L-c), n)
    return np.array([[*(mu1+(c+r*np.cos(t))*e), r*np.sin(t)/np.sqrt(2)] for t in th])
rng = np.random.RandomState(3)
espec = [(rng.uniform(-3.5,3.5,2), rng.uniform(0.15,0.45)) for _ in range(9)]
gener = [(rng.uniform(-1.5,1.5,2), rng.uniform(1.8,3.0)) for _ in range(4)]
figh.add_trace(go.Scatter3d(x=[m[0] for m,_ in espec], y=[m[1] for m,_ in espec], z=[s for _,s in espec], mode='markers', marker=dict(size=4, color='#C44E52'), name=T['h_spec']), 1, 3)
figh.add_trace(go.Scatter3d(x=[m[0] for m,_ in gener], y=[m[1] for m,_ in gener], z=[s for _,s in gener], mode='markers', marker=dict(size=7, color='#4C72B0'), name=T['h_gen']), 1, 3)
for (m1,s1),(m2,s2) in [(espec[0],gener[0]),(espec[3],espec[6]),(gener[1],gener[3]),(espec[8],gener[2])]:
    gp = geodesic_h(m1,s1,m2,s2)
    figh.add_trace(go.Scatter3d(x=gp[:,0], y=gp[:,1], z=gp[:,2], mode='lines', line=dict(color='darkgreen', width=4), showlegend=False), 1, 3)
figh.update_layout(title=dict(text=T['h_title'], x=0.5), height=560, margin=dict(l=0,r=0,t=90,b=0), legend=dict(orientation='h', y=-0.05))
for s in ('scene','scene2','scene3'):
    figh.update_layout({s: dict(aspectmode='cube', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)})
rotate_js = """
var gd = document.querySelectorAll('.plotly-graph-div')[0];
var angle = 0; var paused = false;
gd.addEventListener('mousedown', ()=>{paused=true;});
setInterval(function(){
  if(paused) return;
  angle += 0.004;
  var eye = {x: 1.6*Math.cos(angle), y: 1.6*Math.sin(angle), z: 0.9};
  Plotly.relayout(gd, {'scene.camera.eye': eye, 'scene2.camera.eye': eye, 'scene3.camera.eye': eye});
}, 50);
"""
figh.write_html(os.path.join(out, 'geometrias_interactivas.html'), include_plotlyjs='inline', post_script=rotate_js)
print(lang, 'OK ->', out)
