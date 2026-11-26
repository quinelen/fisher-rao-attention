"""Resume el avance de la última corrida en dos JSON (para la página de réplica).

Uso: python3 progreso_mac.py   -> corridas/<ultima>/mac_progreso.json y mac_tres_geometrias.json
"""
import json, os, re, math

BASE = os.path.dirname(os.path.abspath(__file__))
VC = os.path.join(BASE, "verificacion_claude")
RUN = os.path.join(BASE, "corridas", os.path.basename(open(os.path.join(BASE, "corridas", "ULTIMA.txt")).read().strip()))

PASOS = [
    ("01_manifiesto_codigo", "Corpus de código Python", "§4.8d", None),
    ("02_manifiesto_man", "Corpus de páginas man", "§4.8e", None),
    ("10_tres_geometrias", "Tres geometrías (Dyck-2)", "§4.2–4.3", "resultados_tres_geometrias.json"),
    ("11_ricci_addendum", "Brazo Ricci–Mahalanobis", "§4.2", "resultados_ricci_addendum.json"),
    ("12_fast_gaussian_bench", "Velocidad del módulo", "§4.4", None),
    ("13_fast_gaussian_acc", "Kernel isotrópico (Dyck-2)", "§4.2", "resultados_fast_gaussian.json"),
    ("14_mixta", "Mezcla de curvaturas", "§5", "resultados_mixta.json"),
    ("15_mixta2", "Estrés de estructura mixta", "§5.1", "resultados_mixta2.json"),
    ("16_wasserstein_vs_fr", "Wasserstein vs Fisher–Rao", "§4.6", "resultados_wasserstein_vs_fr.json"),
    ("17_wordnet_incumbentes", "WordNet vs Poincaré/Lorentz", "§5.2", "resultados_wordnet_incumbentes.json"),
    ("18_eficiencia_dim", "Eficiencia por dimensión", "§4.7", "resultados_eficiencia_dimensional.json"),
    ("19_ablacion_kernel", "Ablaciones KL y Laplaciano", "§4.5–4.6", "resultados_ablacion_kernel.json"),
    ("20_cat_incumbente", "Frente a CAT", "§5.3", "resultados_cat_incumbente.json"),
    ("21_estabilizado", "Código con tope de μ", "§4.8d", "resultados_estabilizado.json"),
    ("22_prosa_man", "Prosa técnica (man)", "§4.8e", "resultados_prosa_man.json"),
    ("23_rope", "Posiciones relativas (RoPE)", "§4.8f", "resultados_rope.json"),
]

def hojas(x, r=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from hojas(v, f"{r}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from hojas(v, f"{r}[{i}]")
    else:
        yield r, x

def es_tiempo(k):
    k = k.lower()
    return any(t in k for t in ("secs", "time", "tiempo", "/s]", "_s", "ms"))

def comparar(nuevo, pub):
    a, b = dict(hojas(json.load(open(nuevo)))), dict(hojas(json.load(open(pub))))
    n = eq = 0; peor = 0.0
    for k in set(a) & set(b):
        va, vb = a[k], b[k]
        if isinstance(va, bool) or not isinstance(va, (int, float)) or not isinstance(vb, (int, float)) or es_tiempo(k):
            continue
        n += 1
        if va == vb or (isinstance(va, float) and math.isnan(va) and math.isnan(vb)):
            eq += 1
        else:
            peor = max(peor, abs(va - vb))
    return n, eq, peor

estado_txt = open(os.path.join(RUN, "estado.txt")).read()
inicio = os.path.getmtime(os.path.join(RUN, ".inicio"))
filas = []
for orden, (paso, nombre, sec, js) in enumerate(PASOS, 1):
    fin = re.search(rf"FIN\s+{paso}\s+rc=(\d+)\s+(\d+) min", estado_txt)
    empezo = f"INICIO {paso}" in estado_txt
    estado = "Pendiente"; minutos = None; comp = ""; iguales = None; total = None
    if fin:
        estado = "Terminado" if fin.group(1) == "0" else "Error"
        minutos = int(fin.group(2))
    elif empezo:
        estado = "En curso"
    log = os.path.join(RUN, "logs", paso + ".log")
    if estado in ("Terminado", "Error") and paso.startswith("0"):
        t = open(log).read()
        m = re.search(r"(archivos|páginas) que difieren: \[(.*?)\]", t, re.S)
        if "coincide con el corpus usado en el paper: True" in t:
            comp = "Igual al del paper"
        elif m:
            k = len([x for x in m.group(2).split(",") if x.strip()])
            comp = f"Distinto: {k} {m.group(1)} cambian"
    elif estado == "Terminado" and paso.endswith("bench"):
        t = open(log).read()
        ms = re.findall(r"^\s+(.+?)\s{2,}([\d.]+) ms/paso", t, re.M)
        if ms:
            d = {k.strip(): float(v) for k, v in ms}
            eu = next((v for k, v in d.items() if k.startswith("euclid")), None)
            iso = next((v for k, v in d.items() if k.startswith("isotrop")), None)
            if eu and iso:
                comp = f"Isotrópico {iso/eu:.1f}× euclidiano".replace(".", ",")
    elif estado == "Terminado" and js:
        nuevo = os.path.join(VC, js); pub = os.path.join(RUN, "resultados_publicados", js)
        if os.path.exists(nuevo) and os.path.getmtime(nuevo) > inicio and os.path.exists(pub):
            total, iguales, peor = comparar(nuevo, pub)
            comp = (f"Idéntico ({iguales}/{total} valores)" if iguales == total
                    else f"Difiere: {iguales}/{total} iguales, mayor diferencia {peor:.3g}")
        else:
            comp = "Sin JSON nuevo"
    filas.append({"orden": orden, "paso": paso, "experimento": nombre, "seccion": sec, "estado": estado,
                  "minutos": minutos, "comparacion": comp, "valores_iguales": iguales, "valores_total": total})

# Semillas del benchmark principal, comparadas con lo publicado
pub3 = json.load(open(os.path.join(RUN, "resultados_publicados", "resultados_tres_geometrias.json")))
brazos = {"euclidiana (K=0)": "Euclidiana", "esferica (K=+1/4)": "Esférica", "hiperbolica (K=-1/2)": "Hiperbólica anisotrópica"}
semillas = []
log3 = os.path.join(RUN, "logs", "10_tres_geometrias.log")
if os.path.exists(log3):
    for s, br, acc, loss in re.findall(r"seed (\d+) \| (.+?)\s*: IN acc=[\d.]+%\s+OUT acc=([\d.]+)%\s+OUT loss=([\d.]+)", open(log3).read()):
        s = int(s); p = pub3[br.strip()]
        pa = p["out_acc"][s - 1] * 100; pl = p["out_loss"][s - 1]
        igual = f"{pa:.1f}" == acc and f"{pl:.4f}" == loss
        semillas.append({"clave": f"{s}-{brazos[br.strip()]}", "semilla": s, "brazo": brazos[br.strip()],
                         "mac_out_acc": float(acc), "paper_out_acc": round(pa, 1),
                         "mac_out_loss": float(loss), "paper_out_loss": round(pl, 4),
                         "coincide": "Sí" if igual else "No"})

json.dump(filas, open(os.path.join(RUN, "mac_progreso.json"), "w"), ensure_ascii=False, indent=1)
json.dump(semillas, open(os.path.join(RUN, "mac_tres_geometrias.json"), "w"), ensure_ascii=False, indent=1)
print(RUN)
for f in filas: print(f["orden"], f["paso"], f["estado"], f["minutos"], f["comparacion"])
print(len(semillas), "semillas;", sum(1 for x in semillas if x["coincide"] == "Sí"), "coinciden")
