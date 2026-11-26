#!/bin/bash
# Seguimiento pre-registrado: extrapolación en código con el kernel original, corpus CPython 3.13.
# Protocolo y sello: corridas/seguimiento_py313/PREREGISTRO_PY313.md y SELLO_PREREGISTRO_PY313.txt
set -u
cd "$(dirname "$0")"
REPO="$(pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="$REPO/corridas/$STAMP-py313"
PRE="$REPO/corridas/seguimiento_py313"
mkdir -p "$RUN/logs"
echo "$RUN" > "$REPO/corridas/ULTIMA_PY313.txt"
ESTADO="$RUN/estado.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$ESTADO"; }
command -v caffeinate >/dev/null && caffeinate -i -w $$ &

# 0. Sello del pre-registro: verificar que el protocolo no cambió desde que se registró
ESPERADO="$(awk '{print $1}' "$PRE/SELLO_PREREGISTRO_PY313.txt")"
ACTUAL="$(shasum -a 256 "$PRE/PREREGISTRO_PY313.md" | awk '{print $1}')"
if [ "$ESPERADO" != "$ACTUAL" ]; then log "ERROR: el pre-registro cambió después de sellarse ($ACTUAL != $ESPERADO)"; exit 1; fi
cp "$PRE/PREREGISTRO_PY313.md" "$PRE/SELLO_PREREGISTRO_PY313.txt" "$RUN/"
log "Pre-registro verificado: $ACTUAL"

# 1. Esperar a que la corrida principal termine su benchmark de velocidad (para no contaminar esos tiempos)
CORE="$REPO/corridas/$(basename "$(cat "$REPO/corridas/ULTIMA.txt")")"
log "Esperando a que la corrida principal termine el benchmark de velocidad ($CORE)"
while ! grep -q "FIN    12_fast_gaussian_bench" "$CORE/estado.txt" 2>/dev/null \
      && ! grep -q "=== Terminado" "$CORE/estado.txt" 2>/dev/null \
      && pgrep -f ejecutar_mac.sh >/dev/null; do
  sleep 60
done
log "Listo para empezar"

# 2. Encontrar CPython 3.13
PY=""
for c in /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13 \
         /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
         "$HOME"/.pyenv/versions/3.13*/bin/python3.13 "$(command -v python3.13 2>/dev/null)"; do
  [ -n "$c" ] && [ -x "$c" ] && PY="$c" && break
done
if [ -z "$PY" ] && command -v uv >/dev/null; then
  log "No hay Python 3.13 instalado; lo instalo con uv (queda en la caché de uv)"
  uv python install 3.13 >> "$RUN/logs/00_instalacion.log" 2>&1
  PY="$(uv python find 3.13 2>/dev/null)"
fi
if [ -z "$PY" ] || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)'; then
  log "ERROR: no encontré CPython 3.13 (ni Homebrew, ni python.org, ni pyenv, ni uv). Instálalo y vuelve a lanzar."
  exit 1
fi
log "Intérprete: $PY ($("$PY" --version 2>&1))"

# 3. Entorno virtual propio y dependencias
VENV="$REPO/.venv313"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || { log "ERROR creando .venv313"; exit 1; }
log "Instalando dependencias"
"$VENV/bin/python" -m pip install --quiet --upgrade pip >> "$RUN/logs/00_instalacion.log" 2>&1
"$VENV/bin/python" -m pip install --quiet torch numpy scipy >> "$RUN/logs/00_instalacion.log" 2>&1 \
  || { log "ERROR instalando dependencias (ver logs/00_instalacion.log)"; exit 1; }
"$VENV/bin/python" -c "import platform, torch, numpy, scipy; print('Entorno:', platform.platform(), '| Python', platform.python_version(), '| torch', torch.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)" | tee -a "$ESTADO"

# 4. Copia de trabajo de los scripts (no toca los resultados publicados del repo)
cp -R verificacion_claude "$RUN/codigo"
cd "$RUN/codigo"

# 5. Manifiesto del corpus, antes de entrenar
"$VENV/bin/python" - > "$RUN/logs/01_manifiesto.log" 2>&1 <<'PYEOF'
import json, hashlib, platform, sys, glob, os, sysconfig
sys.path.insert(0, os.getcwd())
from exp_codigo_python import build_corpus, make_vocab
tr, te = build_corpus(); _, vocab = make_vocab(tr)
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
m = {"python": platform.python_version(), "implementation": platform.python_implementation(),
     "executable": sys.executable, "stdlib": sysconfig.get_paths()["stdlib"],
     "train_chars": len(tr), "test_chars": len(te), "vocab_size": vocab,
     "train_sha256": sha(tr), "test_sha256": sha(te),
     "colab_observado": {"train_chars": 1130080, "test_chars": 125909, "vocab_size": 111}}
json.dump(m, open("../manifiesto_corpus_py313.json", "w"), indent=1)
print(json.dumps(m, indent=1))
PYEOF
log "Corpus: $(grep -E '"(train_chars|test_chars|vocab_size)"' "$RUN/manifiesto_corpus_py313.json" | head -3 | tr -d ' \n')"

# 6. Fase A (descriptiva, semillas 1–5) y fase B (confirmatoria, semillas 6–15)
log "INICIO fase A (semillas 1–5)"
"$VENV/bin/python" -u exp_codigo_extrapolativo.py > "$RUN/logs/A_semillas_1_5.log" 2>&1; log "FIN    fase A rc=$?"
log "INICIO fase B (semillas 6–15, endpoint pre-registrado)"
"$VENV/bin/python" -u exp_codigo_extrapolativo_rep.py > "$RUN/logs/B_semillas_6_15.log" 2>&1; log "FIN    fase B rc=$?"

# 7. Veredicto según el pre-registro
"$VENV/bin/python" - <<'PYEOF' | tee "$RUN/veredicto.txt"
import json, numpy as np
from scipy import stats
A = json.load(open("resultados_codigo_extrapolativo.json")); B = json.load(open("resultados_codigo_extrapolativo_rep.json"))
def prueba(res, k):
    e = np.array([r[k] for r in res["euclidean"]]); f = np.array([r[k] for r in res["fisher_rao"]])
    t, p = stats.ttest_rel(f, e)
    return dict(n=len(e), euc=e.mean(), fr=f.mean(), delta=f.mean()-e.mean(), t=t, p=p,
                ganadas=int((f < e).sum()), sd_ratio=f.std(ddof=1)/e.std(ddof=1))
out = {}
for fase, res in (("A_semillas_1_5", A), ("B_semillas_6_15", B)):
    out[fase] = {k: prueba(res, k) for k in ("bpc_extra_L512", "bpc_extra_L256", "bpc_L128")}
AB = {a: A[a] + B[a] for a in ("euclidean", "fisher_rao")}
out["AB_n15_post_hoc"] = {"bpc_extra_L512": prueba(AB, "bpc_extra_L512")}
pb = out["B_semillas_6_15"]["bpc_extra_L512"]
out["endpoint_primario_cumplido"] = bool(pb["delta"] < 0 and pb["p"] < 0.05)
out["estabilidad_cumplida"] = bool(pb["sd_ratio"] <= 1.5)
json.dump(out, open("../veredicto_py313.json", "w"), indent=1, default=float)
for fase in ("A_semillas_1_5", "B_semillas_6_15"):
    for k, v in out[fase].items():
        print(f"{fase:16} {k:15} Δ={v['delta']:+.4f}  p={v['p']:.4f}  FR gana {v['ganadas']}/{v['n']}  dispersión {v['sd_ratio']:.2f}×")
v = out["AB_n15_post_hoc"]["bpc_extra_L512"]
print(f"post hoc n=15    bpc_extra_L512  Δ={v['delta']:+.4f}  p={v['p']:.4f}")
print("ENDPOINT PRIMARIO:", "CUMPLIDO" if out["endpoint_primario_cumplido"] else "NO CUMPLIDO",
      "| estabilidad:", "cumplida" if out["estabilidad_cumplida"] else "no cumplida")
PYEOF
log "=== Terminado. Resultados en $RUN ==="
