#!/bin/bash
# Lanzador de la batería de experimentos de fisher-rao-attention v1.1 en macOS (CPU nativo).
#
# Uso:   ./ejecutar_mac.sh rapido      # prueba de humo, ~15 min
#        ./ejecutar_mac.sh core        # bloque "Reproducing the benchmark" del README, ~5 h en laptop
#        ./ejecutar_mac.sh extendido   # core + evidencia extendida §4.8 (duración larga, sin estimar)
#
# Cada corrida queda en corridas/<fecha-hora>/ con:
#   estado.txt                inicio/fin de cada experimento, código de salida y minutos
#   logs/<experimento>.log    salida completa de cada script
#   resultados_publicados/    copia de los JSON/TXT publicados ANTES de correr
#   resultados_nuevos/        JSON que generó esta corrida
#   comparacion.txt           qué JSON salieron idénticos byte a byte a los publicados
# Los scripts sobrescriben los resultados_*.json en verificacion_claude/; los originales
# quedan en resultados_publicados/ y en git (git checkout -- verificacion_claude los restaura).

set -u
cd "$(dirname "$0")"
REPO="$(pwd)"
MODO="${1:-core}"
case "$MODO" in rapido|core|extendido) ;; *) echo "Modo desconocido: $MODO (usa rapido | core | extendido)"; exit 2;; esac

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="$REPO/corridas/$STAMP-$MODO"
mkdir -p "$RUN/logs" "$RUN/resultados_publicados" "$RUN/resultados_nuevos"
echo "$RUN" > "$REPO/corridas/ULTIMA.txt"
ESTADO="$RUN/estado.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$ESTADO"; }

# Mantener el Mac despierto mientras dure la corrida
command -v caffeinate >/dev/null && caffeinate -i -w $$ &

# Respaldo de lo publicado (y marca de inicio para detectar qué JSON se regeneran)
touch "$RUN/.inicio"
cp verificacion_claude/resultados_*.json verificacion_claude/resultados_*.txt "$RUN/resultados_publicados/" 2>/dev/null

# Python: preferir el CPython 3.9 del sistema (el manifiesto del corpus de código se generó con 3.9.6)
if [ -x /usr/bin/python3 ] && /usr/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,9) else 1)' 2>/dev/null; then
  PY=/usr/bin/python3
else
  PY="$(command -v python3)"
fi
VENV="$REPO/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  log "Creando entorno virtual con $PY ($("$PY" --version 2>&1))"
  "$PY" -m venv "$VENV" || { log "ERROR creando venv"; exit 1; }
fi
log "Instalando dependencias"
"$VENV/bin/python" -m pip install --quiet --upgrade pip >> "$RUN/logs/00_instalacion.log" 2>&1
"$VENV/bin/python" -m pip install --quiet -r requirements.txt nltk >> "$RUN/logs/00_instalacion.log" 2>&1 \
  || { log "ERROR instalando dependencias (ver logs/00_instalacion.log)"; exit 1; }
"$VENV/bin/python" -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)" >> "$RUN/logs/00_instalacion.log" 2>&1
"$VENV/bin/python" - <<PYEOF | tee -a "$ESTADO"
import platform, torch, numpy, scipy, os
print("Entorno:", platform.platform(), "| Python", platform.python_version(),
      "| torch", torch.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| CPUs", os.cpu_count())
PYEOF

run() {  # run <nombre> <script.py> [args...]
  local nombre="$1"; shift
  log "INICIO $nombre"
  local t0=$(date +%s)
  ( cd verificacion_claude && "$VENV/bin/python" -u "$@" ) > "$RUN/logs/$nombre.log" 2>&1
  local rc=$?
  log "FIN    $nombre  rc=$rc  $(( ($(date +%s)-t0)/60 )) min"
}

log "=== Corrida $MODO ==="
# Verificación de corpus (exactitud de §4.8d y §4.8e depende de Python 3.9.6 y de las páginas man del sistema)
run 01_manifiesto_codigo manifiesto_corpus_codigo.py
run 02_manifiesto_man    manifiesto_corpus_man.py

if [ "$MODO" = "rapido" ]; then
  run 10_fast_gaussian_bench fast_gaussian_test.py bench
  run 11_mecanismo_umbral    exp_mecanismo_umbral.py
  run 12_wasserstein_vs_fr   exp_wasserstein_vs_fisherrao.py
else
  run 10_tres_geometrias     exp_tres_geometrias.py
  run 11_ricci_addendum      exp_ricci_addendum.py
  run 12_fast_gaussian_bench fast_gaussian_test.py bench
  run 13_fast_gaussian_acc   fast_gaussian_test.py acc
  run 14_mixta               exp_mixta.py
  run 15_mixta2              exp_mixta2.py
  run 16_wasserstein_vs_fr   exp_wasserstein_vs_fisherrao.py
  run 17_wordnet_incumbentes exp_wordnet_incumbentes.py
  run 18_eficiencia_dim      exp_eficiencia_dimensional.py
  run 19_ablacion_kernel     exp_ablacion_kernel.py
  run 20_cat_incumbente      exp_cat_incumbente.py
  run 21_estabilizado        exp_estabilizado.py
  run 22_prosa_man           exp_prosa_man.py
  run 23_rope                exp_rope.py
  if [ "$MODO" = "extendido" ]; then
    run 30_mecanismo_umbral        exp_mecanismo_umbral.py
    run 31_profundidad_dyck4       exp1_profundidad.py
    run 32_codigo_python           exp_codigo_python.py
    run 33_codigo_extrapolativo    exp_codigo_extrapolativo.py
    run 34_codigo_extrapol_rep     exp_codigo_extrapolativo_rep.py
    run 35_sensibilidad_cap        exp_sensibilidad_cap.py
    run 36_fewshot_jerarquia       exp_fewshot_jerarquia.py
    run 37_respuesta_escala_19M    exp_respuesta_escala.py
  fi
fi

# Recolectar y comparar con lo publicado (solo los JSON escritos durante esta corrida)
{
  echo "Comparación byte a byte con los resultados publicados (JSON regenerados en esta corrida)"
  for f in verificacion_claude/resultados_*.json; do
    [ "$f" -nt "$RUN/.inicio" ] || continue
    b=$(basename "$f"); cp "$f" "$RUN/resultados_nuevos/"
    if [ ! -f "$RUN/resultados_publicados/$b" ]; then echo "NUEVO     $b"
    elif cmp -s "$f" "$RUN/resultados_publicados/$b"; then echo "IDENTICO  $b"
    else echo "DIFIERE   $b"; fi
  done
} > "$RUN/comparacion.txt"
log "=== Terminado. Resultados en $RUN ==="
