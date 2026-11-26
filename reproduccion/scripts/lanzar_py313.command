#!/bin/bash
# Doble clic en Finder: lanza el seguimiento pre-registrado con Python 3.13 en segundo plano.
cd "$(dirname "$0")"
nohup ./seguimiento_py313.sh > corridas/consola_py313.log 2>&1 &
PID=$!
sleep 3
echo "Seguimiento Python 3.13 en segundo plano (PID $PID). Puedes cerrar esta ventana."
echo "Para detenerlo: kill $PID"
echo
tail -f "$(cat corridas/ULTIMA_PY313.txt)/estado.txt"
