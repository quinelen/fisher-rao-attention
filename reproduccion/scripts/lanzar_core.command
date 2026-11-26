#!/bin/bash
# Doble clic en Finder: lanza la batería "core" de fisher-rao-attention en segundo plano.
cd "$(dirname "$0")"
mkdir -p corridas
nohup ./ejecutar_mac.sh core > corridas/consola_core.log 2>&1 &
PID=$!
sleep 3
echo "Corrida 'core' en segundo plano (PID $PID). Puedes cerrar esta ventana sin detenerla."
echo "Para detenerla: kill $PID"
echo "Progreso en: $(cat corridas/ULTIMA.txt)/estado.txt"
echo
tail -f "$(cat corridas/ULTIMA.txt)/estado.txt"
