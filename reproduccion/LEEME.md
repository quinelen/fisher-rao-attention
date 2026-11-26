# reproduccion/

Artefactos del Apéndice A, «Re-execution on Separate Hardware and a Pre-Registered Follow-Up». Todos los scripts del paper se corrieron sin cambios desde la versión v1.1 (commit `b7cd4b4`). Corridas hechas por Miguel Contreras Torrejon en octubre de 2026.

## Máquinas

| Carpeta | Máquina | Entorno |
|---|---|---|
| `reejecucion_mac_studio/` | Mac Studio, Apple M4 Max, 16 núcleos, macOS 27.0.1 | CPython 3.9.6 (sistema), PyTorch 2.8.0, NumPy 1.26.4, SciPy 1.13.1 |
| `seguimiento_py313/` | la misma Mac Studio | CPython 3.13.15, PyTorch 2.14.1, NumPy 2.5.4, SciPy 1.18.1 |
| `colab/` | Google Colab, CPU x86-64, Linux | CPython 3.13, PyTorch 2.11.0, NumPy 1.26.4 |

## Contenido

**`reejecucion_mac_studio/`**: re-ejecución de los 14 experimentos del README (A.1–A.4), 25-oct-2026, 20:16–22:27.
- `logs/`: un registro por paso: instalación (00), manifiestos de corpus (01–02), experimentos (10–23). El de instalación está vacío porque `pip` corrió en modo silencioso; el entorno de la corrida es la línea «Entorno:» de `estado.txt`.
- `resultados/`: los JSON que escribió cada experimento.
- `resultados_publicados_v1.1/`: los JSON publicados en v1.1, copiados al inicio de la corrida, para comparar.
- `comparacion_valor_por_valor.txt`: comparación de cada valor numérico (tabla A.2). Lo genera `scripts/comparar_valores.py`.
- `comparacion.txt`: comparación **byte a byte**. Marca 7 archivos como «DIFIERE»: en 5 la diferencia está solo en los campos de tiempo `secs`; los otros 2 (`resultados_estabilizado.json` y `resultados_prosa_man.json`) difieren en todos sus valores porque el corpus cambió (A.4).
- `mac_progreso.json`: estado y comparación por paso, escrito por `scripts/progreso_mac.py`. De aquí sale el conteo de 679 de la tabla A.2 (ver la nota 1).
- `mac_tres_geometrias.json`: comparación semilla por semilla del benchmark principal.
- `estado.txt`, `consola_core.log`: bitácora de la corrida.

**`manifiestos_corpus/`**: los dos corpus que cambiaron respecto de las corridas originales (A.4).
- `manifiesto_corpus_man_macos27.0.1.json`: 279 páginas man de macOS 27.0.1, con el número de caracteres de cada una y los SHA-256 de los textos de entrenamiento y de prueba. Escrito durante la re-ejecución.
- `manifiesto_corpus_codigo_cpython3.9.6_regenerado.json`: biblioteca estándar de CPython 3.9.6. Ver la nota 2.

**`seguimiento_py313/`**: prueba prerregistrada de A.6.
- `PREREGISTRO_PY313.md`: el protocolo. `SELLO_PREREGISTRO_PY313.txt` tiene su SHA-256 (`96cbe2bc…a494a9722`).
- `manifiesto_corpus_py313.json`: corpus de CPython 3.13.15, registrado antes de entrenar.
- `logs/`: fase A (semillas 1–5, descriptiva) y fase B (semillas 6–15, endpoint primario).
- `resultados/`: JSON de ambas fases.
- `veredicto.txt`, `veredicto_py313.json`: tabla A.6. El endpoint primario no se cumplió.
- `estado.txt`, `consola_py313.log`: bitácora, que incluye la verificación del sello antes de correr.

**`colab/`**: corrida en otra plataforma (A.5).
- `paperEduMig_celdas_0_1.ipynb`: las celdas 0 y 1 del cuaderno de Colab, con sus fuentes y salidas sin modificar. Ver la nota 4.
- `salida_celda1_tres_scripts.txt`: la salida de la celda 1, que corre `fast_gaussian_test.py`, `exp_codigo_extrapolativo.py` y `exp_rope.py`.

**`scripts/`**: los envoltorios usados. Crean el entorno de Python (instalan las dependencias), corren los scripts del paper sin modificarlos, registran el entorno y comparan salidas. El del seguimiento, además, escribe el manifiesto del corpus y calcula las pruebas t pareadas y el veredicto de la tabla A.6.
- `ejecutar_mac.sh`, `lanzar_core.command`: re-ejecución.
- `seguimiento_py313.sh`, `lanzar_py313.command`: seguimiento. Verifica el sello antes de entrenar.
- `progreso_mac.py`: comparación por paso (conteos de la tabla A.2).
- `comparar_valores.py`: comparación valor por valor, con los dos criterios de la nota 1.

Los envoltorios se corrieron desde copias en la raíz del repositorio. Escriben en `corridas/<fecha>-core` y `corridas/<fecha>-py313`, que aquí se archivan como `reejecucion_mac_studio/` y `seguimiento_py313/` (`resultados_nuevos` pasó a llamarse `resultados`, y `resultados_publicados`, `resultados_publicados_v1.1`). El lanzador del seguimiento espera el protocolo y el sello en `corridas/seguimiento_py313/`. Para repetir la corrida hay que copiar los envoltorios a la raíz del repositorio y esos dos archivos a esa carpeta; si faltan, la verificación del sello no detiene el lanzador.

## Notas

1. **Cuántos valores se compararon.** La tabla A.2 muestra dos conteos. El principal excluye solo los campos `secs`: los 11 experimentos sintéticos tienen 976 valores numéricos, y los 976 son idénticos. El segundo, 679, es el del filtro de tiempos de `progreso_mac.py`, que descarta además claves que no son tiempos, como `params` o `arms/…`, porque contienen «ms» o «_s». Ambos conteos están en `comparacion_valor_por_valor.txt`.

2. **Manifiesto del corpus de código.** Durante la re-ejecución, `manifiesto_corpus_codigo.py` solo comparó contra el manifiesto publicado y dejó un resumen en `logs/01_manifiesto_codigo.log`; no guardó un manifiesto propio. El archivo incluido se regeneró el 16-nov-2026 en la misma máquina y con el mismo CPython 3.9.6, ya en macOS 27.2.1. Coincide con el resumen del 25-oct en número de archivos (46), caracteres de entrenamiento y prueba (1 199 078 y 37 771), vocabulario (98) y en los dos archivos que difieren de v1.1 (`_osx_support.py`, `_sysconfigdata__darwin_darwin.py`). Eso no prueba que el corpus sea idéntico byte a byte al del 25-oct.

3. **Corpus no incluidos.** Los textos de corpus no van en esta carpeta: las páginas man tienen licencias heterogéneas y la biblioteca estándar de CPython se regenera con los scripts del repositorio. Los manifiestos permiten verificar una regeneración.

4. **Colab.** Los JSON que escribió Colab no se conservaron: quedaron en la sesión, que es temporal. Lo que queda es la salida de consola del cuaderno, con todas las cifras de A.3 y A.5. El cuaderno incluido conserva solo las celdas 0 y 1, que son las del apéndice; el original tenía otras 12 celdas, prototipos sin relación con el paper. La celda 0 es un primer intento que falló porque buscó los scripts en la carpeta equivocada; su registro de `pip` documenta el entorno. La sesión tenía una GPU asignada (A100), pero los scripts del repositorio no la usan. Según las marcas de tiempo del cuaderno, las dos celdas se ejecutaron el 22-oct-2026. El campo `elapsed` de la celda 1 (unos 5 min) no refleja la duración de la corrida: los tiempos por semilla impresos en su salida suman unos 97 min; no sabemos qué mide ese campo en una sesión que se reconecta. Del cuaderno se quitaron los datos de la cuenta de Google (nombre, identificador de usuario y etiqueta de autoría); las fuentes y las salidas no se tocaron. La fuente de la celda 1 termina con una «q» tecleada después de la ejecución. Los carteles que imprime la celda 1 («La Joya de la Corona», etc.) se escribieron antes de conocer los resultados y no son conclusiones del paper: la observación de Colab sobre código no se replicó en la prueba prerregistrada (A.5–A.6).

5. **Rutas locales.** Los registros contienen rutas absolutas de la máquina donde se generaron.

6. **Veredictos impresos por los scripts.** Algunos scripts imprimen una línea de veredicto («transferencia extrapolativa … DEMOSTRADA», «empate estadístico»). Esas frases se escribieron junto con los experimentos, antes de la redacción de la v1.2, y quedan en los registros porque los scripts se corrieron sin modificar; las conclusiones son las del texto del paper.

7. **Numeración.** En esta carpeta, «tabla A.n» es la tabla de la subsección A.n del apéndice; en el PDF las tablas llevan numeración corrida.

8. **Verificar el sello.** Desde `seguimiento_py313/`: `shasum -a 256 -c SELLO_PREREGISTRO_PY313.txt` (en Linux, `sha256sum -c`).
