# Pre-registro: extrapolación en código con el kernel original, corpus CPython 3.13

**Fecha de registro:** 2026-10-25, antes de ejecutar cualquier entrenamiento de este seguimiento.
**Autor:** Miguel Contreras Torrejon (redactado con Claude). Seguimiento de §4.8d de
"Fisher–Rao Geodesic Attention" (v1.1, DOI 10.5281/zenodo.23486003).

## Antecedente

En el paper, el kernel isotrópico original (sin tope de la norma de μ) no cumplió el endpoint
pre-registrado de extrapolación en código sobre la biblioteca estándar de CPython 3.9.6:
n=5 (semillas 1–5), Δ = −0.077 bpc, p = 0.136; réplica n=10 (semillas 6–15), Δ = −0.049, p = 0.28.
El fallo se atribuyó a varianza entre semillas y se corrigió con un tope ‖μ‖ ≤ 12.

En Google Colab (observación post hoc, no pre-registrada), el mismo script con las semillas 1–5
sobre la biblioteca estándar de Python 3.13 dio Δ = −0.110 bpc, p = 0.012, 5/5 semillas, con
dispersión de Fisher–Rao 0.43× la euclidiana.

## Hipótesis

H1: con un corpus construido desde la biblioteca estándar de CPython 3.13, el kernel original
(sin tope) predice mejor que la atención euclidiana en la zona extrapolativa a L=512.

## Diseño

- Scripts del repositorio v1.1 **sin modificar**: `exp_codigo_extrapolativo.py` (semillas 1–5)
  y `exp_codigo_extrapolativo_rep.py` (semillas 6–15). Mismos hiperparámetros, arquitectura y
  presupuesto que en el paper.
- Único cambio: el intérprete (CPython 3.13 en el Mac Studio de Miguel), y con él el corpus.
- Antes de entrenar se registra el manifiesto del corpus (tamaños, vocabulario y SHA-256 de
  train y test).
- **Fase A (descriptiva):** semillas 1–5. Reproduce la observación del Colab en este equipo.
- **Fase B (confirmatoria):** semillas 6–15, las mismas de la réplica fallida del paper.

## Endpoint primario (fase B, decisorio)

Media de `bpc_extra_L512` (bpc en posiciones ≥ 128 de secuencias de 512): Fisher–Rao menor que
euclidiana, t pareado bilateral, p < 0.05, n = 10.

## Secundarios (reportados, no decisorios)

1. `bpc_extra_L256`, misma prueba.
2. Estabilidad: desviación estándar (ddof=1) de Fisher–Rao dividida por la euclidiana en
   `bpc_extra_L512`, con el mismo límite que el paper usó en §4.8d: ≤ 1.5.
3. Control dentro de distribución `bpc_L128`: se espera que la euclidiana sea mejor.
4. Análisis conjunto n=15 (fases A+B), etiquetado como agregación post hoc.

## Interpretación acordada de antemano

- **Cumplido:** el fallo del kernel original en §4.8d depende del corpus. Se propone a los
  autores una nota de robustez en §4.8d que reporte ambos corpus.
- **No cumplido:** la observación del Colab se considera no replicada y el relato de §4.8d se
  mantiene.
- Ambos resultados se reportan igual.

## Limitaciones conocidas

El corpus 3.13 del Mac no es idéntico al del Colab (archivos que dependen de la plataforma y de
la versión de parche) y la versión de PyTorch difiere de la del paper. Por eso se registra el
manifiesto del corpus antes de entrenar.
