"""Líneas de referencia de n-gramas de caracteres para el corpus de prosa en inglés (§4.8e).

Calcula los bits por carácter (bpc) de modelos de n-gramas de caracteres (n = 1..4)
con suavizado aditivo, entrenados con el texto de entrenamiento y evaluados en el
texto de prueba que usa exp_prosa_man.py. El paper cita el bigrama y el 4-grama
como escala para los 3,60-3,64 bpc de los dos brazos dentro de la distribución.

Requiere corpus_man_train.txt y corpus_man_test.txt en esta carpeta. No se
redistribuyen (licencias heterogéneas): exp_prosa_man.py los genera a partir de las
páginas man del sistema y los deja en caché. Los SHA-256 que imprime este script se
pueden comparar con los de manifiesto_corpus_man.json.

Uso:  python3 ngram_referencia.py
Salida: resultados_ngram_referencia.json
"""
import collections
import hashlib
import json
import math
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
ALFA = 0.05          # suavizado aditivo
ORDENES = (1, 2, 3, 4)


def leer(nombre):
    with open(os.path.join(AQUI, nombre), encoding="utf-8") as f:
        return f.read()


def sha256(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def bpc_ngrama(n, entrenamiento, prueba, tam_vocab):
    """bpc de un n-grama de caracteres con suavizado aditivo ALFA."""
    cuenta = collections.Counter()
    contexto = collections.Counter()
    for i in range(n - 1, len(entrenamiento)):
        g = entrenamiento[i - n + 1:i + 1]
        cuenta[g] += 1
        contexto[g[:-1]] += 1
    total = 0.0
    m = 0
    for i in range(n - 1, len(prueba)):
        g = prueba[i - n + 1:i + 1]
        p = (cuenta[g] + ALFA) / (contexto[g[:-1]] + ALFA * tam_vocab)
        total += -math.log2(p)
        m += 1
    return total / m


def main():
    entrenamiento = leer("corpus_man_train.txt")
    prueba = leer("corpus_man_test.txt")
    # vocabulario: símbolos del entrenamiento más uno para los no vistos
    tam_vocab = len(set(entrenamiento)) + 1
    salida = {
        "corpus": "prosa en inglés de páginas man (exp_prosa_man.py)",
        "caracteres_entrenamiento": len(entrenamiento),
        "caracteres_prueba": len(prueba),
        "sha256_entrenamiento": sha256(entrenamiento),
        "sha256_prueba": sha256(prueba),
        "suavizado_aditivo": ALFA,
        "tam_vocabulario": tam_vocab,
        "bpc": {},
    }
    print(f"entrenamiento: {len(entrenamiento)} caracteres  sha256 {salida['sha256_entrenamiento']}")
    print(f"prueba:        {len(prueba)} caracteres  sha256 {salida['sha256_prueba']}")
    print(f"vocabulario: {tam_vocab} (incluye uno para no vistos); suavizado aditivo {ALFA}")
    for n in ORDENES:
        b = bpc_ngrama(n, entrenamiento, prueba, tam_vocab)
        salida["bpc"][f"{n}-grama"] = b
        print(f"  {n}-grama: {b:.4f} bpc")
    with open(os.path.join(AQUI, "resultados_ngram_referencia.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Guardado: resultados_ngram_referencia.json")


if __name__ == "__main__":
    main()
