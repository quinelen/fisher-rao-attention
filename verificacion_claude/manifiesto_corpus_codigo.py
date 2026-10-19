# ============================================================================
# Manifiesto del corpus de código Python de §4.8d (exp_codigo_python.py,
# exp_codigo_extrapolativo*.py, exp_estabilizado.py, exp_sensibilidad_cap.py).
#
# build_corpus() lee los .py de la biblioteca estándar del intérprete que se
# esté usando, así que el corpus depende de la versión de Python; todas las
# corridas del paper usaron CPython 3.9.6. Este script toma el texto de la
# MISMA función build_corpus() (importada, no copiada), lista los archivos que
# la componen (con su asignación train/test y su número de caracteres), el
# tamaño del vocabulario y el SHA-256 del texto, y lo compara con el manifiesto
# publicado (manifiesto_corpus_codigo.json), que no se modifica.
# Ejecutar: python manifiesto_corpus_codigo.py
#           python manifiesto_corpus_codigo.py --escribir   (re-genera la referencia)
# ============================================================================
import glob, hashlib, json, os, platform, sys, sysconfig
from exp_codigo_python import build_corpus, make_vocab, CORPUS_BYTES


def list_files():
    """Misma selección que build_corpus(): .py ordenados de la stdlib, se
    descartan los de menos de 2000 caracteres, archivo i (índice en la lista
    completa) a test si i % 10 == 0, hasta superar CORPUS_BYTES."""
    stdlib = sysconfig.get_paths()["stdlib"]
    files, total = [], 0
    for i, fp in enumerate(sorted(glob.glob(os.path.join(stdlib, "*.py")))):
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                t = f.read()
        except OSError:
            continue
        if len(t) < 2000:
            continue
        files.append({"file": os.path.basename(fp),
                      "split": "test" if i % 10 == 0 else "train",
                      "chars": len(t), "_text": t})
        total += len(t)
        if total > CORPUS_BYTES:
            break
    return files


if __name__ == "__main__":
    tr, te = build_corpus()
    files = list_files()
    # la lista debe reconstruir exactamente el texto de build_corpus()
    assert "".join(f["_text"] for f in files if f["split"] == "train") == tr
    assert "".join(f["_text"] for f in files if f["split"] == "test") == te
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    _, vocab = make_vocab(tr)
    manifest = {
        "description": "Python source corpus of §4.8d: *.py files of the running "
                       "interpreter's standard library, sorted, files under 2000 "
                       "characters skipped, split 90/10 by file (file index i -> "
                       "test if i % 10 == 0), stopping once 1.2M characters are exceeded.",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "n_files": len(files),
        "train_files": [{"file": f["file"], "chars": f["chars"]}
                        for f in files if f["split"] == "train"],
        "test_files": [{"file": f["file"], "chars": f["chars"]}
                       for f in files if f["split"] == "test"],
        "train_chars": len(tr), "test_chars": len(te),
        "total_chars": len(tr) + len(te),
        "train_sha256": sha(tr), "test_sha256": sha(te),
        "vocab_size": vocab,
        "vocab_note": "distinct characters of the train text + 1 (index 0 = padding/unknown)",
    }
    here = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(here, "manifiesto_corpus_codigo.json")
    print(f"{manifest['python_implementation']} {manifest['python_version']}: "
          f"{len(files)} archivos; train={len(tr)} test={len(te)} caracteres "
          f"({len(manifest['test_files'])} archivos de prueba); vocabulario={vocab}")
    if "--escribir" in sys.argv or not os.path.exists(ref_path):
        with open(ref_path, "w") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
        print("manifiesto escrito en manifiesto_corpus_codigo.json")
    else:
        with open(ref_path) as f:
            ref = json.load(f)
        ok = (manifest["train_sha256"] == ref["train_sha256"]
              and manifest["test_sha256"] == ref["test_sha256"])
        print("coincide con el corpus usado en el paper:", ok)
        if not ok:
            rf = {d["file"]: d["chars"] for d in ref["train_files"] + ref["test_files"]}
            lf = {f["file"]: f["chars"] for f in files}
            print("archivos que difieren:",
                  sorted(k for k in set(rf) | set(lf) if rf.get(k) != lf.get(k)))
