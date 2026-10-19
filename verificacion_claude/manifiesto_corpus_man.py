# ============================================================================
# Manifiesto del corpus de prosa técnica de §4.8e (exp_prosa_man.py).
#
# El corpus se renderiza desde las páginas man1 del sistema local, que tienen
# licencias heterogéneas; por eso no se redistribuye. Este script reconstruye
# la MISMA selección determinista de build_corpus_man(), escribe el manifiesto
# local (lista ordenada de páginas con su asignación train/test, versión del
# sistema operativo y SHA-256 del texto) en manifiesto_corpus_man_local.json y
# lo compara con el manifiesto publicado del corpus del paper
# (manifiesto_corpus_man.json), que no se modifica.
# Ejecutar: python manifiesto_corpus_man.py
# ============================================================================
import hashlib, json, os, platform, subprocess
from exp_prosa_man import TARGET_BYTES

MANDIR = "/usr/share/man/man1"


def render(stem):
    return subprocess.run(
        ["sh", "-c", f"MANWIDTH=80 man -P cat {stem} 2>/dev/null | col -b"],
        capture_output=True, text=True, timeout=20).stdout


if __name__ == "__main__":
    names = sorted(os.listdir(MANDIR))
    train, test, pages, total = [], [], [], 0
    for fn in names:
        stem = fn[:-3] if fn.endswith(".gz") else fn
        if stem.endswith(".1"):
            stem = stem[:-2]
        try:
            out = render(stem)
        except Exception:
            continue
        if len(out) < 1500:
            continue
        split = "test" if len(pages) % 10 == 0 else "train"
        (test if split == "test" else train).append(out)
        pages.append({"page": stem, "split": split, "chars": len(out)})
        total += len(out)
        if total > TARGET_BYTES:
            break
    tr, te = "".join(train), "".join(test)
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    manifest = {
        "description": "Technical English prose corpus of §4.8e, rendered from local man1 pages "
                       "(MANWIDTH=80, man -P cat | col -b); split 90/10 by page.",
        "os": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "n_pages": len(pages),
        "train_chars": len(tr), "test_chars": len(te),
        "train_sha256": sha(tr), "test_sha256": sha(te),
        "pages": pages,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(here, "manifiesto_corpus_man.json")         # publicado
    out_path = os.path.join(here, "manifiesto_corpus_man_local.json")   # local
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(pages)} páginas; train={len(tr)} test={len(te)} caracteres "
          f"(manifiesto local: manifiesto_corpus_man_local.json)")
    if not os.path.exists(ref_path):
        print("No se encontró el manifiesto publicado manifiesto_corpus_man.json")
    else:
        with open(ref_path) as f:
            ref = json.load(f)
        print("misma lista de páginas que el paper:",
              [(p["page"], p["split"]) for p in pages]
              == [(p["page"], p["split"]) for p in ref["pages"]])
        ok = (manifest["train_sha256"] == ref["train_sha256"]
              and manifest["test_sha256"] == ref["test_sha256"])
        print("coincide con el corpus usado en el paper:", ok)
        if not ok:
            rp = {p["page"]: (p["split"], p["chars"]) for p in ref["pages"]}
            lp = {p["page"]: (p["split"], p["chars"]) for p in pages}
            print("páginas que difieren:",
                  sorted(k for k in set(rp) | set(lp) if rp.get(k) != lp.get(k)))
