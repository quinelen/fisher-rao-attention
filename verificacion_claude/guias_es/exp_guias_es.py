# ============================================================================
# PRUEBA ES-GUÍAS — extrapolación de largo en texto clínico en español
# (protocolo en PREREGISTRO_GUIAS_ES.md; no se ejecuta el entrenamiento antes
#  de sellar protocolo, script y manifiesto en SELLO_GUIAS_ES.txt)
#
# Corpus: texto de «evidencia de investigación» de los marcos EtD de guías de
# práctica clínica del MINSAL (campo research_evidence_text), deduplicado,
# normalizado NFC y dividido 90/10 POR GUÍA (determinista).
#
# Diseño idéntico a §4.8e del paper (exp_prosa_man.py): char-level, train L=128,
# eval {128, 256, 512}, 4000 seq, 3 épocas, d128/2capas/4cabezas.
# Brazos: euclidiano vs FR-ESTABILIZADO (tope ‖μ‖ ≤ 12). Semillas 201–220.
# ENDPOINT PRIMARIO: bpc zona extrapolativa L=512, FR < Euc, p<0.05 (n=20).
# Uso:
#   python -B exp_guias_es.py --solo-corpus   # construye corpus y manifiesto
#   python -B exp_guias_es.py                 # corrida completa (tras el sello)
# ============================================================================
import sys
sys.dont_write_bytecode = True
import os, json, hashlib, sqlite3, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_VC = os.path.dirname(HERE)  # publicación: ruta relativa (ver LEEME.md)
DB = os.environ.get("GUIAS_ES_DB", "copia_postgres_guias_recs_etd_20260819.sqlite")  # publicación: ruta configurable (ver LEEME.md)
CACHE_TRAIN = os.path.join(HERE, "corpus_guias_es_train.txt")
CACHE_TEST = os.path.join(HERE, "corpus_guias_es_test.txt")
MANIFIESTO = os.path.join(HERE, "manifiesto_corpus_guias_es.json")
SEEDS = list(range(201, 221))


def sha256_archivo(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_corpus():
    """Lee la base en solo lectura, deduplica textos exactos (se conserva la primera
    aparición en orden guide_id, id), normaliza NFC y divide por guía: la guía k-ésima
    (en orden de guide_id) va a prueba si k % 10 == 0. Documentos unidos con una línea
    en blanco. Cachea texto y manifiesto."""
    if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
        with open(CACHE_TRAIN, encoding="utf-8") as f1, open(CACHE_TEST, encoding="utf-8") as f2:
            return f1.read(), f2.read()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        """select r.guide_id, ec.id, ec.research_evidence_text
           from etd_criteria ec
           join etds e on e.id = ec.etd_id
           join recommendations r on r.id = e.recommendation_id
           where ec.research_evidence_text is not null
           order by r.guide_id, ec.id""").fetchall()
    titulos = dict(con.execute("select id, title from guides").fetchall())
    seen, por_guia = set(), {}
    for g, _, t in rows:
        t = unicodedata.normalize("NFC", t).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        por_guia.setdefault(g, []).append(t)
    guias = sorted(por_guia)
    train, test, man = [], [], []
    for k, g in enumerate(guias):
        txt = "\n\n".join(por_guia[g])
        parte = "test" if k % 10 == 0 else "train"
        (test if parte == "test" else train).append(txt)
        man.append({"guide_id": g, "titulo": titulos.get(g), "parte": parte,
                    "documentos": len(por_guia[g]), "caracteres": len(txt)})
    tr, te = "\n\n".join(train), "\n\n".join(test)
    with open(CACHE_TRAIN, "w", encoding="utf-8") as f1, open(CACHE_TEST, "w", encoding="utf-8") as f2:
        f1.write(tr); f2.write(te)
    simbolos_train = sorted(set(tr))
    json.dump({
        "fuente": "copia_postgres_guias_recs_etd_20260819.sqlite (tabla etd_criteria, campo research_evidence_text)",
        "sha256_base": sha256_archivo(DB),
        "guias": len(guias), "guias_train": sum(m["parte"] == "train" for m in man),
        "guias_test": sum(m["parte"] == "test" for m in man),
        "caracteres_train": len(tr), "caracteres_test": len(te),
        "documentos": sum(m["documentos"] for m in man),
        "vocabulario_train_con_pad": len(simbolos_train) + 1,
        "fraccion_test_fuera_de_vocabulario": round(sum(c not in set(simbolos_train) for c in te) / max(1, len(te)), 6),
        "sha256_train": hashlib.sha256(tr.encode("utf-8")).hexdigest(),
        "sha256_test": hashlib.sha256(te.encode("utf-8")).hexdigest(),
        "por_guia": man,
    }, open(MANIFIESTO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return tr, te


if __name__ == "__main__":
    train_txt, test_txt = build_corpus()
    print(f"Corpus guías ES: train={len(train_txt):,} test={len(test_txt):,} caracteres")
    if "--solo-corpus" in sys.argv:
        sys.exit(0)

    sys.path.insert(0, REPO_VC)
    import torch, numpy as np
    from scipy import stats
    from exp_codigo_extrapolativo import (seqs_of, run, TRAIN_LEN, EVAL_LENS,
                                           N_TRAIN_SEQ, N_EVAL_SEQ)
    from exp_codigo_python import make_vocab, encode
    from exp_tres_geometrias import EuclideanAttention
    from exp_estabilizado import StableFR

    torch.set_num_threads(8)
    stoi, VOCAB = make_vocab(train_txt)
    tr_arr, te_arr = encode(train_txt, stoi), encode(test_txt, stoi)
    rng0 = np.random.RandomState(0)
    Xtr = seqs_of(tr_arr, N_TRAIN_SEQ, TRAIN_LEN, rng0)
    evals = {L: seqs_of(te_arr, N_EVAL_SEQ, L, rng0) for L in EVAL_LENS}
    print(f"vocab={VOCAB}; train {Xtr.shape} (len {TRAIN_LEN}); eval {EVAL_LENS}\n")

    ARMS = {"euclidean": EuclideanAttention, "fr_estable": StableFR}
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for name, cls in ARMS.items():
            r = run(cls, seed, Xtr, evals, VOCAB)
            res[name].append(r)
            print(f"seed {seed} {name:<11}: extra256={r['bpc_extra_L256']:.4f} "
                  f"extra512={r['bpc_extra_L512']:.4f} in128={r['bpc_L128']:.4f} ({r['secs']}s)",
                  flush=True)

    n = len(SEEDS)
    print(f"\n*** VEREDICTO PRERREGISTRADO (guías ES, n={n}, semillas {SEEDS[0]}–{SEEDS[-1]}) ***")
    for k, rol in [("bpc_extra_L512", "PRIMARIO"), ("bpc_extra_L256", "secundario"),
                   ("bpc_L128", "control IN")]:
        e = np.array([r[k] for r in res["euclidean"]])
        f_ = np.array([r[k] for r in res["fr_estable"]])
        t, p = stats.ttest_rel(f_, e)
        print(f"  [{rol:>10}] {k}: Euc={e.mean():.4f}±{e.std():.4f}  FRs={f_.mean():.4f}±{f_.std():.4f}  "
              f"Δ={f_.mean()-e.mean():+.4f}  t={t:+.2f}  p={p:.4f}")
    e512 = np.array([r["bpc_extra_L512"] for r in res["euclidean"]])
    f512 = np.array([r["bpc_extra_L512"] for r in res["fr_estable"]])
    t, p = stats.ttest_rel(f512, e512)
    exito = f512.mean() < e512.mean() and p < 0.05
    estable = f512.std() <= 1.5 * e512.std()
    print(f"\n  semillas ganadas por FR en extra_L512: {int((f512 < e512).sum())}/{n}")
    print(f"  estabilidad (sd_FR <= 1.5 sd_Euc): {'SÍ' if estable else 'NO'} "
          f"(sd_FR={f512.std():.4f}, sd_Euc={e512.std():.4f})")
    print("  ENDPOINT PRIMARIO: " + ("CUMPLIDO" if exito else "NO CUMPLIDO — negativo declarado"))
    with open(os.path.join(HERE, "resultados_guias_es.json"), "w") as fj:
        json.dump({a: res[a] for a in ARMS}, fj, indent=1)
    print("\nGuardado: resultados_guias_es.json")
