"""Compara, valor por valor, los JSON de la re-ejecución con los publicados en v1.1.

Uso (desde reproduccion/):
    python3 scripts/comparar_valores.py reejecucion_mac_studio/resultados \
        reejecucion_mac_studio/resultados_publicados_v1.1

Recorre todas las hojas numéricas de cada JSON y excluye únicamente los campos de
tiempo de reloj (claves que terminan en "secs"). Para referencia, informa también
el conteo de progreso_mac.py, que es el de la tabla A.2 (679): su filtro de tiempos
descarta además cualquier clave que contenga "ms" o "_s" (p. ej. params, arms/…,
*_seeds), aunque no sean tiempos.
"""
import json
import math
import os
import sys

NATURALES = {"resultados_estabilizado.json", "resultados_prosa_man.json"}  # corpus que cambiaron (A.4)


def hojas(x, ruta=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from hojas(v, f"{ruta}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from hojas(v, f"{ruta}[{i}]")
    else:
        yield ruta, x


def es_numero(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def filtro_progreso_mac(k):
    return any(t in k.lower() for t in ("secs", "time", "tiempo", "/s]", "_s", "ms"))


def iguales(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def main(nuevos, publicados):
    tot = {"sinteticos": [0, 0], "progreso_mac": [0, 0]}
    print(f"{'archivo':42s} {'sin secs':>12s} {'def. 679':>10s}  otros")
    for fn in sorted(f for f in os.listdir(nuevos) if f.endswith(".json")):
        a = dict(hojas(json.load(open(os.path.join(nuevos, fn)))))
        b = dict(hojas(json.load(open(os.path.join(publicados, fn)))))
        claves = sorted(set(a) & set(b))
        num = [k for k in claves if es_numero(a[k]) and es_numero(b[k]) and not k.endswith("secs")]
        eq = sum(iguales(a[k], b[k]) for k in num)
        pm = [k for k in claves if es_numero(a[k]) and es_numero(b[k]) and not filtro_progreso_mac(k)]
        eq_pm = sum(iguales(a[k], b[k]) for k in pm)
        otros = sum(a[k] != b[k] for k in claves if not (es_numero(a[k]) and es_numero(b[k])))
        nota = "  (corpus distinto, ver A.4)" if fn in NATURALES else ""
        if set(a) != set(b):
            nota += f"  claves solo en uno: {len(set(a) ^ set(b))}"
        print(f"{fn:42s} {eq:5d}/{len(num):<6d} {eq_pm:4d}/{len(pm):<5d} {otros}{nota}")
        if fn not in NATURALES:
            tot["sinteticos"][0] += eq; tot["sinteticos"][1] += len(num)
            tot["progreso_mac"][0] += eq_pm; tot["progreso_mac"][1] += len(pm)
    s, p = tot["sinteticos"], tot["progreso_mac"]
    print(f"\nExperimentos sintéticos (11), excluyendo solo 'secs': {s[0]}/{s[1]} valores idénticos")
    print(f"Mismos experimentos con el filtro de progreso_mac.py (tabla A.2): {p[0]}/{p[1]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
