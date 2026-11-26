# PRERREGISTRO — Prueba ES-GUÍAS: extrapolación de largo en texto clínico en español

**Estado:** aprobado por Eduardo el 21-oct-2026. El entrenamiento se ejecuta solo después de sellar este protocolo, `exp_guias_es.py` y `manifiesto_corpus_guias_es.json` con SHA-256 en `SELLO_GUIAS_ES.txt`.
**Fecha:** 21-oct-2026. **Autores:** Eduardo Quiñelen Rojas y Miguel Contreras Torrejon.
**Relación con el paper:** réplica del diseño de §4.8e de *Fisher–Rao Geodesic Attention* (v1.1, Zenodo 10.5281/zenodo.23486003) en un idioma y un dominio nuevos. Ninguna parte de esta prueba entra al paper sin decisión explícita de los autores.

## Pregunta

En §4.8e, el núcleo de Fisher–Rao con tope ‖μ‖ ≤ 12 modeló mejor que la atención euclidiana la prosa técnica en inglés **más allá del largo de entrenamiento** (−0,058 bpc en L = 512, p = 0,0002, 10/10 semillas), y peor dentro de ese largo. ¿Se mantiene la ventaja en **texto clínico en español**?

## Corpus (fijado antes de entrenar; ya construido con `--solo-corpus`)

- **Fuente:** marcos de la evidencia a la decisión (EtD) de las guías de práctica clínica del MINSAL; campo «evidencia de investigación» (`etd_criteria.research_evidence_text`) de una copia en solo lectura de la base de guías (`copia_postgres_guias_recs_etd_20260819.sqlite`, SHA-256 `b66c1fae…`). Es texto redactado por los equipos elaboradores; los campos marcados como rellenados por IA no se usan (no hay IA en este campo).
- **Procesamiento:** normalización Unicode NFC; textos idénticos deduplicados (se conserva la primera aparición en orden de guía e id); documentos unidos por una línea en blanco. Sin otra limpieza (se conservan cifras, citas y URL).
- **División por guía (no por documento), determinista:** las guías se ordenan por `guide_id` y la k-ésima va a prueba si k mod 10 = 0. Así ningún texto de una guía de prueba aparece en entrenamiento.
- **Tamaño:** 82 guías y 2.418 documentos. Entrenamiento: 73 guías y 5.549.216 caracteres. Prueba: 9 guías y 913.858 caracteres (Analgesia del parto; Cáncer colorrectal; Hipertensión arterial; Enfermedad renal crónica; Leucemias en menores de 15 años; Cáncer de tiroides; Enfermedad de Parkinson; Trauma ocular grave; Depresión en adolescentes). Vocabulario: 250 símbolos con el de relleno; 0,0005 % de los caracteres de prueba quedan fuera de vocabulario (se ignoran en la pérdida, igual que en §4.8e).
- **Manifiesto:** `manifiesto_corpus_guias_es.json` (SHA-256 de la base, del texto de entrenamiento y del de prueba, y lista de guías con su parte).

## Diseño (idéntico a §4.8e salvo el corpus y el número de semillas)

- Modelo de lenguaje a nivel de carácter; d_model 128, 2 capas, 4 cabezas (d_h = 32, sobre el umbral d_h ≥ 16 de §4.7); posiciones aprendidas absolutas hasta 512, como en §4.8d–e.
- Entrenamiento con secuencias de largo 128: 4.000 secuencias, 3 épocas, lotes de 32, AdamW (lr 5·10⁻⁴, wd 0,01).
- Evaluación en 400 secuencias de prueba de largo 128, 256 y 512.
- **Brazos:** atención euclidiana (`EuclideanAttention`) frente a Fisher–Rao estabilizado con tope ‖μ‖ ≤ 12 (`StableFR`), importados sin modificar del repositorio.
- **Semillas:** 201–220 (n = 20, nunca usadas en el proyecto). Se usan 20 en vez de las 10 de §4.8e para tener más poder; cada par de brazos tarda ≈ 1 min en CPU.

## Endpoint primario

bpc en la **zona extrapolativa a L = 512** (posiciones de destino ≥ 127), Fisher–Rao frente a euclidiana, t pareado bilateral, n = 20:
- p < 0,05 y FR < Euc: «la ventaja en extrapolación se traslada a texto clínico en español a esta escala»;
- p < 0,05 y FR > Euc: negativo declarado (se informa igual);
- p ≥ 0,05: «sin diferencia demostrada».

## Secundarios (descriptivos, sin corrección)

1. Zona extrapolativa a L = 256.
2. Control dentro de distribución (L = 128). **Predicción declarada:** gana la euclidiana, como en §4.8d–e.
3. Estabilidad: sd(FR) ≤ 1,5·sd(Euc) en la zona extrapolativa a L = 512.
4. Semillas ganadas por FR en el primario.

## Reglas

- No se cambian corpus, brazos, semillas, métricas ni umbrales después de ver datos; cualquier análisis adicional se rotula «post hoc».
- Se informa el resultado completo, sea positivo o negativo.
- Antes de ejecutar: sello SHA-256 de este protocolo, del script y del manifiesto en `SELLO_GUIAS_ES.txt`.

## Uso de la base

Para publicar un resultado se difundirán la lista de guías con sus URL públicas y los SHA-256, sin redistribuir la base.

## Limitaciones conocidas de antemano

- La escala es mínima (modelo pequeño en CPU), la misma de §4.8e.
- El texto de los EtD tiene fórmulas repetidas entre guías (p. ej., «No se realizó búsqueda de costo-efectividad…»). Se deduplican solo los textos idénticos; las fórmulas parecidas se conservan, como en cualquier corpus natural.
- Las posiciones son absolutas en ambos brazos, para replicar §4.8e. Con posiciones relativas (RoPE) la comparación sería otra; queda para una prueba aparte.
