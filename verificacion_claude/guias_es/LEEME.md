# Prueba prerregistrada: extrapolación de largo en texto clínico en español (§4.8g del paper)

- `PREREGISTRO_GUIAS_ES.md`: protocolo, sellado antes de entrenar (`SELLO_GUIAS_ES.txt`, 21-oct-2026 18:51 UTC).
- `manifiesto_corpus_guias_es.json`: las 82 guías de práctica clínica (73 de entrenamiento y 9 de prueba), con el número de documentos y caracteres de cada una y los SHA-256 del texto de entrenamiento y de prueba.
- `lista_guias_url.csv`: las 82 guías con su URL pública, como anunciaba el protocolo. Se agregó al publicar y no forma parte de los archivos sellados.
- `exp_guias_es.py`: script de la prueba.
- `log_guias_es.txt` y `resultados_guias_es.json`: registro y resultados (n = 20, semillas 201–220).

**Corpus.** El texto proviene de guías clínicas públicas, pero se extrajo de una copia de trabajo de una base de datos que no se redistribuye. Tampoco se incluye el texto del corpus. Los SHA-256 del manifiesto permiten confirmar el corpus a quien tenga el mismo extracto de la base; no bastan para reconstruirlo byte a byte desde las páginas públicas, porque el texto es un campo de la base, sin duplicados.

**Diferencias con la versión sellada.** El script publicado difiere del sellado solo en dos líneas: la ruta local de esta carpeta (`REPO_VC`), que ahora es relativa, y la ruta local de la base (`DB`), que ahora se toma de la variable de entorno `GUIAS_ES_DB`. El protocolo publicado omite dos frases de la sección «Uso de la base», de carácter administrativo sobre la copia de la base; el resto del texto es idéntico al sellado. Por eso los SHA-256 de estos dos archivos no coinciden con los del sello. El manifiesto no cambió, y su hash coincide con el sello.
