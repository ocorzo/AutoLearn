# Recomendaciones a partir de playlists externas

Estas herramientas construyen embeddings `item2vec` a partir de playlists
humanas y los usan para proponer agrupaciones de una biblioteca local. El
repositorio trae sólo el código: no incluye el Spotify Million Playlist Dataset
(MPD), música, coincidencias con una biblioteca ni artefactos derivados.

## Requisitos

- Python 3.11 o posterior.
- Acceso legítimo al dataset que se vaya a procesar y autorización para el uso
  previsto.
- Espacio suficiente fuera del repositorio para el corpus, DuckDB y vectores.

Instala las dependencias en un entorno virtual aislado:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/requirements-recommender.txt
```

Cuando se use el MPD, hay que respetar sus términos vigentes. En particular,
no añadas el dataset ni sus transformaciones, índices, embeddings o resultados
de playlists a Git. `.gitignore` protege las ubicaciones locales habituales,
pero conviene trabajar también fuera de este repositorio.

## Flujo de trabajo

1. Aplana el dataset autorizado a JSON Lines. Usa rutas de tu propio equipo:

   ```bash
   python3 tools/flatten_spotify_mpd.py \
     /ruta/al/dataset/Data \
     /ruta/privada/mpd-tracks.jsonl.gz
   ```

   Cada línea conserva únicamente el identificador de playlist, su nombre,
   canción y artista. Para una prueba de un slice, añade `--limit-files 1`.

2. Crea fuera de Git una base DuckDB con el corpus ordenado por playlist y las
   coincidencias de tu biblioteca local. Esta preparación depende de la fuente
   de biblioteca y por ello no se automatiza con una ruta personal fija.

3. Entrena los vectores de contexto:

   ```bash
   python3 tools/train_item2vec.py \
     --db /ruta/privada/catalogo.duckdb \
     --output-dir /ruta/privada/modelo
   ```

4. Analiza comunidades de las canciones que ya posees y genera borradores:

   ```bash
   python3 tools/analyze_item2vec.py \
     --db /ruta/privada/catalogo.duckdb \
     --vectors /ruta/privada/modelo/mpd_item2vec_64d.kv \
     --output-dir /ruta/privada/analisis

   python3 tools/recommend_from_embeddings.py \
     --db /ruta/privada/catalogo.duckdb \
     --vectors /ruta/privada/modelo/mpd_item2vec_64d.kv \
     --communities /ruta/privada/analisis/local_embedding_communities.csv \
     --output-dir /ruta/privada/borradores
   ```

Los CSV resultantes deben revisarse antes de importar playlists. La similitud
del embedding es una señal de afinidad; para comunicar una certeza, hace falta
calibrarla contra un conjunto de validación independiente.
