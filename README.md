# Chaak

Chaak reúne herramientas locales para trabajar con modelos de lenguaje y con
una biblioteca musical personal. Incluye un runtime compatible con OpenAI, un
pipeline de entrenamiento/evaluación y un recomendador experimental de
playlists basado en coocurrencias humanas.

No contiene música, una biblioteca personal, ni el Spotify Million Playlist
Dataset (MPD), ni resultados derivados de ellos.

## Estructura

```text
autolearn/                Pipeline de entrenamiento y ensayos
services/chaak-runtime/   Gateway, MCP, Mem0 y Dockerfiles propios
deploy/compose.yml        Despliegue Docker del runtime
deploy/chatbox/           Configuración de Chatbox Lite
docs/                     Arquitectura y documentación operativa
tools/                    Utilidades de metadatos y recomendación musical
```

## Recomendador de playlists (experimental)

El flujo `item2vec` aprende asociaciones entre canciones a partir de playlists
humanas, agrupa las canciones que ya existen en una biblioteca local y propone
canciones que faltan. Cada propuesta es una afinidad de embedding, **no una
probabilidad ni una garantía de gusto**.

```text
playlists autorizadas -> corpus DuckDB -> embeddings item2vec
                                           |
biblioteca local ------ coincidencias -----+-> comunidades -> playlists y sugerencias
```

El código se ejecuta localmente. Las entradas, vectores, bases de datos,
informes y playlists generadas se ignoran deliberadamente por Git, porque
pueden contener información personal o estar sujetos a licencias de terceros.

Consulta [la guía del MPD](tools/README-spotify-mpd.md) para conocer los
requisitos y el orden de ejecución. Si se usa el MPD, cada persona debe contar
con acceso autorizado y respetar sus términos: este repositorio no concede
derechos sobre ese dataset.

## Arranque del runtime

Requisitos: Docker Compose, una red Docker compartida con el LLM local y un
servidor llama.cpp accesible desde esa red.

```bash
cp .env.example .env
# Edita .env: define MEMORY_ADMIN_TOKEN con un valor aleatorio y revisa LLAMA_API_BASE.
mkdir -p runtime/allowed-data runtime/audit runtime/memory
docker compose --env-file .env -f deploy/compose.yml config
docker compose --env-file .env -f deploy/compose.yml up -d --build
```

El gateway queda publicado en `http://localhost:8090`; consulta
`http://localhost:8090/health` para comprobarlo. Configura Chatbox siguiendo
[su guía local](deploy/chatbox/README.md).

## Información que no se sube

El repositorio público excluye `.env`, secretos, datos de memoria, auditorías,
archivos autorizados para MCP, dependencias descargadas y pesos de modelos.
También excluye los informes y datos de bibliotecas musicales, el MPD, sus
índices y los embeddings. Consulta [la arquitectura](docs/arquitectura.md)
antes de desplegar.

## Licencia

El código publicado bajo este repositorio está bajo la [licencia MIT](LICENSE).
Los datos externos conservan sus propias licencias y condiciones de uso.
