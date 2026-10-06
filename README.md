# AutoLearn

AutoLearn es un asistente local compuesto por un frontend Chatbox Lite, un gateway
compatible con OpenAI, herramientas MCP, un LLM ejecutado localmente y memoria
persistente con Mem0/Chroma. También incluye el pipeline `autolearn/` para
entrenamiento y evaluación.

## Estructura

```text
autolearn/                Pipeline de entrenamiento y ensayos
services/chaak-runtime/   Gateway, MCP, Mem0 y Dockerfiles propios
deploy/compose.yml        Despliegue Docker del runtime
deploy/chatbox/           Configuración de Chatbox Lite
docs/                     Arquitectura y documentación operativa
```

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
Consulta [la arquitectura](docs/arquitectura.md) antes de desplegar.
