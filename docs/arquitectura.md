# Arquitectura de Chaak

```text
Chatbox Lite (frontend, normalmente :3084)
              |
              | API compatible con OpenAI
              v
Agente-gateway (:8090/v1)
   |              |                 |
   |              |                 +-- Mem0 + Chroma local
   |              |                     (memoria persistente)
   |              |
   |              +-- llama.cpp / LLM local
   |                  (red Docker existente)
   |
   +-- Servidor MCP interno (:8000, no publicado)
       +-- archivos autorizados, cálculo y Python
       +-- navegación web con Chromium + Playwright
```

El gateway es el punto central: recibe las peticiones de Chatbox, entrega las
conversaciones normales al LLM local y usa `smolagents` más MCP cuando una
petición requiere herramientas. La capa Mem0 se integra en el gateway, no como
una herramienta que el modelo deba recordar invocar.

## Límites de seguridad

- MCP no expone un puerto al host; sólo se comunica con el gateway por la red
  Docker.
- Las herramientas de archivos se limitan al volumen `runtime/allowed-data/`.
- Las memorias, auditorías, conversaciones y credenciales son datos locales y
  no se versionan.
- El LLM se referencia mediante `LLAMA_API_BASE`; los pesos/modelos no forman
  parte de este repositorio.

## Componentes versionados

El repositorio contiene el código de AutoLearn y el runtime propio de Chaak.
Chatbox Lite es una dependencia externa: documentamos su configuración en
`deploy/chatbox/README.md`, pero no copiamos su código ni sus datos de usuario.
