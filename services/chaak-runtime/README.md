# Runtime de Chaak

Este directorio reúne los componentes propios del runtime:

- `app_server.py`: gateway OpenAI-compatible, streaming, perfiles de
  razonamiento, auditoría y endpoints de actividad.
- `mcp_server.py`: herramientas MCP de archivos, cálculo, Python, fecha y
  navegación web restringida.
- `memory_service.py`: memoria persistente Mem0 con backend Chroma local.
- `Dockerfile.agent` y `Dockerfile.mcp`: imágenes separadas del gateway y MCP.

La definición de despliegue está en `../../deploy/compose.yml`. Los datos de
ejecución se montan en `runtime/` y están excluidos de Git.
