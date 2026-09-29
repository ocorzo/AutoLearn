# Chatbox Lite como frontend

Chatbox Lite es el cliente web de Chaak. No incluimos su código fuente ni los
datos de sus usuarios: Chaak sólo conserva la configuración de integración.

En Chatbox, crea un proveedor compatible con OpenAI y usa:

- **Base URL:** `http://<host-de-chaak>:8090/v1`
- **Modelo:** `chaak` (o el valor de `MODEL_ID`)
- **Clave API:** el valor de `GATEWAY_API_KEY` definido en `.env`. Si esa
  variable está vacía, el gateway no valida la clave y basta una cadena
  cualquiera, por ejemplo `local` (solo aceptable dentro de una red de
  confianza). Los paneles `/control` y `/activity` piden la misma clave una
  vez y la guardan en el navegador.

Chatbox puede publicarse en el puerto 3084, como en el despliegue actual. El
gateway se publica en el 8090 y expone una API compatible con OpenAI. Para
acceso fuera de la red local, pon ambos detrás de HTTPS y de una capa de
autenticación; no expongas el MCP ni el almacenamiento de memoria directamente.

No guardes en este repositorio exportaciones de conversaciones, perfiles de
Chatbox ni contraseñas.
