from datetime import datetime, timezone
import asyncio
from pathlib import Path
import ast, fnmatch, ipaddress, json, operator, os, platform, re, resource, socket, subprocess, sys, threading
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import quote_plus, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from mcp.server.fastmcp import FastMCP
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

mcp = FastMCP("bigone-tools")
ALLOWED_ROOT = Path(os.environ.get("ALLOWED_ROOT", "/data"))
AUDIT_LOG = Path(os.environ.get("AUDIT_LOG", "/audit/activity.jsonl"))
_AUDIT_LOCK = threading.Lock()

def _audit_event(kind: str, **details) -> None:
    """Append a small, structured event; never store file contents here."""
    event = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **details}
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[MCP audit] could not write event: {exc}", flush=True)

def _trace(tool: str, **details) -> None:
    """Emit a compact, human-readable audit line to the MCP container log."""
    fields = " ".join(f"{key}={value!r}" for key, value in details.items())
    print(f"[MCP tool] {tool}" + (f" {fields}" if fields else ""), flush=True)
    _audit_event("tool", tool=tool, details=details)

def _safe_path(name: str) -> Path:
    root = ALLOWED_ROOT.resolve(); candidate = (root / name).resolve()
    if candidate != root and root not in candidate.parents: raise ValueError("ruta fuera del directorio permitido")
    return candidate

def _validate_public_url(url: str) -> str:
    """Accept only ordinary public HTTP(S) destinations, never local networks."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("solo se permiten URLs públicas HTTP o HTTPS")
    if parsed.port not in (None, 80, 443):
        raise ValueError("solo se permiten puertos web estándar")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("no fue posible resolver el sitio") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("la navegación a redes privadas o localhost está bloqueada")
    return url

class _PublicOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def _fetch_public_text(url: str, max_chars: int) -> tuple[str, str]:
    _validate_public_url(url)
    opener = build_opener(ProxyHandler({}), _PublicOnlyRedirect())
    request = Request(url, headers={"User-Agent": "Balam-MCP/1.0 (+local research)"})
    try:
        with opener.open(request, timeout=10) as response:
            _validate_public_url(response.geturl())
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "text/plain", "application/json"}:
                raise ValueError("el sitio no devolvió texto legible")
            raw = response.read(min(max_chars * 12, 250_000))
            charset = response.headers.get_content_charset() or "utf-8"
            return response.geturl(), raw.decode(charset, errors="replace")
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError("no fue posible consultar el sitio web") from exc

def _fetch_public_json(url: str, max_chars: int = 30_000) -> tuple[str, dict]:
    final_url, text = _fetch_public_text(url, max_chars)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("el servicio público no devolvió JSON válido") from exc
    if not isinstance(payload, dict):
        raise ValueError("el servicio público devolvió una respuesta inesperada")
    return final_url, payload

def _browser_context(page):
    """Abort navigations/subresources that resolve to private destinations."""
    def guard(route):
        target = route.request.url
        if target.startswith(("http://", "https://")):
            try:
                _validate_public_url(target)
            except ValueError:
                route.abort(); return
        route.continue_()
    page.route("**/*", guard)

def _browser_page():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        executable_path="/usr/bin/chromium", headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    context = browser.new_context(
        user_agent="Balam-MCP/1.0 (+public browser)",
        java_script_enabled=True,
        service_workers="block",
    )
    page = context.new_page(); _browser_context(page)
    return pw, browser, context, page

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.title = ""; self._in_title = False; self._skip = 0
    def handle_starttag(self, tag, attrs):
        self._in_title = tag == "title"
        if tag in {"script", "style", "noscript"}: self._skip += 1
    def handle_endtag(self, tag):
        if tag == "title": self._in_title = False
        if tag in {"script", "style", "noscript"} and self._skip: self._skip -= 1
    def handle_data(self, data):
        text = " ".join(data.split())
        if not text or self._skip: return
        if self._in_title: self.title += (" " if self.title else "") + text
        self.parts.append(text)

_PYTHON_RUNNER = r'''
import ast, builtins, contextlib, io, json, sys
code = json.loads(sys.stdin.read())
allowed_calls = {"print", "len", "sum", "min", "max", "range", "round", "sorted", "abs", "enumerate", "zip", "str", "int", "float", "list", "dict", "set", "tuple"}
allowed_nodes = (ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.If, ast.For, ast.Break, ast.Continue, ast.Pass, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.Call, ast.keyword, ast.JoinedStr, ast.FormattedValue)
tree = ast.parse(code, mode="exec")
for node in ast.walk(tree):
    if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)):
        continue
    if not isinstance(node, allowed_nodes):
        raise ValueError("esa construcción de Python no está permitida")
    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ValueError("los nombres internos no están permitidos")
    if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls):
        raise ValueError("solo se permiten llamadas a funciones básicas")
safe_builtins = {name: getattr(builtins, name) for name in allowed_calls}
scope = {"__builtins__": safe_builtins}
out = io.StringIO()
with contextlib.redirect_stdout(out):
    exec(compile(tree, "<balam-python>", "exec"), scope, scope)
print(out.getvalue(), end="")
'''

def _python_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024, 256 * 1024))

@mcp.tool()
def run_python(code: str) -> dict:
    """Ejecuta Python básico aislado para cálculos y transformación de datos. Sin imports, red, archivos ni comandos."""
    if len(code) > 12_000: raise ValueError("el código excede 12,000 caracteres")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _PYTHON_RUNNER], input=json.dumps(code), text=True,
            capture_output=True, timeout=4, preexec_fn=_python_limits,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("el código superó el límite de 3 segundos") from exc
    if completed.returncode != 0:
        error = completed.stderr.strip().splitlines()[-1:] or ["error de ejecución"]
        raise ValueError(error[0][:500])
    output = completed.stdout[:8_000] or "(sin salida; usa print para mostrar un resultado)"
    _trace("run_python", code_chars=len(code), output_chars=len(output))
    return {"output": output, "limits": "3 s CPU, 128 MiB; sin imports, red ni archivos"}

@mcp.tool()
async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Busca en la web pública con Chromium local. Úsala solo si la persona pidió explícitamente buscar información en internet."""
    # DuckDuckGo's static endpoint now returns an anti-bot challenge to this
    # server, which looks like valid HTML but contains no results.  Chromium
    # is already bundled with this MCP service and Bing works reliably through
    # it, so use the local-browser implementation as the one search path.
    # FastMCP invokes tools inside an asyncio loop.  Playwright's sync API
    # cannot run in that loop, so keep the browser work in a worker thread.
    result = await asyncio.to_thread(_browser_search_sync, query, max_results)
    _trace("search_web", query_chars=len(query), count=len(result))
    return result

@mcp.tool()
def get_weather(location: str = "", hours: int = 12) -> dict:
    """Devuelve el pronóstico horario actual de una ciudad mediante Open-Meteo, sin API key.

    Indica la ciudad salvo que ya conozcas una ubicación del usuario en el contexto.
    """
    location = location.strip() or os.environ.get("WEATHER_DEFAULT_LOCATION", "").strip()
    if not location:
        raise ValueError("indica la ciudad para consultar el pronóstico")
    hours = min(max(int(hours), 1), 24)
    _, places = _fetch_public_json(
        "https://geocoding-api.open-meteo.com/v1/search?name=" + quote_plus(location) + "&count=1&language=es&format=json"
    )
    matches = places.get("results") or []
    if not matches or not isinstance(matches[0], dict):
        raise ValueError("no encontré esa ciudad para el pronóstico")
    place = matches[0]
    latitude, longitude = place.get("latitude"), place.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("el servicio meteorológico no devolvió coordenadas válidas")
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?latitude=" + str(latitude)
        + "&longitude=" + str(longitude)
        + "&current=temperature_2m,precipitation,weather_code"
        + "&hourly=temperature_2m,precipitation_probability,precipitation,weather_code"
        + "&forecast_days=2&timezone=auto"
    )
    _, forecast = _fetch_public_json(forecast_url)
    hourly = forecast.get("hourly") or {}
    timestamps = hourly.get("time") or []
    current_time = str((forecast.get("current") or {}).get("time") or "")
    start = next((index for index, stamp in enumerate(timestamps) if str(stamp) >= current_time), 0)
    periods = []
    for index in range(start, min(start + hours, len(timestamps))):
        periods.append({
            "time": timestamps[index],
            "temperature_c": (hourly.get("temperature_2m") or [None])[index],
            "precipitation_probability": (hourly.get("precipitation_probability") or [None])[index],
            "precipitation_mm": (hourly.get("precipitation") or [None])[index],
            "weather_code": (hourly.get("weather_code") or [None])[index],
        })
    result = {
        "location": {
            "name": place.get("name"),
            "admin1": place.get("admin1"),
            "country": place.get("country"),
            "latitude": latitude,
            "longitude": longitude,
        },
        "timezone": forecast.get("timezone"),
        "current": forecast.get("current"),
        "next_hours": periods,
    }
    _trace("get_weather", location=location, hours=len(periods))
    return result

@mcp.tool()
def read_web_page(url: str, max_chars: int = 12_000) -> dict:
    """Lee texto de una página web pública HTTP(S). Bloquea localhost y redes privadas."""
    max_chars = min(max(int(max_chars), 500), 30_000)
    final_url, page = _fetch_public_text(url, max_chars)
    parser = _TextExtractor(); parser.feed(page)
    text = "\n".join(parser.parts)
    result = {"url": final_url, "title": parser.title or "(sin título)", "text": text[:max_chars]}
    _trace("read_web_page", host=urlparse(final_url).hostname, returned_chars=len(result["text"]))
    return result

def _browser_read_page_sync(url: str, max_chars: int = 12_000) -> dict:
    """Lee una página pública con Chromium headless para sitios que requieren JavaScript."""
    max_chars = min(max(int(max_chars), 500), 30_000)
    _validate_public_url(url)
    pw = browser = context = page = None
    try:
        pw, browser, context, page = _browser_page()
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_timeout(500)
        result = {"url": page.url, "title": page.title() or "(sin título)", "text": page.locator("body").inner_text(timeout=5_000)[:max_chars]}
        _trace("browser_read_page", host=urlparse(page.url).hostname, returned_chars=len(result["text"]))
        return result
    except (PlaywrightTimeoutError, ValueError) as exc:
        raise ValueError("no fue posible renderizar la página pública") from exc
    finally:
        if context: context.close()
        if browser: browser.close()
        if pw: pw.stop()

def _browser_search_sync(query: str, max_results: int = 5) -> list[dict]:
    """Busca en Bing mediante Chromium headless y devuelve resultados públicos."""
    if not query.strip(): raise ValueError("la consulta no puede estar vacía")
    max_results = min(max(int(max_results), 1), 10)
    pw = browser = context = page = None
    try:
        pw, browser, context, page = _browser_page()
        page.goto("https://www.bing.com/search?q=" + quote_plus(query), wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_timeout(500)
        rows = page.locator("li.b_algo h2 a").evaluate_all("els => els.map(a => ({title:a.innerText, url:a.href}))")
        result = [{"title": str(row.get("title", "")).strip(), "url": row.get("url", "")} for row in rows[:max_results] if row.get("url", "").startswith(("http://", "https://"))]
        _trace("browser_search", query_chars=len(query), count=len(result))
        return result
    except (PlaywrightTimeoutError, ValueError) as exc:
        raise ValueError("no fue posible realizar la búsqueda con el navegador") from exc
    finally:
        if context: context.close()
        if browser: browser.close()
        if pw: pw.stop()

@mcp.tool()
async def browser_read_page(url: str, max_chars: int = 12_000) -> dict:
    """Lee una página pública con Chromium headless para sitios que requieren JavaScript."""
    return await asyncio.to_thread(_browser_read_page_sync, url, max_chars)

@mcp.tool()
async def browser_search(query: str, max_results: int = 5) -> list[dict]:
    """Busca en Bing mediante Chromium headless y devuelve resultados públicos."""
    return await asyncio.to_thread(_browser_search_sync, query, max_results)

@mcp.tool()
def get_datetime() -> dict:
    """Devuelve la fecha y hora UTC actual en formato estructurado."""
    result = {"datetime": datetime.now(timezone.utc).isoformat(), "timezone": "UTC"}
    _trace("get_datetime")
    return result

@mcp.tool()
def get_system_info() -> dict:
    """Devuelve información básica del sistema MCP."""
    result = {"system": platform.system(), "release": platform.release(), "python": platform.python_version()}
    _trace("get_system_info")
    return result

@mcp.tool()
def calculate(expression: str) -> float | int:
    """Evalúa una expresión aritmética simple."""
    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd)
    tree = ast.parse(expression, mode="eval")
    if any(not isinstance(node, allowed) for node in ast.walk(tree)): raise ValueError("solo operaciones aritméticas")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.UnaryOp): return (-1 if isinstance(node.op, ast.USub) else 1) * ev(node.operand)
        if isinstance(node, ast.BinOp): return ops[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("expresión no permitida")
    result = ev(tree)
    _trace("calculate", expression=expression, result=result)
    return result

@mcp.tool()
def list_files(subdir: str = "") -> list[str]:
    """Lista archivos del directorio permitido."""
    base = _safe_path(subdir)
    if not base.is_dir(): raise ValueError("no es un directorio")
    result = sorted(str(p.relative_to(ALLOWED_ROOT)) for p in base.iterdir())[:200]
    _trace("list_files", subdir=subdir, count=len(result))
    return result

@mcp.tool()
def list_directory_tree(subdir: str = "", max_depth: int = 3, max_entries: int = 200) -> list[str]:
    """Muestra un árbol de directorios dentro de /data, sin leer contenidos."""
    base = _safe_path(subdir)
    if not base.is_dir(): raise ValueError("no es un directorio")
    max_depth = min(max(int(max_depth), 0), 6)
    max_entries = min(max(int(max_entries), 1), 500)
    result = []
    for path in sorted(base.rglob("*")):
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        if len(relative.parts) > max_depth or len(result) >= max_entries:
            continue
        suffix = "/" if path.is_dir() else ""
        result.append(str(relative) + suffix)
    _trace("list_directory_tree", subdir=subdir, depth=max_depth, count=len(result))
    return result

@mcp.tool()
def find_files(pattern: str, subdir: str = "", max_results: int = 100) -> list[str]:
    """Busca nombres de archivo por patrón, por ejemplo '*.pdf' o 'reporte*'."""
    base = _safe_path(subdir)
    if not base.is_dir(): raise ValueError("no es un directorio")
    max_results = min(max(int(max_results), 1), 300)
    result = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and fnmatch.fnmatch(path.name.lower(), pattern.lower()):
            result.append(str(path.relative_to(ALLOWED_ROOT)))
            if len(result) >= max_results: break
    _trace("find_files", pattern=pattern, subdir=subdir, count=len(result))
    return result

@mcp.tool()
def file_metadata(path: str) -> dict:
    """Devuelve tipo, tamaño y fecha de modificación de un archivo permitido."""
    p = _safe_path(path)
    if not p.exists(): raise ValueError("archivo inexistente")
    stat = p.stat()
    result = {
        "path": str(p.relative_to(ALLOWED_ROOT)),
        "type": "directory" if p.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    _trace("file_metadata", path=path)
    return result

@mcp.tool()
def read_file(path: str, max_chars: int = 12000) -> str:
    """Lee un archivo de texto permitido."""
    p = _safe_path(path)
    if not p.is_file(): raise ValueError("archivo inexistente")
    result = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    _trace("read_file", path=path, returned_chars=len(result))
    return result

@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = True) -> dict:
    """Crea o reemplaza un archivo de texto dentro de /data."""
    p = _safe_path(path)
    if p == ALLOWED_ROOT.resolve() or p.is_dir():
        raise ValueError("la ruta debe ser un archivo, no un directorio")
    if not overwrite and p.exists():
        raise ValueError("el archivo ya existe; usa overwrite=true para reemplazarlo")
    if len(content) > 500_000:
        raise ValueError("el contenido supera el límite de 500000 caracteres")
    if not p.parent.is_dir():
        raise ValueError("el directorio padre no existe")
    p.write_text(content, encoding="utf-8")
    _trace("write_file", path=path, chars=len(content), overwrite=overwrite)
    return {"status": "written", "path": str(p.relative_to(ALLOWED_ROOT)), "size_bytes": p.stat().st_size}

@mcp.tool()
def replace_in_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> dict:
    """Modifica un archivo reemplazando texto exacto dentro de /data."""
    p = _safe_path(path)
    if not p.is_file():
        raise ValueError("archivo inexistente")
    if not old_text:
        raise ValueError("old_text no puede estar vacío")
    content = p.read_text(encoding="utf-8")
    matches = content.count(old_text)
    if matches == 0:
        raise ValueError("no se encontró old_text en el archivo")
    if not replace_all and matches > 1:
        raise ValueError("old_text aparece varias veces; usa replace_all=true o un texto más específico")
    updated = content.replace(old_text, new_text, -1 if replace_all else 1)
    p.write_text(updated, encoding="utf-8")
    _trace("replace_in_file", path=path, matches=matches, replaced=matches if replace_all else 1)
    return {"status": "modified", "path": str(p.relative_to(ALLOWED_ROOT)), "replacements": matches if replace_all else 1}

@mcp.tool()
def delete_file(path: str) -> dict:
    """Borra un archivo dentro de /data; no permite borrar directorios."""
    p = _safe_path(path)
    if not p.is_file():
        raise ValueError("solo se pueden borrar archivos existentes; los directorios están protegidos")
    p.unlink()
    _trace("delete_file", path=path)
    return {"status": "deleted", "path": str(p.relative_to(ALLOWED_ROOT))}

@mcp.tool()
def read_file_lines(path: str, start_line: int = 1, end_line: int = 100) -> list[dict]:
    """Lee un intervalo acotado de líneas de un archivo de texto permitido."""
    p = _safe_path(path)
    if not p.is_file(): raise ValueError("archivo inexistente")
    start_line = max(int(start_line), 1)
    end_line = min(max(int(end_line), start_line), start_line + 499)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    result = [
        {"line": number, "text": text[:2000]}
        for number, text in enumerate(lines[start_line - 1:end_line], start_line)
    ]
    _trace("read_file_lines", path=path, start=start_line, end=end_line, returned=len(result))
    return result

@mcp.tool()
def search_text(query: str, subdir: str = "") -> list[dict]:
    """Busca texto literal en archivos permitidos."""
    base = _safe_path(subdir); results = []
    for p in base.rglob("*"):
        if len(results) >= 100 or not p.is_file(): continue
        try:
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if query.lower() in line.lower(): results.append({"file": str(p.relative_to(ALLOWED_ROOT)), "line": n, "text": line[:500]})
        except OSError: pass
    _trace("search_text", query=query, subdir=subdir, matches=len(results))
    return results

@mcp.tool()
def list_environment() -> dict:
    """Devuelve solo variables explícitamente permitidas."""
    result = {k: os.environ[k] for k in ("MODEL_ID", "ALLOWED_ROOT") if k in os.environ}
    _trace("list_environment")
    return result

@mcp.tool()
def get_local_datetime(timezone_name: str = "America/Mexico_City") -> dict:
    """Devuelve fecha y hora actual en una zona IANA, como America/Mexico_City."""
    try:
        result = {"datetime": datetime.now(ZoneInfo(timezone_name)).isoformat(), "timezone": timezone_name}
    except ZoneInfoNotFoundError as exc:
        raise ValueError("zona horaria IANA no reconocida") from exc
    _trace("get_local_datetime", timezone=timezone_name)
    return result

if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8000
    mcp.run(transport="streamable-http")
