"""Métricas de consumo y rendimiento (spec §5.6): CONTRATO v2, implementado.

Fija firmas, columnas y semántica para que el dashboard, el benchmark y la interfaz se
desarrollen en paralelo. La implementación la hace Dev-métricas.

Reglas de diseño (aprobadas por el jefe-revisor con cambios):
- Los tiempos por etapa NO se miden aquí: vienen de `Resultado.tiempos_ms` (ya medidos con
  `roadvision.timing.Cronometro` y sincronización CUDA).
- No se edita `pipeline.py`: se acopla desde afuera con `analizar_medido` + `registrar`.
- Sin GPU NVIDIA, torch, nvidia-ml-py o pandas el módulo se importa igual (imports
  perezosos); los campos de GPU quedan en None.
- El calentamiento va por `calentar_medido` con `calentamiento=True`; `leer_csv` lo excluye
  por defecto (anti-meta del spec §2).
- Un solo usuario: `_CANDADO` serializa `analizar_medido` y `escribir_csv`, porque
  `reset_peak_memory_stats` y NVML son globales del dispositivo.

Semántica de las métricas (rotular así en terminal y dashboard):
- `imagenes_por_s` es EQUIVALENTE (1000 / ms_total de una imagen), no throughput real.
- `gpu_pct_*` y `vram_usada_mb` son del DISPOSITIVO completo (incluye otros procesos);
  `vram_usada_mb` es el PICO de VRAM usada del dispositivo durante el análisis.
- `vram_pico_torch_mb` usa `max_memory_allocated` (excluye la caché del asignador).
- `cpu_pct_prom` puede superar 100 % (cuenta varios núcleos).

Notas de implementación:
- `MuestreadorSistema.cpu_pct_intervalo` es el `cpu_percent()` de todo el intervalo (se
  fija en `__exit__`); los llamadores lo pasan a `construir_metricas(cpu_pct_prom=...)`.
- `vram_usada_mb` de la fila es el PICO de VRAM usada del dispositivo durante el análisis.
"""
from __future__ import annotations

import atexit
import csv
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from PIL import Image
    from rich.console import Console

    from roadvision.config import Umbrales
    from roadvision.contracts import Resultado
    from roadvision.pipeline import Pipeline

# Anclada a la raíz del repo: app.py, la CLI y el dashboard leen y escriben el mismo archivo
# sin importar desde qué carpeta se lancen.
RUTA_CSV = Path(__file__).resolve().parent.parent / "logs" / "metricas.csv"
INTERVALO_MUESTREO_S = 0.1
MIN_MUESTRAS_FIABLES = 3  # reporte_terminal avisa si hay menos

# Copia de pipeline.ETAPAS (importarlo arrastra torch); un test verifica que coinciden.
ETAPAS = ("total", "preproceso", "deteccion", "clasificacion", "respaldo")

COLUMNAS_CSV = (
    "marca_tiempo", "dispositivo", "precision", "calentamiento",
    "ancho", "alto",
    "detecciones", "reconocidas", "aproximadas", "solo_categoria", "no_reconocidas",
    "ms_total", "ms_preproceso", "ms_deteccion", "ms_clasificacion", "ms_respaldo",
    "imagenes_por_s",
    "muestras_gpu", "gpu_pct_pico", "gpu_pct_prom", "vram_usada_mb", "vram_pico_torch_mb",
    "temperatura_c_max", "potencia_w_prom",
    "cpu_pct_prom", "ram_proceso_mb_pico",
)

_CANDADO = threading.Lock()
_registro = logging.getLogger(__name__)

# --- NVML: singleton perezoso por proceso -------------------------------------------------
_CANDADO_NVML = threading.Lock()
_nvml_intentado = False
_nvml_modulo: Any = None


def _cerrar_nvml() -> None:
    try:
        if _nvml_modulo is not None:
            _nvml_modulo.nvmlShutdown()
    except Exception:  # noqa: BLE001 - al salir del proceso no debe fallar nada
        pass


def _obtener_nvml() -> Any:
    """Devuelve el módulo `pynvml` inicializado, o None si no hay NVIDIA/NVML.

    `nvmlInit` se llama una sola vez por proceso; `nvmlShutdown` queda en `atexit`.
    """
    global _nvml_intentado, _nvml_modulo
    with _CANDADO_NVML:
        if _nvml_intentado:
            return _nvml_modulo
        _nvml_intentado = True
        try:
            import pynvml
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
        except (pynvml.NVMLError, OSError):
            return None
        _nvml_modulo = pynvml
        atexit.register(_cerrar_nvml)
        return _nvml_modulo


def _resolver_handle(nvml: Any, indice_gpu: int) -> Any:
    """Handle NVML de la GPU. Con torch cargado se busca por PCI bus id (varias GPUs)."""
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                bus_id = f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0"
                return nvml.nvmlDeviceGetHandleByPciBusId(bus_id)
        except Exception:  # noqa: BLE001 - cae al índice
            pass
    return nvml.nvmlDeviceGetHandleByIndex(indice_gpu)


@dataclass(frozen=True)
class MuestraSistema:
    """Una lectura instantánea. Campos de GPU en None si no hay NVIDIA."""

    t_s: float  # time.perf_counter()
    cpu_pct: float
    ram_proceso_mb: float
    gpu_pct: float | None = None
    vram_usada_mb: float | None = None
    temperatura_c: float | None = None
    potencia_w: float | None = None


@dataclass(frozen=True)
class MetricasAnalisis:
    """Una fila del CSV (plana: un campo por columna de `COLUMNAS_CSV`).

    `detecciones == reconocidas + aproximadas + solo_categoria + no_reconocidas` y se cuenta
    como `len(resultado.senales)` (después del filtrado de cajas).
    """

    marca_tiempo: str  # datetime.now().astimezone().isoformat(timespec="milliseconds")
    dispositivo: str
    precision: str  # "fp16" | "fp32", de pipeline.dispositivo.dtype
    calentamiento: bool
    ancho: int
    alto: int
    detecciones: int
    reconocidas: int
    aproximadas: int
    solo_categoria: int
    no_reconocidas: int
    ms_total: float | None = None  # None en el calentamiento
    ms_preproceso: float | None = None
    ms_deteccion: float | None = None
    ms_clasificacion: float | None = None
    ms_respaldo: float | None = None
    imagenes_por_s: float | None = None
    muestras_gpu: int = 0
    gpu_pct_pico: float | None = None
    gpu_pct_prom: float | None = None
    vram_usada_mb: float | None = None
    vram_pico_torch_mb: float | None = None
    temperatura_c_max: float | None = None
    potencia_w_prom: float | None = None
    cpu_pct_prom: float | None = None
    ram_proceso_mb_pico: float | None = None

    def a_fila(self) -> dict[str, object]:
        """`{c: getattr(self, c) for c in COLUMNAS_CSV}`. Test: campos == COLUMNAS_CSV."""
        return {c: getattr(self, c) for c in COLUMNAS_CSV}


class MuestreadorSistema:
    """Context manager con un hilo que lee psutil y NVML cada `intervalo_s`.

    Semántica obligatoria:
    - Toma una muestra al entrar y otra al salir además de las del hilo.
    - CPU %: preparar `cpu_percent()` en `__enter__`; el promedio es `cpu_percent()` en
      `__exit__` (intervalo completo; la primera llamada devuelve 0.0).
    - Hilo `daemon=True`, espera con `threading.Event().wait(intervalo)`, `join(timeout=1)`;
      cualquier excepción del hilo apaga las métricas de GPU sin propagarse; `__exit__`
      devuelve None y no tapa excepciones del bloque.
    - NVML: `nvmlInit` una vez por proceso (singleton perezoso con candado, `nvmlShutdown`
      en `atexit`). Captura NVMLError, ImportError y OSError → `gpu_disponible=False`.
    - Varias GPUs: resolver el handle por PCI bus id con
      `torch.cuda.get_device_properties(0)` si torch está cargado; si no, `indice_gpu`.
    """

    def __init__(
        self,
        intervalo_s: float = INTERVALO_MUESTREO_S,
        *,
        indice_gpu: int = 0,
        max_muestras: int | None = None,
    ) -> None:
        """`max_muestras` acota la memoria en usos de larga duración (monitor en vivo):
        al llenarse se descartan las más antiguas. None = sin límite (un análisis).
        """
        self._intervalo_s = intervalo_s
        self._indice_gpu = indice_gpu
        self._muestras: deque[MuestraSistema] = deque(maxlen=max_muestras)
        self._cerrojo = threading.Lock()
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._gpu_activa = False
        self._proc: Any = None  # su cpu_percent() cubre todo el intervalo
        self._proc_hilo: Any = None  # cpu_percent() instantáneo de las muestras del hilo
        self._cpu_pct_intervalo: float | None = None

    @property
    def cpu_pct_intervalo(self) -> float | None:
        return self._cpu_pct_intervalo

    @property
    def muestras(self) -> list[MuestraSistema]:
        with self._cerrojo:
            return list(self._muestras)

    @property
    def gpu_disponible(self) -> bool:
        return self._gpu_activa

    def __enter__(self) -> MuestreadorSistema:
        import psutil

        with self._cerrojo:
            self._muestras.clear()
        self._cpu_pct_intervalo = None
        self._proc = psutil.Process()
        self._proc_hilo = psutil.Process()
        self._proc.cpu_percent()  # prepara el intervalo (la primera llamada da 0.0)
        self._proc_hilo.cpu_percent()
        self._iniciar_gpu()
        self._parar.clear()
        self._agregar(self._tomar(self._proc_hilo.cpu_percent()))
        self._hilo = threading.Thread(
            target=self._bucle, name="muestreador-sistema", daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=1)
        # La muestra final lleva el promedio de CPU de todo el intervalo.
        try:
            cpu = self._proc.cpu_percent()
            self._cpu_pct_intervalo = cpu
            self._agregar(self._tomar(cpu))
        except Exception:  # noqa: BLE001 - las métricas nunca tapan la excepción del bloque
            _registro.warning("No se pudo tomar la muestra final de métricas", exc_info=True)
        return None

    def _iniciar_gpu(self) -> None:
        self._gpu_activa = False
        self._nvml = self._handle = None
        nvml = _obtener_nvml()
        if nvml is None:
            return
        try:
            self._handle = _resolver_handle(nvml, self._indice_gpu)
        except (nvml.NVMLError, OSError):
            return
        self._nvml = nvml
        self._gpu_activa = True

    def _agregar(self, muestra: MuestraSistema) -> None:
        with self._cerrojo:
            self._muestras.append(muestra)

    def _bucle(self) -> None:
        try:
            while not self._parar.wait(self._intervalo_s):
                self._agregar(self._tomar(self._proc_hilo.cpu_percent()))
        except Exception:  # noqa: BLE001 - un fallo del hilo apaga la GPU, no la app
            self._gpu_activa = False

    def _tomar(self, cpu_pct: float) -> MuestraSistema:
        ram = self._proc.memory_info().rss / 2**20
        gpu = vram = temp = pot = None
        if self._gpu_activa:
            nvml, h = self._nvml, self._handle
            try:
                gpu = float(nvml.nvmlDeviceGetUtilizationRates(h).gpu)
                vram = nvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
            except (nvml.NVMLError, OSError):
                self._gpu_activa = False
                gpu = vram = None
            if self._gpu_activa:
                try:
                    temp = float(nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU))
                except (nvml.NVMLError, OSError):
                    temp = None
                try:
                    pot = nvml.nvmlDeviceGetPowerUsage(h) / 1000
                except (nvml.NVMLError, OSError):
                    pot = None
        return MuestraSistema(
            t_s=time.perf_counter(), cpu_pct=float(cpu_pct), ram_proceso_mb=ram,
            gpu_pct=gpu, vram_usada_mb=vram, temperatura_c=temp, potencia_w=pot)


def _promedio(valores: list[float]) -> float | None:
    return sum(valores) / len(valores) if valores else None


def _maximo(valores: list[float]) -> float | None:
    return max(valores) if valores else None


def _ahora() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _precision(pipeline: Pipeline) -> str:
    d = str(pipeline.dispositivo.dtype)
    return {"torch.float16": "fp16", "torch.float32": "fp32"}.get(d, d.removeprefix("torch."))


def construir_metricas(
    resultado: Resultado,
    muestras: list[MuestraSistema],
    *,
    precision: str,
    vram_pico_torch_mb: float | None = None,
    cpu_pct_prom: float | None = None,
) -> MetricasAnalisis:
    """Función PURA y testeable: resume `muestras` (pico/promedio) y copia los tiempos."""
    conteo = {"reconocida": 0, "aproximada": 0, "solo_categoria": 0, "no_reconocida": 0}
    for senal in resultado.senales:
        conteo[senal.estado.value] += 1
    t = resultado.tiempos_ms
    ms_total = t.get("total")
    con_gpu = [m for m in muestras if m.gpu_pct is not None]
    return MetricasAnalisis(
        marca_tiempo=_ahora(),
        dispositivo=resultado.dispositivo,
        precision=precision,
        calentamiento=False,
        ancho=resultado.ancho,
        alto=resultado.alto,
        detecciones=len(resultado.senales),
        reconocidas=conteo["reconocida"],
        aproximadas=conteo["aproximada"],
        solo_categoria=conteo["solo_categoria"],
        no_reconocidas=conteo["no_reconocida"],
        ms_total=ms_total,
        ms_preproceso=t.get("preproceso"),
        ms_deteccion=t.get("deteccion"),
        ms_clasificacion=t.get("clasificacion"),
        ms_respaldo=t.get("respaldo"),
        imagenes_por_s=1000 / ms_total if ms_total else None,
        muestras_gpu=len(con_gpu),
        gpu_pct_pico=_maximo([m.gpu_pct for m in con_gpu]),
        gpu_pct_prom=_promedio([m.gpu_pct for m in con_gpu]),
        vram_usada_mb=_maximo(
            [m.vram_usada_mb for m in muestras if m.vram_usada_mb is not None]),
        vram_pico_torch_mb=vram_pico_torch_mb,
        temperatura_c_max=_maximo(
            [m.temperatura_c for m in muestras if m.temperatura_c is not None]),
        potencia_w_prom=_promedio(
            [m.potencia_w for m in muestras if m.potencia_w is not None]),
        cpu_pct_prom=cpu_pct_prom,
        ram_proceso_mb_pico=_maximo([m.ram_proceso_mb for m in muestras]),
    )


def analizar_medido(
    pipeline: Pipeline,
    imagen: Image.Image,
    umbrales: Umbrales | None = None,
) -> tuple[Resultado, MetricasAnalisis]:
    """Envuelve `pipeline.analizar` (bajo `_CANDADO`).

    Orden: `reset_peak_memory_stats()` (solo si `pipeline.dispositivo.nombre == "cuda"` y
    torch ya está en `sys.modules`), `analizar` dentro de un `MuestreadorSistema`, luego
    `max_memory_allocated() / 2**20`. No importa torch de forma obligatoria.
    """
    with _CANDADO:
        torch = sys.modules.get("torch") if pipeline.dispositivo.nombre == "cuda" else None
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        with MuestreadorSistema() as muestreador:
            resultado = pipeline.analizar(imagen, umbrales)
        vram_pico = torch.cuda.max_memory_allocated() / 2**20 if torch is not None else None
        metricas = construir_metricas(
            resultado, muestreador.muestras, precision=_precision(pipeline),
            vram_pico_torch_mb=vram_pico, cpu_pct_prom=muestreador.cpu_pct_intervalo)
    return resultado, metricas


def calentar_medido(pipeline: Pipeline) -> MetricasAnalisis:
    """Llama `pipeline.calentar()`; fila con `calentamiento=True`, `ms_total`,
    `ms_preproceso` e `imagenes_por_s` en None, conteos 0, ancho/alto 640x480 (imagen
    sintética). Documentar: no ejecuta el decisor; `respaldo` cubre una sola pasada de CLIP.
    """
    with _CANDADO:
        torch = sys.modules.get("torch") if pipeline.dispositivo.nombre == "cuda" else None
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        with MuestreadorSistema() as muestreador:
            tiempos = pipeline.calentar()
        vram_pico = torch.cuda.max_memory_allocated() / 2**20 if torch is not None else None
    base = construir_metricas(
        _resultado_vacio(pipeline, tiempos), muestreador.muestras,
        precision=_precision(pipeline), vram_pico_torch_mb=vram_pico,
        cpu_pct_prom=muestreador.cpu_pct_intervalo)
    return _con(base, calentamiento=True, ms_total=None, ms_preproceso=None,
                imagenes_por_s=None)


def _resultado_vacio(pipeline: Pipeline, tiempos: dict[str, float]) -> Resultado:
    from roadvision.contracts import Resultado

    return Resultado(
        senales=[], conteo_por_nombre={}, conteo_por_categoria={}, tiempos_ms=dict(tiempos),
        ancho=640, alto=480, dispositivo=pipeline.dispositivo.descripcion)


def _con(m: MetricasAnalisis, **cambios: object) -> MetricasAnalisis:
    from dataclasses import replace

    return replace(m, **cambios)


def registrar(metricas: MetricasAnalisis, ruta: Path = RUTA_CSV) -> None:
    """`reporte_terminal` + `escribir_csv`. Un fallo de escritura avisa y no rompe la app."""
    try:
        reporte_terminal(metricas)
    except Exception:  # noqa: BLE001
        _registro.warning("No se pudo mostrar el reporte de métricas", exc_info=True)
    try:
        escribir_csv(metricas, ruta)
    except Exception:  # noqa: BLE001
        _registro.warning("No se pudo registrar la métrica en %s", ruta, exc_info=True)


def _cabecera_de(ruta: Path) -> list[str] | None:
    """Primera fila del CSV, o None si no existe o está vacío."""
    if not ruta.exists() or ruta.stat().st_size == 0:
        return None
    with ruta.open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f), None)


def escribir_csv(metricas: MetricasAnalisis, ruta: Path = RUTA_CSV) -> None:
    """Agrega una fila bajo `_CANDADO`. `csv.DictWriter(f, COLUMNAS_CSV)`, `newline=""`,
    utf-8, None → celda vacía. Crea carpeta y cabecera si faltan. Si la cabecera existente
    difiere de `COLUMNAS_CSV`, archiva el viejo como `<nombre>_<fecha-hora>.csv` con aviso y
    empieza uno nuevo en `ruta` (que sigue siendo el archivo canónico). Captura
    `PermissionError` (CSV abierto en Excel) con aviso, sin lanzar hacia la UI.
    """
    with _CANDADO:
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            cabecera = _cabecera_de(ruta)
            if cabecera is not None and cabecera != list(COLUMNAS_CSV):
                archivo = ruta.with_name(f"{ruta.stem}_{datetime.now():%Y%m%d-%H%M%S}.csv")
                ruta.replace(archivo)
                _registro.warning(
                    "La cabecera de %s no coincide; se archivó como %s y se empieza uno nuevo",
                    ruta, archivo)
                cabecera = None
            nuevo = cabecera is None
            with ruta.open("a", newline="", encoding="utf-8") as f:
                escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
                if nuevo:
                    escritor.writeheader()
                escritor.writerow({
                    c: ("" if v is None else v) for c, v in metricas.a_fila().items()})
        except PermissionError:
            _registro.warning(
                "No se pudo escribir %s (¿está abierto en Excel?). Se omite esta fila.", ruta)


def leer_csv(
    ruta: Path = RUTA_CSV, *, incluir_calentamiento: bool = False
) -> pd.DataFrame:
    """DataFrame con `COLUMNAS_CSV`; vacío si el archivo no existe. Por defecto excluye
    `calentamiento == True` (el dashboard y el benchmark nunca lo mezclan).
    """
    import pandas as pd

    if not ruta.exists():
        return pd.DataFrame(columns=list(COLUMNAS_CSV))
    try:
        df = pd.read_csv(ruta, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(COLUMNAS_CSV))
    df = df.reindex(columns=list(COLUMNAS_CSV))
    df["calentamiento"] = df["calentamiento"].astype(str).str.strip().str.lower() == "true"
    if not incluir_calentamiento:
        df = df[~df["calentamiento"]].reset_index(drop=True)
    return df


def _fmt(valor: float | None, unidad: str = "", decimales: int = 1) -> str:
    return "n/d" if valor is None else f"{valor:.{decimales}f}{unidad}"


def reporte_terminal(
    metricas: MetricasAnalisis, *, consola: Console | None = None
) -> None:
    """Tabla `rich`. Rotula "equivalente" y "dispositivo"; avisa si
    `muestras_gpu < MIN_MUESTRAS_FIABLES`.
    """
    from rich.console import Console
    from rich.table import Table

    consola = consola or Console()
    m = metricas
    tabla = Table(title="Métricas del análisis" + (" (calentamiento)" if m.calentamiento else ""))
    tabla.add_column("Métrica")
    tabla.add_column("Valor", justify="right")
    filas = [
        ("Dispositivo / precisión", f"{m.dispositivo} / {m.precision}"),
        ("Imagen", f"{m.ancho}x{m.alto}"),
        ("Detecciones (rec/apr/cat/no)",
         f"{m.detecciones} ({m.reconocidas}/{m.aproximadas}/{m.solo_categoria}/"
         f"{m.no_reconocidas})"),
        ("Tiempo total", _fmt(m.ms_total, " ms")),
        ("  preproceso", _fmt(m.ms_preproceso, " ms")),
        ("  detección", _fmt(m.ms_deteccion, " ms")),
        ("  clasificación", _fmt(m.ms_clasificacion, " ms")),
        ("  respaldo", _fmt(m.ms_respaldo, " ms")),
        ("Imágenes/s (equivalente)", _fmt(m.imagenes_por_s, "", 2)),
        ("GPU % pico / prom (dispositivo)",
         f"{_fmt(m.gpu_pct_pico, '', 0)} / {_fmt(m.gpu_pct_prom, '', 0)}"),
        ("VRAM usada pico (dispositivo)", _fmt(m.vram_usada_mb, " MB", 0)),
        ("VRAM pico torch", _fmt(m.vram_pico_torch_mb, " MB", 0)),
        ("Temperatura máx", _fmt(m.temperatura_c_max, " °C", 0)),
        ("Potencia prom", _fmt(m.potencia_w_prom, " W")),
        ("CPU % prom", _fmt(m.cpu_pct_prom, " %")),
        ("RAM proceso pico", _fmt(m.ram_proceso_mb_pico, " MB", 0)),
    ]
    for nombre, valor in filas:
        tabla.add_row(nombre, valor)
    consola.print(tabla)
    if m.muestras_gpu < MIN_MUESTRAS_FIABLES:
        consola.print(
            f"[yellow]Aviso: solo {m.muestras_gpu} muestras de GPU "
            f"(< {MIN_MUESTRAS_FIABLES}); los valores de GPU no son fiables.[/yellow]")


def _info_gpu() -> dict[str, str]:
    """Nombre, VRAM total y driver vía NVML; vacío si no hay NVIDIA."""
    nvml = _obtener_nvml()
    if nvml is None:
        return {}
    try:
        h = _resolver_handle(nvml, 0)
        nombre = nvml.nvmlDeviceGetName(h)
        if isinstance(nombre, bytes):
            nombre = nombre.decode("utf-8", "replace")
        driver = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", "replace")
        return {
            "nombre": str(nombre),
            "vram": f"{nvml.nvmlDeviceGetMemoryInfo(h).total / 2**20:.0f} MB",
            "driver": str(driver),
        }
    except (nvml.NVMLError, OSError):
        return {}


def reporte_arranque(
    pipeline: Pipeline,
    calentamiento: MetricasAnalisis,
    *,
    consola: Console | None = None,
) -> None:
    """GPU (nombre, VRAM, driver, CUDA), torch, dispositivo/precisión, `pipeline.carga`
    y el calentamiento (marcado, fuera de estadísticas).
    """
    from rich.console import Console
    from rich.table import Table

    consola = consola or Console()
    tabla = Table(title="Arranque de RoadVision")
    tabla.add_column("Elemento")
    tabla.add_column("Valor")
    gpu = _info_gpu()
    tabla.add_row("GPU", gpu.get("nombre", "no disponible"))
    tabla.add_row("VRAM total", gpu.get("vram", "n/d"))
    tabla.add_row("Driver", gpu.get("driver", "n/d"))
    torch = sys.modules.get("torch")
    tabla.add_row("torch", getattr(torch, "__version__", "no cargado"))
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    tabla.add_row("CUDA", str(cuda) if cuda else "n/d")
    tabla.add_row("Dispositivo / precisión",
                  f"{pipeline.dispositivo.descripcion} / {_precision(pipeline)}")
    for nombre, carga in pipeline.carga.items():
        tabla.add_row(f"Carga {nombre}", f"{carga.tiempo_ms:.0f} ms, {carga.vram_mb:.0f} MB VRAM")
    c = calentamiento
    tabla.add_row(
        "Calentamiento (fuera de estadísticas)",
        f"detección {_fmt(c.ms_deteccion, ' ms', 0)}, "
        f"clasificación {_fmt(c.ms_clasificacion, ' ms', 0)}, "
        f"respaldo {_fmt(c.ms_respaldo, ' ms', 0)}")
    consola.print(tabla)
