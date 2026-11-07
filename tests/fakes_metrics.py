"""Dobles para probar `roadvision.metrics` sin torch ni GPU."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from roadvision.contracts import Categoria, Estado, Fuente, Resultado, Senal


def senal(numero: int, estado: Estado) -> Senal:
    return Senal(
        numero=numero, caja=(0, 0, 10, 10), recorte=Image.new("RGB", (10, 10)),
        estado=estado, fuente=Fuente.GTSRB, id_catalogo=None, nombre="x",
        categoria=Categoria.REGLAMENTARIA, significado=None, confianza=0.9,
        puntaje_deteccion=0.5)


def resultado(estados: list[Estado], total_ms: float = 20.0) -> Resultado:
    return Resultado(
        senales=[senal(i, e) for i, e in enumerate(estados, start=1)],
        conteo_por_nombre={}, conteo_por_categoria={},
        tiempos_ms={"total": total_ms, "preproceso": 2.0, "deteccion": 10.0,
                    "clasificacion": 5.0, "respaldo": 3.0},
        ancho=640, alto=480, dispositivo="CPU")


class PipelineFalso:
    """Imita lo que `metrics` usa de `Pipeline`: dispositivo, analizar y calentar."""

    def __init__(self, estados: list[Estado] | None = None) -> None:
        self.dispositivo = SimpleNamespace(nombre="cpu", dtype="torch.float32", descripcion="CPU")
        self.estados = estados or []
        self.analizados = 0
        self.calentado = 0

    def analizar(self, imagen, umbrales=None) -> Resultado:
        self.analizados += 1
        return resultado(self.estados)

    def calentar(self) -> dict[str, float]:
        self.calentado += 1
        return {"deteccion": 12.0, "clasificacion": 4.0, "respaldo": 2.0}


class NvmlFalso:
    """Reemplazo mínimo de `pynvml` con una GPU ficticia."""

    NVML_TEMPERATURE_GPU = 0

    class NVMLError(Exception):
        pass

    def nvmlDeviceGetHandleByIndex(self, i):
        return ("gpu", i)

    def nvmlDeviceGetHandleByPciBusId(self, bus_id):
        return ("gpu", bus_id)

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=40)

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=512 * 2**20, total=8192 * 2**20)

    def nvmlDeviceGetTemperature(self, h, tipo):
        return 60

    def nvmlDeviceGetPowerUsage(self, h):
        return 90_000  # mW
