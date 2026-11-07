"""Dobles y ayudantes de prueba compartidos."""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image

from roadvision.catalog import cargar_catalogo
from roadvision.config import Config
from roadvision.contracts import Deteccion, Prediccion
from roadvision.device import Dispositivo
from roadvision.pipeline import Pipeline

CATALOGO_PRUEBA = Path(__file__).parent / "fixtures" / "catalogo_prueba.yaml"


def jpeg_con_orientacion(ancho: int, alto: int, orientacion: int) -> Image.Image:
    """JPEG en memoria con la etiqueta EXIF Orientation indicada."""
    exif = Image.Exif()
    exif[0x0112] = orientacion
    buffer = io.BytesIO()
    Image.new("RGB", (ancho, alto), (10, 20, 30)).save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    return Image.open(buffer)


def recorte(n: int) -> Image.Image:
    """Recorte de prueba de tamaño (n, n); su ancho lo identifica."""
    return Image.new("RGB", (n, n), (0, 0, 0))


class FakeRespaldo:
    """Respaldo CLIP falso: identifica cada recorte por su ancho en píxeles."""

    def __init__(
        self,
        puntajes: dict[int, dict[str, float]] | None = None,
        por_defecto: dict[str, float] | None = None,
    ) -> None:
        self.puntajes = puntajes or {}
        self.por_defecto = por_defecto or {}
        self.llamadas: list[tuple[int, int]] = []

    def puntuar(self, recortes: list[Image.Image], textos: list[str]) -> list[list[float]]:
        self.llamadas.append((len(recortes), len(textos)))
        filas = []
        for r in recortes:
            propios = self.puntajes.get(r.width, {})
            fila = [propios.get(t, self.por_defecto.get(t, 0.01)) for t in textos]
            total = sum(fila)
            filas.append([v / total for v in fila])
        return filas


class FakeDetector:
    """Detector falso: devuelve siempre las mismas detecciones."""

    def __init__(self, detecciones: list[Deteccion]) -> None:
        self.detecciones = detecciones
        self.ultima_imagen: Image.Image | None = None

    def detectar(self, imagen: Image.Image, umbral: float) -> list[Deteccion]:
        self.ultima_imagen = imagen
        return list(self.detecciones)


class FakeClasificador:
    """Clasificador falso: consume `predicciones` en orden, cíclicamente."""

    def __init__(self, predicciones: list[Prediccion], etiquetas: list[str] | None = None) -> None:
        self.predicciones = predicciones
        self.etiquetas = etiquetas or []
        self.recibidos = 0
        self._siguiente = 0

    def clasificar(self, recortes: list[Image.Image]) -> list[Prediccion]:
        self.recibidos += len(recortes)
        resultado = []
        for _ in recortes:
            resultado.append(self.predicciones[self._siguiente % len(self.predicciones)])
            self._siguiente += 1
        return resultado


def pipeline_falso(
    detecciones: list[Deteccion],
    predicciones: list[Prediccion],
    respaldo: FakeRespaldo | None = None,
) -> Pipeline:
    return pipeline_con_clasificador(detecciones, FakeClasificador(predicciones), respaldo)


def pipeline_con_clasificador(
    detecciones: list[Deteccion],
    clasificador: FakeClasificador,
    respaldo: FakeRespaldo | None = None,
) -> Pipeline:
    return Pipeline(
        Config(),
        FakeDetector(detecciones),
        clasificador,
        respaldo or FakeRespaldo(),
        cargar_catalogo(CATALOGO_PRUEBA),
        Dispositivo("cpu", torch.float32, "CPU"),
    )
