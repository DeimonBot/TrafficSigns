"""Dobles y ayudantes de prueba compartidos."""
from __future__ import annotations

import io

from PIL import Image


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
