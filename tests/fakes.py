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
