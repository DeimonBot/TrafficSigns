"""Utilidades de imagen: normalización (EXIF, RGB) y recorte con margen."""
from __future__ import annotations

import math

from PIL import Image, ImageOps

from roadvision.contracts import Caja


def normalizar(imagen: Image.Image) -> Image.Image:
    """Aplica la orientación EXIF y devuelve RGB (transparencia sobre blanco)."""
    imagen = ImageOps.exif_transpose(imagen)
    tiene_alfa = imagen.mode in ("RGBA", "LA", "PA") or (
        imagen.mode == "P" and "transparency" in imagen.info
    )
    if tiene_alfa:
        rgba = imagen.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(fondo, rgba).convert("RGB")
    return imagen.convert("RGB")


def _redondear(valor: float) -> int:
    return math.floor(valor + 0.5)


def recortar(imagen: Image.Image, caja: Caja, margen: float) -> Image.Image:
    """Recorta `caja` ampliada por `margen` (fracción de ancho/alto) en cada lado."""
    x1, y1, x2, y2 = caja
    dx = margen * (x2 - x1)
    dy = margen * (y2 - y1)
    izq = max(0, _redondear(x1 - dx))
    arr = max(0, _redondear(y1 - dy))
    der = min(imagen.width, _redondear(x2 + dx))
    aba = min(imagen.height, _redondear(y2 + dy))
    return imagen.crop((izq, arr, der, aba))
