"""Utilidades puras sobre cajas: IoU y filtrado con supresión de no-máximos."""
from __future__ import annotations

from roadvision.contracts import Caja, Deteccion


def iou(a: Caja, b: Caja) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    interseccion = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return interseccion / (area_a + area_b - interseccion)


def filtrar_cajas(
    detecciones: list[Deteccion], umbral: float, tamano_minimo: int, iou_max: float
) -> list[Deteccion]:
    """Quita por puntaje y tamaño, y suprime duplicados (NMS voraz por puntaje)."""
    candidatas = [
        d
        for d in detecciones
        if d.puntaje >= umbral
        and min(d.caja[2] - d.caja[0], d.caja[3] - d.caja[1]) >= tamano_minimo
    ]
    candidatas.sort(key=lambda d: d.puntaje, reverse=True)
    conservadas: list[Deteccion] = []
    for d in candidatas:
        if all(iou(d.caja, k.caja) <= iou_max for k in conservadas):
            conservadas.append(d)
    return conservadas
