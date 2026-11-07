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


def contencion(a: Caja, b: Caja) -> float:
    """Fracción de la caja MÁS PEQUEÑA que queda dentro de la otra (1.0 = totalmente contenida).

    A diferencia del IoU, no se diluye cuando una caja alta contiene a una pequeña.
    """
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return (ix * iy) / min(area_a, area_b)


def _area(caja: Caja) -> int:
    return (caja[2] - caja[0]) * (caja[3] - caja[1])


def filtrar_cajas(
    detecciones: list[Deteccion],
    umbral: float,
    tamano_minimo: int,
    iou_max: float,
    contencion_max: float = 0.8,
) -> list[Deteccion]:
    """Quita por puntaje y tamaño, y suprime duplicados.

    1. NMS voraz por puntaje (IoU): quita la misma caja repetida.
    2. Cajas anidadas: si una caja contiene (>= `contencion_max`) a otra más pequeña, se
       descarta la grande y se conserva la pequeña, que encuadra mejor la señal. Sin este paso,
       una señal con su poste detectada dos veces sale como dos señales distintas (el IoU
       entre una caja alta y una pequeña es bajo aunque una esté dentro de la otra).
    """
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

    finales: list[Deteccion] = []
    for d in sorted(conservadas, key=lambda d: _area(d.caja)):  # de pequeña a grande
        if all(contencion(d.caja, k.caja) < contencion_max for k in finales):
            finales.append(d)
    finales.sort(key=lambda d: d.puntaje, reverse=True)
    return finales
