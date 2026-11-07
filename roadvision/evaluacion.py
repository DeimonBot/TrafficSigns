"""Evaluación del clasificador (nivel A: exactitud sobre el test oficial de GTSRB)."""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from PIL import Image

from roadvision.contracts import Clasificador


@dataclass(frozen=True)
class ReporteA:
    n: int
    exactitud: float
    exactitud_por_etiqueta: dict[str, float]
    imagenes_por_segundo: float


def _lotes(
    muestras: Iterable[tuple[Image.Image, int]], tamano: int
) -> Iterator[list[tuple[Image.Image, int]]]:
    iterador = iter(muestras)
    while lote := list(islice(iterador, tamano)):
        yield lote


def evaluar_clasificador(
    clasificador: Clasificador,
    muestras: Iterable[tuple[Image.Image, int]],
    tamano_lote: int = 64,
) -> ReporteA:
    """Clasifica `muestras` por lotes y mide la exactitud.

    La etiqueta correcta de una muestra es `clasificador.etiquetas[clase]`.
    """
    etiquetas = clasificador.etiquetas
    total: Counter[str] = Counter()
    aciertos: Counter[str] = Counter()
    segundos = 0.0
    for lote in _lotes(muestras, tamano_lote):
        imagenes = [imagen for imagen, _ in lote]
        inicio = time.perf_counter()
        predicciones = clasificador.clasificar(imagenes)
        segundos += time.perf_counter() - inicio
        for (_, clase), prediccion in zip(lote, predicciones, strict=True):
            esperada = etiquetas[clase]
            total[esperada] += 1
            aciertos[esperada] += prediccion.etiqueta == esperada
    n = sum(total.values())
    return ReporteA(
        n=n,
        exactitud=sum(aciertos.values()) / n if n else 0.0,
        exactitud_por_etiqueta={e: aciertos[e] / c for e, c in total.items()},
        imagenes_por_segundo=n / max(segundos, 1e-9),
    )
