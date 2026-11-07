"""Cronómetro por etapas, con sincronización opcional de la GPU."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class Cronometro:
    def __init__(self, sincronizar: Callable[[], None] | None = None) -> None:
        self._sincronizar = sincronizar
        self.tiempos_ms: dict[str, float] = {}

    @contextmanager
    def etapa(self, nombre: str) -> Iterator[None]:
        if self._sincronizar:
            self._sincronizar()
        inicio = time.perf_counter()
        try:
            yield
        finally:
            if self._sincronizar:
                self._sincronizar()
            ms = (time.perf_counter() - inicio) * 1000
            self.tiempos_ms[nombre] = self.tiempos_ms.get(nombre, 0.0) + ms
