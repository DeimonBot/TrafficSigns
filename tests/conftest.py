"""Fixtures de sesión con modelos reales; solo las usan las pruebas `modelos`."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from roadvision.config import Config, ruta_datasets
from roadvision.contracts import Caja
from roadvision.device import Dispositivo, elegir_dispositivo

if TYPE_CHECKING:
    from roadvision.classifier import ClasificadorGtsrb


@pytest.fixture(scope="session")
def dispositivo() -> Dispositivo:
    return elegir_dispositivo("auto")


@pytest.fixture(scope="session")
def clasificador(dispositivo: Dispositivo) -> ClasificadorGtsrb:
    from roadvision.classifier import ClasificadorGtsrb

    return ClasificadorGtsrb(Config().modelo_clasificador, dispositivo)


@pytest.fixture(scope="session")
def muestra_gtsrb() -> Callable[[int], Image.Image]:
    """Función clase -> primera imagen del test de GTSRB con esa clase."""
    from torchvision.datasets import GTSRB

    conjunto = GTSRB(root=ruta_datasets(), split="test", download=True)
    primera: dict[int, int] = {}
    for indice, (_, clase) in enumerate(conjunto._samples):
        primera.setdefault(clase, indice)
    cache: dict[int, Image.Image] = {}

    def obtener(clase: int) -> Image.Image:
        if clase not in cache:
            imagen, _ = conjunto[primera[clase]]
            cache[clase] = imagen.convert("RGB")
        return cache[clase]

    return obtener


@pytest.fixture(scope="session")
def escena_con_pare(muestra_gtsrb) -> Callable[..., tuple[Image.Image, Caja]]:
    def crear(ancho: int, alto: int, x: int, y: int, lado: int) -> tuple[Image.Image, Caja]:
        escena = Image.new("RGB", (ancho, alto), (128, 128, 128))
        pare = muestra_gtsrb(14).resize((lado, lado))
        escena.paste(pare, (x, y))
        return escena, (x, y, x + lado, y + lado)

    return crear
