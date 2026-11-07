from __future__ import annotations

import pytest
from PIL import Image

from roadvision.contracts import Prediccion
from roadvision.evaluacion import evaluar_clasificador
from tests.fakes import FakeClasificador


def test_evaluar_clasificador():
    clasif = FakeClasificador(
        [Prediccion("A", 0.9), Prediccion("B", 0.9), Prediccion("A", 0.9)],
        etiquetas=["A", "B"],
    )
    img = Image.new("RGB", (32, 32))
    r = evaluar_clasificador(clasif, [(img, 0), (img, 1), (img, 1)], tamano_lote=2)
    assert r.n == 3
    assert r.exactitud == pytest.approx(2 / 3)
    assert r.exactitud_por_etiqueta == {"A": 1.0, "B": 0.5}
    assert r.imagenes_por_segundo > 0
