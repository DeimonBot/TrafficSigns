"""Pruebas del respaldo CLIP con el modelo real."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.modelos


def test_filas_suman_uno_y_pare_prefiere_octogono(respaldo, muestra_gtsrb):
    [fila] = respaldo.puntuar(
        [muestra_gtsrb(14)],
        ["a red octagonal stop sign", "a yellow diamond-shaped warning road sign"],
    )
    assert sum(fila) == pytest.approx(1.0, abs=1e-3)
    assert fila[0] > fila[1]


def test_vacio(respaldo):
    assert respaldo.puntuar([], ["x"]) == []


def test_sin_textos_falla(respaldo, muestra_gtsrb):
    with pytest.raises(ValueError):
        respaldo.puntuar([muestra_gtsrb(14)], [])
