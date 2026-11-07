"""Pruebas del clasificador GTSRB con el modelo real (marca `modelos`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.modelos


def test_etiquetas_en_orden(clasificador):
    esperadas = json.loads((FIXTURES / "gtsrb_etiquetas.json").read_text("utf-8"))
    assert clasificador.etiquetas == esperadas


def test_clasifica_pare_de_gtsrb(clasificador, muestra_gtsrb):
    [p] = clasificador.clasificar([muestra_gtsrb(14)])
    assert p.etiqueta == "Stop" and p.probabilidad >= 0.8


def test_lista_vacia(clasificador):
    assert clasificador.clasificar([]) == []
