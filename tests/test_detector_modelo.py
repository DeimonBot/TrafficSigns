"""Pruebas del detector OWLv2 con el modelo real."""
from __future__ import annotations

import pytest
from PIL import Image

from roadvision.cajas import iou

pytestmark = pytest.mark.modelos


def test_detecta_senal_pegada(detector, escena_con_pare):
    img, caja = escena_con_pare(800, 600, 420, 120, 128)
    dets = detector.detectar(img, umbral=0.10)
    assert max(iou(d.caja, caja) for d in dets) >= 0.3


def test_imagen_no_cuadrada_cajas_en_su_sitio(detector, escena_con_pare):
    img, caja = escena_con_pare(1200, 400, 1000, 250, 128)
    dets = detector.detectar(img, umbral=0.10)
    assert max(iou(d.caja, caja) for d in dets) >= 0.3
    assert all(
        0 <= d.caja[0] < d.caja[2] <= 1200 and 0 <= d.caja[1] < d.caja[3] <= 400 for d in dets
    )


def test_imagen_lisa_sin_detecciones(detector):
    assert detector.detectar(Image.new("RGB", (640, 480), (128, 128, 128)), umbral=0.15) == []
