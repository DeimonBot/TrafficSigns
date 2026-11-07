import pytest

from roadvision.cajas import filtrar_cajas, iou
from roadvision.contracts import Deteccion


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_filtrar_umbral_tamano_y_duplicados():
    dets = [Deteccion((100, 100, 150, 150), 0.9),
            Deteccion((102, 101, 152, 149), 0.6),   # misma señal, otra consulta
            Deteccion((300, 300, 310, 330), 0.9),   # 10 px de ancho
            Deteccion((400, 100, 460, 160), 0.1)]   # bajo el umbral
    assert filtrar_cajas(dets, umbral=0.15, tamano_minimo=20, iou_max=0.5) == [
        Deteccion((100, 100, 150, 150), 0.9)]
