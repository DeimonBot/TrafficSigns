import pytest

from roadvision.cajas import contencion, filtrar_cajas, iou
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


def test_contencion():
    assert contencion((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert contencion((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # una caja pequeña totalmente dentro de otra grande: 1.0 aunque el IoU sea bajo
    assert contencion((0, 0, 100, 400), (10, 10, 90, 90)) == 1.0
    assert iou((0, 0, 100, 400), (10, 10, 90, 90)) < 0.2


def test_caja_alta_con_poste_y_caja_ajustada_se_quedan_con_la_ajustada():
    """La misma señal detectada con y sin el poste: IoU bajo, pero es una sola señal."""
    dets = [Deteccion((40, 40, 120, 440), 0.7),    # señal + poste (alta)
            Deteccion((45, 45, 115, 115), 0.9)]    # solo la señal
    assert filtrar_cajas(dets, umbral=0.15, tamano_minimo=20, iou_max=0.5) == [
        Deteccion((45, 45, 115, 115), 0.9)]


def test_cajas_anidadas_conserva_la_pequena_aunque_tenga_menos_puntaje():
    dets = [Deteccion((0, 0, 200, 600), 0.95), Deteccion((20, 20, 120, 120), 0.4)]
    assert filtrar_cajas(dets, umbral=0.15, tamano_minimo=20, iou_max=0.5) == [
        Deteccion((20, 20, 120, 120), 0.4)]


def test_dos_senales_separadas_dentro_de_una_caja_grande_se_conservan_las_dos():
    dets = [Deteccion((0, 0, 100, 400), 0.9),      # abarca las dos
            Deteccion((10, 10, 90, 90), 0.8),
            Deteccion((10, 250, 90, 330), 0.7)]
    resultado = filtrar_cajas(dets, umbral=0.15, tamano_minimo=20, iou_max=0.5)
    assert {d.caja for d in resultado} == {(10, 10, 90, 90), (10, 250, 90, 330)}


def test_cajas_vecinas_sin_contencion_no_se_tocan():
    dets = [Deteccion((0, 0, 50, 50), 0.9), Deteccion((60, 0, 110, 50), 0.8)]
    assert len(filtrar_cajas(dets, umbral=0.15, tamano_minimo=20, iou_max=0.5)) == 2


def test_contencion_max_configurable():
    dets = [Deteccion((0, 0, 100, 100), 0.9), Deteccion((20, 20, 70, 70), 0.8)]
    assert len(filtrar_cajas(dets, 0.15, 20, 0.5, contencion_max=0.8)) == 1
    assert len(filtrar_cajas(dets, 0.15, 20, 0.5, contencion_max=1.01)) == 2   # desactivado
