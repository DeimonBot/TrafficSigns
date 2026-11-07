from __future__ import annotations

from PIL import Image

from roadvision.config import Umbrales
from roadvision.contracts import Deteccion, Prediccion
from tests.fakes import (
    FakeClasificador,
    FakeRespaldo,
    jpeg_con_orientacion,
    pipeline_con_clasificador as pipeline_con,
    pipeline_falso,
)

IMG = Image.new("RGB", (640, 480))


def test_sin_detecciones():
    respaldo = FakeRespaldo()
    p = pipeline_falso([], [], respaldo)
    r = p.analizar(Image.new("RGB", (640, 480)))
    assert (r.senales, r.conteo_por_nombre, r.conteo_por_categoria) == ([], {}, {})
    assert respaldo.llamadas == []
    assert set(r.tiempos_ms) == {"total", "preproceso", "deteccion", "clasificacion", "respaldo"}
    assert (r.ancho, r.alto) == (640, 480)


def test_dos_pare_y_una_no_reconocida():
    dets = [Deteccion((300, 50, 360, 110), 0.9), Deteccion((100, 50, 160, 110), 0.8),
            Deteccion((500, 50, 560, 110), 0.7)]
    preds = [Prediccion("Stop", 0.97), Prediccion("Stop", 0.95), Prediccion("Priority road", 0.99)]
    r = pipeline_falso(dets, preds, FakeRespaldo(por_defecto={"otro": 5})).analizar(
        Image.new("RGB", (640, 480)))
    assert [s.numero for s in r.senales] == [1, 2, 3]
    assert [s.caja[0] for s in r.senales] == [100, 300, 500]
    assert [s.nombre for s in r.senales] == ["PARE", "PARE", "Señal no reconocida"]
    assert r.conteo_por_nombre == {"PARE": 2, "Señal no reconocida": 1}
    assert r.conteo_por_categoria == {"reglamentaria": 2, "sin_categoria": 1}


def test_duplicados_y_pequenas():
    dets = [Deteccion((100, 100, 150, 150), 0.9), Deteccion((102, 101, 152, 149), 0.6),
            Deteccion((300, 300, 310, 330), 0.9)]
    clasif = FakeClasificador([Prediccion("Stop", 0.97)])
    r = pipeline_con(dets, clasif).analizar(Image.new("RGB", (640, 480)))
    assert len(r.senales) == 1 and clasif.recibidos == 1


def test_umbral_de_deteccion_por_llamada():
    dets = [Deteccion((100, 100, 150, 150), 0.20), Deteccion((300, 100, 350, 150), 0.12)]
    p = pipeline_falso(dets, [Prediccion("Stop", 0.97)] * 2)
    assert len(p.analizar(IMG, Umbrales(deteccion=0.15)).senales) == 1
    assert len(p.analizar(IMG, Umbrales(deteccion=0.10)).senales) == 2


def test_exif_rotado():
    r = pipeline_falso([], []).analizar(jpeg_con_orientacion(400, 200, 6))
    assert (r.ancho, r.alto) == (200, 400)


def test_calentar_usa_cada_modelo():
    respaldo = FakeRespaldo()
    p = pipeline_falso([], [Prediccion("Stop", 0.9)], respaldo)
    assert set(p.calentar()) == {"deteccion", "clasificacion", "respaldo"}
    assert len(respaldo.llamadas) == 1
