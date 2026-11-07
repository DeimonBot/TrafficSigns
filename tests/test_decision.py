from __future__ import annotations

from pathlib import Path

from roadvision.catalog import cargar_catalogo
from roadvision.config import Umbrales
from roadvision.contracts import Categoria, Estado, Fuente, Prediccion
from roadvision.decision import Decision, Decisor
from tests.fakes import FakeRespaldo, recorte

CATALOGO = cargar_catalogo(Path(__file__).parent / "fixtures" / "catalogo_prueba.yaml")
U = Umbrales()


def armar(puntajes=None):
    respaldo = FakeRespaldo(puntajes)
    return Decisor(CATALOGO, respaldo), respaldo


def test_stop_confiable_no_usa_clip():
    decisor, respaldo = armar()
    assert decisor.decidir([recorte(1)], [Prediccion("Stop", 0.97)], U) == [
        Decision(Estado.RECONOCIDA, Fuente.GTSRB, "pare", Categoria.REGLAMENTARIA, 0.97)]
    assert respaldo.llamadas == []


def test_priority_road_se_ignora_y_va_a_clip():
    decisor, _ = armar({1: {"resalto": 9}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Priority road", 0.99)], U)
    assert (d.estado, d.fuente, d.id_catalogo, d.categoria) == (
        Estado.APROXIMADA, Fuente.CLIP, "resalto", Categoria.PREVENTIVA)


def test_baja_confianza_queda_en_categoria():
    decisor, _ = armar({1: {"cat-prev": 5}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Stop", 0.5)], U)
    assert (d.estado, d.fuente, d.id_catalogo, d.categoria) == (
        Estado.SOLO_CATEGORIA, Fuente.CLIP, None, Categoria.PREVENTIVA)


def test_velocidad_confirmada():
    decisor, _ = armar({1: {"v50": 5}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (50km/h)", 0.95)], U)
    assert d == Decision(Estado.RECONOCIDA, Fuente.GTSRB, "velocidad_50", Categoria.REGLAMENTARIA, 0.95)


def test_velocidad_que_gtsrb_no_tiene():
    decisor, _ = armar({1: {"v40": 5}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (30km/h)", 0.93)], U)
    assert (d.estado, d.fuente, d.id_catalogo) == (Estado.APROXIMADA, Fuente.CLIP, "velocidad_40")


def test_velocidad_en_desacuerdo():
    decisor, _ = armar({1: {"v60": 5}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (30km/h)", 0.93)], U)
    assert d == Decision(Estado.SOLO_CATEGORIA, Fuente.GTSRB, "velocidad_no_verificada",
                         Categoria.REGLAMENTARIA, 0.93)


def test_velocidad_clip_inseguro():
    decisor, _ = armar({1: {"v40": 0.02}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (30km/h)", 0.93)], U)
    assert d.id_catalogo == "velocidad_no_verificada"


def test_otro_gana_no_reconocida():
    decisor, _ = armar({1: {"otro": 5}})
    [d] = decisor.decidir([recorte(1)], [Prediccion("Priority road", 0.99)], U)
    assert d == Decision(Estado.NO_RECONOCIDA, Fuente.NINGUNA, None, None, 0.0)


def test_lote_agrupa_llamadas():
    decisor, respaldo = armar({2: {"v30": 5}, 3: {"resalto": 5}})
    decisor.decidir([recorte(1), recorte(2), recorte(3)],
                    [Prediccion("Stop", 0.97), Prediccion("Speed limit (30km/h)", 0.93),
                     Prediccion("Priority road", 0.99)], U)
    assert respaldo.llamadas == [(1, 4), (1, 5)]


def test_umbral_gtsrb_por_llamada():
    decisor, _ = armar({1: {"otro": 5}})
    p = [Prediccion("Stop", 0.85)]
    assert decisor.decidir([recorte(1)], p, Umbrales(gtsrb=0.80))[0].estado == Estado.RECONOCIDA
    assert decisor.decidir([recorte(1)], p, Umbrales(gtsrb=0.90))[0].estado == Estado.NO_RECONOCIDA
