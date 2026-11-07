import json
from pathlib import Path

import pytest
import yaml

from roadvision.catalog import ErrorCatalogo, cargar_catalogo, etiquetas_sin_cubrir
from roadvision.config import Config

FIXTURES = Path(__file__).parent / "fixtures"
PRUEBA = FIXTURES / "catalogo_prueba.yaml"


def _cargar_yaml():
    return yaml.safe_load(PRUEBA.read_text("utf-8"))


def _escribir(tmp_path, datos):
    ruta = tmp_path / "cat.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True), "utf-8")
    return ruta


def test_catalogo_real_cubre_las_43_etiquetas():
    cat = cargar_catalogo(Config().ruta_catalogo)
    etiquetas = json.loads((FIXTURES / "gtsrb_etiquetas.json").read_text("utf-8"))
    assert len(etiquetas) == 43
    assert etiquetas_sin_cubrir(cat, etiquetas) == []


def test_catalogo_real_entradas_minimas():
    cat = cargar_catalogo(Config().ruta_catalogo)
    assert cat.por_etiqueta_gtsrb("Stop").id == "pare"
    assert cat.por_etiqueta_gtsrb("Yield").id == "ceda_el_paso"
    assert {e.velocidad_kmh for e in cat.velocidades()} >= {20, 30, 40, 50, 60, 70, 80, 90, 100, 120}
    assert cat.entradas["velocidad_40"].gtsrb == () and cat.entradas["velocidad_90"].gtsrb == ()
    assert {"Priority road", "End of speed limit (80km/h)", "End speed + passing limits",
            "End of no passing", "End no passing veh > 3.5 tons"} <= cat.gtsrb_sin_equivalente
    assert all(e.significado.strip() for e in cat.entradas.values())


def test_catalogo_real_textos_fijos():
    cat = cargar_catalogo(Config().ruta_catalogo)
    assert cat.nombre_no_reconocida == "Señal no reconocida"
    assert cat.entradas["pare"].nombre == "PARE"
    assert cat.entradas["velocidad_no_verificada"].clip is None
    assert cat.entradas["velocidad_30"].clip == "a circular speed limit road sign showing the number 30"


def test_rechaza_etiqueta_en_dos_entradas(tmp_path):
    d = _cargar_yaml()
    d["senales"][1]["gtsrb"] = ["Stop"]
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_rechaza_categoria_invalida(tmp_path):
    d = _cargar_yaml()
    d["senales"][0]["categoria"] = "inventada"
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_rechaza_mapeada_y_sin_equivalente(tmp_path):
    d = _cargar_yaml()
    d["gtsrb_sin_equivalente"].append("Stop")
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_rechaza_ids_repetidos(tmp_path):
    d = _cargar_yaml()
    d["senales"][1]["id"] = "pare"
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_rechaza_categoria_generica_faltante(tmp_path):
    d = _cargar_yaml()
    del d["categorias"]["informativa"]
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_requiere_velocidad_no_verificada(tmp_path):
    d = _cargar_yaml()
    d["senales"] = [s for s in d["senales"] if s["id"] != "velocidad_no_verificada"]
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_velocidad_no_verificada_debe_ser_reglamentaria(tmp_path):
    d = _cargar_yaml()
    for s in d["senales"]:
        if s["id"] == "velocidad_no_verificada":
            s["categoria"] = "preventiva"
    with pytest.raises(ErrorCatalogo):
        cargar_catalogo(_escribir(tmp_path, d))


def test_catalogo_de_prueba_carga():
    cat = cargar_catalogo(PRUEBA)
    assert [e.velocidad_kmh for e in cat.velocidades()] == [30, 50, 60, 40]
    assert [e.id for e in cat.con_clip()] == [
        "velocidad_30", "velocidad_50", "velocidad_60", "velocidad_40", "resalto"]
    assert cat.por_etiqueta_gtsrb("Priority road") is None
    assert etiquetas_sin_cubrir(cat, ["Stop", "Priority road", "Yield"]) == ["Yield"]
