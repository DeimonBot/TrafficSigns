"""Pruebas del etiquetado: lógica pura (tmp_path) y la app Streamlit con AppTest.

Autor: jhoan
Fecha: 2026-11-06
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from roadvision.catalog import cargar_catalogo
from roadvision.config import RAIZ
from roadvision.etiquetado import (
    CLASE_NO_SENAL,
    CLASE_OMITIDA,
    CLASE_OTRA,
    Recorte,
    agregar_al_indice,
    clases_disponibles,
    conteos_por_clase,
    etiquetas_vigentes,
    deshacer_ultima,
    duplicadas_por_sha1,
    fotos_distintas,
    fotos_indexadas,
    guardar_etiqueta,
    leer_etiquetas,
    leer_indice,
    listar_fotos,
    nombre_recorte,
    rutas_por_defecto,
    sha1_archivo,
    siguiente_sin_etiquetar,
    stem_unico,
)

CATALOGO = RAIZ / "data" / "catalogo_senales.yaml"


def _recorte(n: int, foto: str = "a.jpg", sha1: str = "aa") -> Recorte:
    return Recorte(nombre_recorte(Path(foto).stem, n), foto, (1, 2, 30, 40), 0.5, 29, 38, sha1)


def test_nombre_recorte():
    assert nombre_recorte("foto", 3) == "foto__03.png"


def test_indice_ida_y_vuelta_e_idempotente(tmp_path):
    ruta = tmp_path / "indice.csv"
    rs = [_recorte(1), _recorte(2)]
    assert agregar_al_indice(ruta, rs) == 2
    assert agregar_al_indice(ruta, rs) == 0  # no duplica
    assert leer_indice(ruta) == [
        Recorte(r.recorte, r.foto, r.caja, 0.5, 29, 38, "aa") for r in rs]
    assert ruta.read_text(encoding="utf-8").splitlines()[0] == (
        "recorte,foto,caja_x1,caja_y1,caja_x2,caja_y2,puntaje,ancho,alto,sha1_foto")


def test_leer_sin_archivo(tmp_path):
    assert leer_indice(tmp_path / "no.csv") == []
    assert leer_etiquetas(tmp_path / "no.csv") == []


def test_etiquetas_siguiente_y_deshacer(tmp_path):
    ruta = tmp_path / "etiquetas.csv"
    indice = [_recorte(1), _recorte(2), _recorte(3)]
    assert siguiente_sin_etiquetar(indice, []) == indice[0]
    e = guardar_etiqueta(ruta, indice[0].recorte, "a.jpg", "pare")
    assert "T" in e.marca_tiempo and (e.marca_tiempo[-6] in "+-" or e.marca_tiempo.endswith("Z"))
    guardar_etiqueta(ruta, indice[1].recorte, "a.jpg", CLASE_NO_SENAL)
    etiquetas = leer_etiquetas(ruta)
    assert [x.clase for x in etiquetas] == ["pare", CLASE_NO_SENAL]
    assert siguiente_sin_etiquetar(indice, etiquetas) == indice[2]
    guardar_etiqueta(ruta, indice[2].recorte, "a.jpg", "pare")
    assert siguiente_sin_etiquetar(indice, leer_etiquetas(ruta)) is None
    assert deshacer_ultima(ruta).recorte == indice[2].recorte
    assert siguiente_sin_etiquetar(indice, leer_etiquetas(ruta)) == indice[2]
    assert not list(tmp_path.glob("*.tmp"))
    assert deshacer_ultima(ruta).recorte == indice[1].recorte
    assert deshacer_ultima(ruta).recorte == indice[0].recorte
    assert deshacer_ultima(ruta) is None
    assert leer_etiquetas(ruta) == []


def test_conteos_excluyen_omitida_y_gana_la_ultima(tmp_path):
    ruta = tmp_path / "e.csv"
    guardar_etiqueta(ruta, "a__01.png", "a.jpg", "pare")
    guardar_etiqueta(ruta, "a__02.png", "a.jpg", "pare")
    guardar_etiqueta(ruta, "b__01.png", "b.jpg", CLASE_NO_SENAL)
    guardar_etiqueta(ruta, "c__01.png", "c.jpg", CLASE_OMITIDA)
    guardar_etiqueta(ruta, "a__02.png", "a.jpg", CLASE_OTRA)  # corrige la anterior
    etiquetas = leer_etiquetas(ruta)
    assert conteos_por_clase(etiquetas) == {CLASE_NO_SENAL: 1, CLASE_OTRA: 1, "pare": 1}
    assert fotos_distintas(etiquetas) == 2  # a y b; c solo omitida


def test_clases_agrupadas_del_catalogo_real():
    catalogo = cargar_catalogo(CATALOGO)
    grupos = clases_disponibles(catalogo)
    assert list(grupos) == ["reglamentaria", "preventiva", "informativa", "especiales"]
    assert any(c.id == "pare" for c in grupos["reglamentaria"])
    assert grupos["reglamentaria"] and grupos["preventiva"]
    # el catálogo actual no trae informativas: el grupo existe aunque esté vacío
    assert "informativa" in grupos
    assert [c.id for c in grupos["especiales"]] == [CLASE_OTRA, CLASE_NO_SENAL, CLASE_OMITIDA]
    ids = [c.id for g in grupos.values() for c in g]
    assert len(ids) == len(set(ids))
    assert "velocidad_no_verificada" not in ids
    pare = next(c for c in grupos["reglamentaria"] if c.id == "pare")
    assert pare.nombre == catalogo.entradas["pare"].nombre


def test_duplicados_por_sha1(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"uno")
    (tmp_path / "b.JPG").write_bytes(b"uno")
    (tmp_path / "c.png").write_bytes(b"dos")
    (tmp_path / "d.txt").write_bytes(b"x")
    fotos = listar_fotos(tmp_path)
    assert [f.name for f in fotos] == ["a.jpg", "b.JPG", "c.png"]
    dup = duplicadas_por_sha1((f.name, sha1_archivo(f)) for f in fotos)
    assert list(dup.values()) == [["a.jpg", "b.JPG"]]
    indice = [_recorte(1, "a.jpg", sha1_archivo(fotos[0]))]
    assert fotos_indexadas(indice) == {sha1_archivo(fotos[0]): "a.jpg"}


def test_stem_unico_evita_choque_entre_fotos_distintas():
    indice = [_recorte(1, "a.jpg", "aa")]
    assert stem_unico("a", "aa", indice) == "a"
    assert stem_unico("a", "bbbbbbbbcc", indice) == "a-bbbbbbbb"


def test_rutas_por_defecto_usan_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADVISION_DATASETS", str(tmp_path))
    r = rutas_por_defecto()
    assert r.base == tmp_path / "co"
    assert r.fotos == tmp_path / "co" / "fotos" and r.recortes == tmp_path / "co" / "recortes"
    assert r.indice.name == "indice.csv" and r.etiquetas.name == "etiquetas.csv"


# ------------------------------------------------------------------- app Streamlit

def _preparar_app(tmp_path: Path, n: int = 3):
    rutas = rutas_por_defecto(tmp_path / "co")
    rutas.fotos.mkdir(parents=True)
    rutas.recortes.mkdir(parents=True)
    Image.new("RGB", (200, 150), (90, 90, 90)).save(rutas.fotos / "calle.jpg")
    recortes = []
    for i in range(1, n + 1):
        nombre = nombre_recorte("calle", i)
        Image.new("RGB", (40, 50), (200, 30 * i, 0)).save(rutas.recortes / nombre)
        recortes.append(Recorte(nombre, "calle.jpg", (10, 10, 50, 60), 0.4, 40, 50, "h"))
    agregar_al_indice(rutas.indice, recortes)
    return rutas


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(RAIZ / "scripts" / "etiquetador.py"), default_timeout=60)


def test_app_arranca_guarda_y_avanza(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADVISION_DATASETS", str(tmp_path))
    rutas = _preparar_app(tmp_path)
    at = _app().run()
    assert not at.exception
    assert "Etiquetados: 0 / 3" in str(at.main.get("progress")[0].proto.text)
    assert any("calle__01.png" in str(i.proto.caption) for i in at.main.get("imgs").__iter__()) \
        or "calle__01.png" in "".join(str(e.proto) for e in at.main)
    at.button(key=f"clase_{CLASE_NO_SENAL}").click().run()
    assert not at.exception
    filas = leer_etiquetas(rutas.etiquetas)
    assert [(f.recorte, f.clase) for f in filas] == [("calle__01.png", CLASE_NO_SENAL)]
    assert "Etiquetados: 1 / 3" in str(at.main.get("progress")[0].proto.text)
    assert [s.value for s in at.success] == [f"Guardado: calle__01.png → NO es una señal"]
    # el siguiente recorte es el 02: etiquetarlo con una clase del catálogo
    at.button(key="clase_pare").click().run()
    assert [f.recorte for f in leer_etiquetas(rutas.etiquetas)][-1] == "calle__02.png"
    # deshacer devuelve el 02 a la cola
    at.button(key="deshacer").click().run()
    assert [f.recorte for f in leer_etiquetas(rutas.etiquetas)] == ["calle__01.png"]
    assert not at.exception


def test_app_sin_indice_muestra_instrucciones(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADVISION_DATASETS", str(tmp_path))
    at = _app().run()
    assert not at.exception
    assert any("extraer_recortes.py" in i.value for i in at.info)


def test_app_corregir_abre_dialogo_con_las_clases(tmp_path, monkeypatch):
    """Pulsar «Corregir» bajo un recorte viejo abre el diálogo con los botones de clase.

    El clic *dentro* del diálogo es una re-ejecución de fragmento que AppTest no emula; el
    guardado se cubre en test_corregir_gana_la_ultima y a mano en el navegador.
    """
    monkeypatch.setenv("ROADVISION_DATASETS", str(tmp_path))
    rutas = _preparar_app(tmp_path)
    at = _app().run()
    at.button(key=f"clase_{CLASE_NO_SENAL}").click().run()   # 01 = no_senal
    at.button(key="clase_pare").click().run()                # 02 = pare
    assert not at.exception
    at.button(key="rev_sel_calle__01.png").click().run()
    assert not at.exception
    assert at.button(key="rev_clase_pare") is not None       # el diálogo mostró las clases
    # abrir el diálogo no cambia nada por sí solo ni altera la cola
    assert [(f.recorte, f.clase) for f in leer_etiquetas(rutas.etiquetas)] == [
        ("calle__01.png", CLASE_NO_SENAL), ("calle__02.png", "pare")]


def test_corregir_gana_la_ultima(tmp_path):
    """Una corrección es una fila nueva del mismo recorte: gana la última y la cola no cambia."""
    rutas = _preparar_app(tmp_path)
    guardar_etiqueta(rutas.etiquetas, "calle__01.png", "calle.jpg", CLASE_NO_SENAL)
    guardar_etiqueta(rutas.etiquetas, "calle__02.png", "calle.jpg", "pare")
    guardar_etiqueta(rutas.etiquetas, "calle__01.png", "calle.jpg", "pare")   # corrección
    etiquetas = leer_etiquetas(rutas.etiquetas)
    assert {r: e.clase for r, e in etiquetas_vigentes(etiquetas).items()} == {
        "calle__01.png": "pare", "calle__02.png": "pare"}
    assert conteos_por_clase(etiquetas) == {"pare": 2}
    assert siguiente_sin_etiquetar(leer_indice(rutas.indice), etiquetas).recorte == "calle__03.png"


def test_app_revisar_pagina_mas_alla_de_30(tmp_path, monkeypatch):
    """Con 35 etiquetas, la página 2 llega a las más antiguas (más atrás que 30)."""
    monkeypatch.setenv("ROADVISION_DATASETS", str(tmp_path))
    rutas = _preparar_app(tmp_path, n=35)
    for i in range(1, 36):
        guardar_etiqueta(rutas.etiquetas, nombre_recorte("calle", i), "calle.jpg",
                         CLASE_NO_SENAL, marca_tiempo=f"2026-11-06T10:00:{i:02d}+00:00")
    at = _app().run()
    assert not at.exception
    assert any("1–30" in c.value for c in at.caption)
    keys_p1 = {b.key for b in at.button if b.key and b.key.startswith("rev_sel_")}
    assert len(keys_p1) == 30 and "rev_sel_calle__01.png" not in keys_p1
    at.number_input(key="rev_pagina").set_value(2).run()
    assert not at.exception
    assert any("31–35" in c.value for c in at.caption)
    keys_p2 = {b.key for b in at.button if b.key and b.key.startswith("rev_sel_")}
    assert len(keys_p2) == 5 and "rev_sel_calle__01.png" in keys_p2
