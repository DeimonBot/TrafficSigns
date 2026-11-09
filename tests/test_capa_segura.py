"""Pruebas de la capa segura: política (modelos simulados), predecir vs referencia, artefactos."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from roadvision import capa_segura as cs
from roadvision.capa_segura import (
    ErrorCapaSegura,
    FuenteSegura,
    UmbralesCapa,
    cargar_cabezas,
    cargar_meta,
    predecir,
)
from roadvision.contracts import Categoria, Estado, Prediccion
from roadvision.device import Dispositivo
from roadvision.metrics import construir_metricas
from tests.fakes import FakeRespaldo
from tests.fakes_capa import (
    ancho_recorte,
    cabezas_falsas,
    catalogo_real,
    e_nombra,
    e_puerta_060,
    e_puerta_093,
    e_puerta_baja,
    emb,
    pipeline_seguro_falso,
)

FIXT = Path(__file__).parent / "fixtures" / "capa_segura"
IMG = Image.new("RGB", (1000, 600))
STOP = Prediccion("Stop", 0.97)
V50 = Prediccion("Speed limit (50km/h)", 0.95)
PRIORIDAD = Prediccion("Priority road", 0.99)


def _una(embedding, pred, respaldo=None):
    p = pipeline_seguro_falso([(100, embedding, pred)], respaldo)
    [s] = p.analizar(IMG).senales
    return s


def _respaldo_v50():
    cat = catalogo_real()
    return FakeRespaldo(
        puntajes={ancho_recorte(100): {cat.entradas["velocidad_50"].clip: 5}},
        por_defecto={cat.clip_otro: 5})


# ---------- precondiciones de los vectores de prueba ----------

def test_vectores_de_prueba_tienen_las_propiedades_declaradas():
    pg, cl, p1, mg, nb = predecir(
        np.stack([e_nombra("pare"), e_puerta_093("pare"), e_puerta_060("pare"),
                  e_puerta_baja("pare")]), cabezas_falsas(), UmbralesCapa())
    assert list(cl) == ["pare"] * 4
    assert pg[0] >= 0.95 and 0.9 <= pg[1] < 0.95 and 0.5 <= pg[2] < 0.9 and pg[3] < 0.05
    assert list(nb) == [True, False, False, False]


# ---------- política: cada rama ----------

def test_rama1_la_capa_nombra():
    s = _una(e_nombra("pare"), STOP)
    assert (s.estado, s.id_catalogo, s.nombre) == (Estado.RECONOCIDA, "pare", "PARE")
    assert s.fuente is FuenteSegura.SIGLIP and s.categoria is Categoria.REGLAMENTARIA
    assert 0.9 < s.confianza <= 1.0


def test_rama1_la_capa_manda_sobre_el_vit():
    s = _una(e_nombra("resalto"), STOP)  # el ViT dice Stop al 97 %
    assert (s.estado, s.id_catalogo) == (Estado.RECONOCIDA, "resalto")


def test_rama1_guarda_de_direccion_no_nombra():
    cat = catalogo_real()
    s = _una(e_nombra("curva_izquierda"), PRIORIDAD)
    assert s.estado is Estado.SOLO_CATEGORIA and s.id_catalogo is None
    assert s.categoria is cat.entradas["curva_izquierda"].categoria
    assert s.nombre == cat.categorias[s.categoria].nombre
    assert s.fuente is FuenteSegura.SIGLIP


def test_rama2_union_velocidad_del_vit():
    s = _una(e_puerta_093("otra_senal"), V50, _respaldo_v50())
    assert (s.estado, s.id_catalogo, s.fuente) == (
        Estado.RECONOCIDA, "velocidad_50", FuenteSegura.GTSRB)
    assert s.confianza == pytest.approx(0.95)


def test_rama2_no_aplica_con_puerta_baja_ni_a_otras_senales():
    s = _una(e_puerta_060("otra_senal"), V50, _respaldo_v50())  # puerta 0.6 < 0.9
    assert (s.estado, s.id_catalogo, s.categoria) == (
        Estado.SOLO_CATEGORIA, None, Categoria.REGLAMENTARIA)
    s = _una(e_puerta_093("otra_senal"), STOP)  # Stop no es velocidad: no hay unión
    assert (s.estado, s.id_catalogo) == (Estado.SOLO_CATEGORIA, None)


def test_rama3_solo_categoria():
    s = _una(e_puerta_093("otra_senal"), STOP)
    assert s.estado is Estado.SOLO_CATEGORIA and s.id_catalogo is None
    assert s.categoria is Categoria.REGLAMENTARIA
    assert s.nombre == catalogo_real().categorias[Categoria.REGLAMENTARIA].nombre


def test_rama3_frontera_de_puerta_05():
    assert _una(e_puerta_060("otra_senal"), STOP).estado is Estado.SOLO_CATEGORIA
    assert _una(e_puerta_baja("otra_senal"), STOP).estado is Estado.NO_RECONOCIDA


def test_rama4_no_reconocida():
    s = _una(e_puerta_baja("pare"), STOP)
    assert (s.estado, s.id_catalogo, s.categoria, s.confianza) == (
        Estado.NO_RECONOCIDA, None, None, 0.0)
    assert s.nombre == catalogo_real().nombre_no_reconocida
    # Puerta alta pero el decisor tampoco da categoría: no reconocida.
    s = _una(e_puerta_093("otra_senal"), Prediccion("Priority road", 0.3))
    assert s.estado is Estado.NO_RECONOCIDA


def test_nunca_aproximada():
    # ViT poco seguro y CLIP muy seguro de una señal del catálogo: el Decisor daría APROXIMADA.
    cat = catalogo_real()
    clip_resalto = cat.entradas["resalto"].clip
    vectores = (e_nombra("pare"), e_puerta_093("otra_senal"), e_puerta_060("otra_senal"),
                e_puerta_baja("otra_senal"), e_nombra("curva_izquierda"))
    casos = [(100 + 10 * i, v, Prediccion("Priority road", 0.3)) for i, v in enumerate(vectores)]
    punt = {ancho_recorte(a): {clip_resalto: 9} for a, _, _ in casos}
    p = pipeline_seguro_falso(casos, FakeRespaldo(puntajes=punt, por_defecto={cat.clip_otro: 1}))
    estados = {s.estado for s in p.analizar(IMG).senales}
    assert Estado.APROXIMADA not in estados
    assert estados <= {Estado.RECONOCIDA, Estado.SOLO_CATEGORIA, Estado.NO_RECONOCIDA}


def test_margen_bajo_no_nombra():
    # Dos clases casi empatadas: probabilidad < ti y margen ~0 -> la capa no nombra.
    e = emb(1.0, pare=1.0, resalto=0.97)
    assert not predecir(e, cabezas_falsas(), UmbralesCapa())[4][0]
    s = _una(e, STOP)
    assert s.fuente is not FuenteSegura.SIGLIP and s.estado is Estado.SOLO_CATEGORIA


def test_varias_senales_orden_y_lote_unico():
    casos = [(100, e_nombra("pare"), STOP), (80, e_puerta_baja("pare"), STOP),
             (60, e_nombra("resalto"), PRIORIDAD)]
    p = pipeline_seguro_falso(casos)
    r = p.analizar(IMG)
    assert [s.numero for s in r.senales] == [1, 2, 3]
    assert [s.id_catalogo for s in r.senales] == ["pare", None, "resalto"]
    assert p.embedder.llamadas == [3]
    assert r.conteo_por_nombre["PARE"] == 1


def test_sin_detecciones_no_llama_a_siglip():
    p = pipeline_seguro_falso([])
    r = p.analizar(IMG)
    assert r.senales == [] and p.embedder.llamadas == []
    assert set(r.tiempos_ms) == {"total", "preproceso", "deteccion", "clasificacion", "respaldo"}


def test_tiempos_y_metricas_aceptan_los_estados():
    casos = [(100, e_nombra("pare"), STOP), (80, e_puerta_baja("pare"), STOP),
             (60, e_puerta_093("otra_senal"), STOP)]
    r = pipeline_seguro_falso(casos).analizar(IMG)
    assert set(r.tiempos_ms) == {"total", "preproceso", "deteccion", "clasificacion", "respaldo"}
    m = construir_metricas(r, [], precision="fp32")
    assert (m.reconocidas, m.solo_categoria, m.no_reconocidas, m.aproximadas) == (1, 1, 1, 0)


def test_calentar_suma_siglip_a_clasificacion():
    p = pipeline_seguro_falso([])
    t = p.calentar()
    assert set(t) == {"deteccion", "clasificacion", "respaldo"}
    assert p.embedder.llamadas == [1]


# ---------- predecir == referencia (ref.py de exp3/final) ----------

def test_predecir_igual_a_la_referencia():
    ref = np.load(FIXT / "referencia.npz", allow_pickle=False)
    h = cargar_cabezas(catalogo_real())
    pg, cl, p1, mg, nb = predecir(ref["emb"], h, UmbralesCapa(*ref["umbrales"]))
    assert len(pg) == 19 and ref["nombra"].sum() >= 5
    np.testing.assert_allclose(pg, ref["pg"], atol=1e-9)
    np.testing.assert_allclose(p1, ref["p1"], atol=1e-9)
    np.testing.assert_allclose(mg, ref["margen"], atol=1e-9)
    assert list(cl) == list(ref["clase"]) and list(nb) == list(ref["nombra"])


def test_umbrales_por_defecto_vienen_de_meta():
    meta = cargar_meta()
    assert meta.umbrales == UmbralesCapa(0.95, 0.6, 0.6)
    assert meta.revision and len(meta.revision) == 40


# ---------- artefactos ----------

def test_cabezas_validan_sha256(tmp_path):
    cat = catalogo_real()
    cargar_cabezas(cat)  # los reales cargan
    d = tmp_path / "capa"
    shutil.copytree(cs.RUTA_CAPA, d)
    with open(d / "cabezas.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(ErrorCapaSegura, match="sha256"):
        cargar_cabezas(cat, d)


def test_cabezas_con_clase_fuera_del_catalogo(tmp_path):
    d = tmp_path / "capa"
    shutil.copytree(cs.RUTA_CAPA, d)
    z = dict(np.load(d / "cabezas.npz"))
    z["clases"] = np.array(["inventada"] + list(z["clases"][1:]))
    np.savez(d / "cabezas.npz", **z)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    meta["sha256_cabezas"] = hashlib.sha256((d / "cabezas.npz").read_bytes()).hexdigest()
    (d / "meta.json").write_text(json.dumps(meta), "utf-8")
    with pytest.raises(ErrorCapaSegura, match="inventada"):
        cargar_cabezas(catalogo_real(), d)


def test_faltan_artefactos(tmp_path):
    with pytest.raises(ErrorCapaSegura, match="data/capa_segura"):
        cs.construir_pipeline_seguro(ruta=tmp_path)
    assert cs.modo_seguro_disponible(tmp_path)[0] is False


def test_siglip_no_descargado_dice_que_comando_correr():
    with pytest.raises(ErrorCapaSegura, match="descargar_siglip.py"):
        cs.EmbedderSiglip("google/siglip-base-patch16-224",
                          Dispositivo("cpu", torch.float32, "CPU"), revision="0" * 40)


def test_modo_seguro_disponible_con_revision_inexistente(tmp_path):
    d = tmp_path / "capa"
    shutil.copytree(cs.RUTA_CAPA, d)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    meta["revision_encoder"] = "0" * 40
    (d / "meta.json").write_text(json.dumps(meta), "utf-8")
    ok, motivo = cs.modo_seguro_disponible(d)
    assert ok is False and "descargar_siglip.py" in motivo
