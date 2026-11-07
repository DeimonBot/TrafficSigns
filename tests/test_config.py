from roadvision.config import Config, Umbrales
from roadvision.contracts import Categoria, Estado, Fuente


def test_config_valores_del_diseno():
    c = Config()
    assert c.modelo_detector == "google/owlv2-base-patch16-ensemble"
    assert c.modelo_clasificador == "kelvinandreas/vit-traffic-sign-GTSRB"
    assert c.modelo_respaldo == "openai/clip-vit-base-patch32"
    assert c.consultas_detector == ("a traffic sign", "a road sign")
    assert c.umbrales == Umbrales(deteccion=0.15, gtsrb=0.80, clip_senal=0.50, clip_categoria=0.50)
    assert (c.margen_recorte, c.tamano_minimo_caja, c.iou_duplicados) == (0.10, 20, 0.5)
    assert (c.dispositivo, c.fp16) == ("auto", True)
    assert c.ruta_catalogo.is_absolute() and c.ruta_catalogo.name == "catalogo_senales.yaml"


def test_enums_en_minuscula():
    assert Estado.SOLO_CATEGORIA.value == "solo_categoria"
    assert Categoria.PREVENTIVA.value == "preventiva"
    assert Fuente.NINGUNA.value == "ninguna"
