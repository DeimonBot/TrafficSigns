"""Configuración del núcleo con los valores del diseño."""
import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def ruta_datasets() -> Path:
    """Carpeta de datasets: ROADVISION_DATASETS si está definida, si no `datasets/`."""
    valor = os.environ.get("ROADVISION_DATASETS")
    return Path(valor) if valor else RAIZ / "datasets"


@dataclass(frozen=True)
class Umbrales:
    deteccion: float = 0.15
    gtsrb: float = 0.80
    clip_senal: float = 0.50
    clip_categoria: float = 0.50


@dataclass(frozen=True)
class Config:
    modelo_detector: str = "google/owlv2-base-patch16-ensemble"
    modelo_clasificador: str = "kelvinandreas/vit-traffic-sign-GTSRB"
    modelo_respaldo: str = "openai/clip-vit-base-patch32"
    consultas_detector: tuple[str, ...] = ("a traffic sign", "a road sign")
    umbrales: Umbrales = field(default_factory=Umbrales)
    margen_recorte: float = 0.10
    tamano_minimo_caja: int = 20
    iou_duplicados: float = 0.5
    dispositivo: str = "auto"
    fp16: bool = True
    ruta_catalogo: Path = RAIZ / "data" / "catalogo_senales.yaml"
