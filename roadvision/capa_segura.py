"""Capa segura: SigLIP B/16 congelado + puerta + identidad + política de nombre.

Módulo de ejecución del diseño de `exp3` (cabezas logísticas entrenadas sobre embeddings
SigLIP). El `Decisor` del ViT/CLIP no se modifica: su decisión se reutiliza para la unión
de velocidades y para la categoría. La capa solo NOMBRA una señal cuando es muy segura; si
no, cae a "solo categoría" o "no reconocida". Nunca emite `Estado.APROXIMADA`.

Nota para el compañero: `contracts.Fuente` no tiene un valor para SigLIP y no se edita aquí;
por eso `FuenteSegura` lo reemplaza. Se pedirá añadir `SIGLIP = "siglip"` a `Fuente` y
entonces `FuenteSegura` se puede eliminar.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from roadvision.cajas import filtrar_cajas
from roadvision.catalog import Catalogo, cargar_catalogo
from roadvision.config import RAIZ, Config, Umbrales
from roadvision.contracts import Estado, Fuente, Resultado
from roadvision.decision import Decision, Decisor
from roadvision.device import Dispositivo, elegir_dispositivo, sincronizar
from roadvision.imaging import normalizar, recortar
from roadvision.pipeline import ETAPAS, CargaModelo, Pipeline, _cargar, _contar
from roadvision.timing import Cronometro

RUTA_CAPA = RAIZ / "data" / "capa_segura"
COMANDO_DESCARGA = "python scripts/descargar_siglip.py"
ID_OTRA = "otra_senal"
TAMANO_LOTE = 32

# Clases con pareja espejo (izquierda/derecha): SigLIP las confunde hasta en un 28 %, así que
# aunque la capa las vea con seguridad no se nombra la dirección: solo la categoría.
GUARDA_DIRECCION = frozenset({
    "curva_izquierda", "curva_derecha", "no_girar_izquierda", "no_girar_derecha"})

PUERTA_UNION_VELOCIDAD = 0.9   # puerta mínima para aceptar una velocidad del ViT
PUERTA_CATEGORIA = 0.5         # puerta mínima para dar al menos la categoría


class FuenteSegura(str, Enum):
    """Origen de la decisión en modo seguro (ver nota del módulo sobre `Fuente`)."""

    SIGLIP = "siglip"
    GTSRB = "gtsrb"


class ErrorCapaSegura(RuntimeError):
    """Faltan artefactos de la capa segura o SigLIP no está disponible."""


@dataclass(frozen=True)
class Cabezas:
    gate_mu: np.ndarray
    gate_sd: np.ndarray
    gate_w: np.ndarray
    gate_b: float
    id_mu: np.ndarray
    id_sd: np.ndarray
    id_W: np.ndarray
    id_b: np.ndarray
    T: float
    clases: tuple[str, ...]


@dataclass(frozen=True)
class UmbralesCapa:
    puerta: float = 0.95
    identidad: float = 0.6
    margen: float = 0.6


@dataclass(frozen=True)
class MetaCapa:
    encoder: str
    revision: str | None
    umbrales: UmbralesCapa
    version: str


# ---------------------------------------------------------------- lógica pura (solo numpy)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predecir(embeddings, cabezas: Cabezas, umbrales: UmbralesCapa):
    """Reimplementación de `ref.predecir` (exp3/final). Devuelve
    (puerta, clase, prob, margen, nombra): arrays de longitud n.

    `nombra` = puerta>=tp & prob>=ti & margen>=tm & clase != 'otra_senal' (sin guarda de
    dirección; esa la aplica `decidir_recorte`)."""
    h = cabezas
    E = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    pg = 1.0 / (1.0 + np.exp(-(((E - h.gate_mu) / h.gate_sd) @ h.gate_w + h.gate_b)))
    P = _softmax((((E - h.id_mu) / h.id_sd) @ h.id_W + h.id_b) / h.T)
    o = np.argsort(-P, 1)
    r = np.arange(len(P))
    p1, p2 = P[r, o[:, 0]], P[r, o[:, 1]]
    clase = np.array(h.clases)[o[:, 0]].astype(str)
    nombra = ((pg >= umbrales.puerta) & (p1 >= umbrales.identidad)
              & ((p1 - p2) >= umbrales.margen) & (clase != ID_OTRA))
    return pg, clase, p1, p1 - p2, nombra


_NO_RECONOCIDA = Decision(Estado.NO_RECONOCIDA, Fuente.NINGUNA, None, None, 0.0)


def decidir_recorte(
    catalogo: Catalogo,
    puerta: float,
    clase: str,
    prob: float,
    nombra: bool,
    vit: Decision,
) -> Decision:
    """Política por recorte, en orden; no emite nunca `Estado.APROXIMADA`."""
    # (1) La capa nombra, salvo clases de dirección (solo categoría).
    if nombra and clase in catalogo.entradas:
        entrada = catalogo.entradas[clase]
        if clase in GUARDA_DIRECCION:
            return Decision(Estado.SOLO_CATEGORIA, FuenteSegura.SIGLIP,  # type: ignore[arg-type]
                            None, entrada.categoria, prob)
        return Decision(Estado.RECONOCIDA, FuenteSegura.SIGLIP,  # type: ignore[arg-type]
                        entrada.id, entrada.categoria, prob)
    # (2) Unión: velocidad reconocida por el ViT con puerta alta.
    if (vit.estado is Estado.RECONOCIDA and vit.id_catalogo is not None
            and vit.id_catalogo.startswith("velocidad_") and puerta >= PUERTA_UNION_VELOCIDAD):
        return Decision(vit.estado, FuenteSegura.GTSRB,  # type: ignore[arg-type]
                        vit.id_catalogo, vit.categoria, vit.confianza)
    # (3) Es una señal pero no se identifica: solo la categoría que dio el decisor.
    if puerta >= PUERTA_CATEGORIA and vit.categoria is not None:
        fuente = FuenteSegura.GTSRB if vit.fuente is Fuente.GTSRB else vit.fuente
        return Decision(Estado.SOLO_CATEGORIA, fuente,  # type: ignore[arg-type]
                        None, vit.categoria, vit.confianza)
    # (4) Resto.
    return _NO_RECONOCIDA


# ---------------------------------------------------------------- carga de artefactos


def cargar_meta(ruta: Path = RUTA_CAPA) -> MetaCapa:
    try:
        meta = json.loads((ruta / "meta.json").read_text(encoding="utf-8"))
        u = meta["umbrales"]
        return MetaCapa(
            encoder=meta["encoder"], revision=meta.get("revision_encoder"),
            umbrales=UmbralesCapa(u["tp_puerta"], u["ti_prob_identidad"], u["tm_margen"]),
            version=meta["version"])
    except (OSError, KeyError, ValueError) as exc:
        raise ErrorCapaSegura(
            f"No se pudo leer {ruta / 'meta.json'} ({exc}). Faltan los artefactos de la capa "
            f"segura: restaure data/capa_segura/ del repositorio.") from exc


def cargar_cabezas(catalogo: Catalogo, ruta: Path = RUTA_CAPA) -> Cabezas:
    """Carga `cabezas.npz` verificando su sha256 (meta.json) y que las clases sean del catálogo."""
    archivo = ruta / "cabezas.npz"
    try:
        meta = json.loads((ruta / "meta.json").read_text(encoding="utf-8"))
        datos = archivo.read_bytes()
    except (OSError, ValueError) as exc:
        raise ErrorCapaSegura(
            f"Faltan artefactos de la capa segura en {ruta} ({exc}). "
            f"Restaure data/capa_segura/ del repositorio.") from exc
    esperado = meta.get("sha256_cabezas")
    real = hashlib.sha256(datos).hexdigest()
    if esperado != real:
        raise ErrorCapaSegura(
            f"{archivo.name} no coincide con el sha256 de meta.json (esperado {esperado}, "
            f"real {real}): el archivo cambió o está corrupto.")
    import io

    z = np.load(io.BytesIO(datos), allow_pickle=False)
    clases = tuple(str(c) for c in z["clases"])
    invalidas = [c for c in clases if c != ID_OTRA and c not in catalogo.entradas]
    if invalidas:
        raise ErrorCapaSegura(f"Clases de la capa segura que no están en el catálogo: {invalidas}")
    return Cabezas(
        gate_mu=z["gate_mu"].astype(np.float64), gate_sd=z["gate_sd"].astype(np.float64),
        gate_w=z["gate_w"], gate_b=float(z["gate_b"]),
        id_mu=z["id_mu"].astype(np.float64), id_sd=z["id_sd"].astype(np.float64),
        id_W=z["id_W"], id_b=z["id_b"], T=float(z["T"]), clases=clases)


# ---------------------------------------------------------------- SigLIP


class EmbedderSiglip:
    """Embedding de imagen SigLIP B/16 (768, SIN normalizar), igual que exp3/enc.py.

    Usa solo la torre de visión (`SiglipVisionModel.pooler_output`), que es idéntica a
    `SiglipModel.get_image_features` (SigLIP no tiene proyección extra) y ahorra la torre de
    texto en VRAM; la prueba de regresión lo verifica contra los embeddings guardados.
    """

    def __init__(self, modelo_id: str, dispositivo: Dispositivo, revision: str | None = None):
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self._dispositivo = dispositivo
        try:
            self._procesador = SiglipImageProcessor.from_pretrained(
                modelo_id, revision=revision, local_files_only=True)
            self._modelo = (
                SiglipVisionModel.from_pretrained(
                    modelo_id, revision=revision, local_files_only=True,
                    torch_dtype=dispositivo.dtype)
                .to(dispositivo.nombre).eval())
        except OSError as exc:
            raise ErrorCapaSegura(
                f"SigLIP ({modelo_id}, revisión {revision}) no está descargado. "
                f"Ejecute: {COMANDO_DESCARGA}") from exc

    def embeber(self, recortes: list[Image.Image], tamano_lote: int = TAMANO_LOTE) -> np.ndarray:
        import torch

        salida = []
        for a in range(0, len(recortes), tamano_lote):
            lote = [r.convert("RGB") for r in recortes[a:a + tamano_lote]]
            x = self._procesador(images=lote, return_tensors="pt")["pixel_values"].to(
                device=self._dispositivo.nombre, dtype=self._dispositivo.dtype)
            with torch.inference_mode():
                o = self._modelo(pixel_values=x)
            t = o.pooler_output  # (n, 768), sin normalizar
            salida.append(t.float().cpu().numpy())
        return np.concatenate(salida) if salida else np.zeros((0, 768))


# ---------------------------------------------------------------- pipeline


class PipelineSeguro(Pipeline):
    def __init__(
        self,
        config: Config,
        detector,
        clasificador,
        respaldo,
        catalogo: Catalogo,
        dispositivo: Dispositivo,
        embedder,
        cabezas: Cabezas,
        umbrales_capa: UmbralesCapa | None = None,
        carga: dict[str, CargaModelo] | None = None,
    ) -> None:
        super().__init__(config, detector, clasificador, respaldo, catalogo, dispositivo, carga)
        self.embedder = embedder
        self.cabezas = cabezas
        self.umbrales_capa = umbrales_capa or UmbralesCapa()

    def analizar(self, imagen: Image.Image, umbrales: Umbrales | None = None) -> Resultado:
        u = umbrales or self.config.umbrales
        cfg = self.config
        crono = Cronometro(sincronizar=lambda: sincronizar(self.dispositivo))

        with crono.etapa("total"):
            with crono.etapa("preproceso"):
                img = normalizar(imagen)
            with crono.etapa("deteccion"):
                detecciones = filtrar_cajas(
                    self.detector.detectar(img, u.deteccion),
                    u.deteccion, cfg.tamano_minimo_caja, cfg.iou_duplicados)
            recortes: list[Image.Image] = []
            predicciones = []
            salida = None
            with crono.etapa("clasificacion"):
                recortes = [recortar(img, d.caja, cfg.margen_recorte) for d in detecciones]
                if recortes:
                    predicciones = self.clasificador.clasificar(recortes)
                    embeddings = self.embedder.embeber(recortes)
                    salida = predecir(embeddings, self.cabezas, self.umbrales_capa)
            with crono.etapa("respaldo"):
                if recortes:
                    vit = self._decisor.decidir(recortes, predicciones, u)
                    pg, clase, p1, _, nombra = salida  # type: ignore[misc]
                    decisiones = [
                        decidir_recorte(self.catalogo, float(pg[i]), str(clase[i]),
                                        float(p1[i]), bool(nombra[i]), vit[i])
                        for i in range(len(recortes))]
                else:
                    decisiones = []

        orden = sorted(range(len(detecciones)), key=lambda i: detecciones[i].caja[:2])
        senales = [
            self._senal(n, detecciones[i], recortes[i], decisiones[i])
            for n, i in enumerate(orden, start=1)
        ]
        tiempos = {etapa: 0.0 for etapa in ETAPAS} | crono.tiempos_ms
        return Resultado(
            senales=senales,
            conteo_por_nombre=_contar(s.nombre for s in senales),
            conteo_por_categoria=_contar(
                s.categoria.value if s.categoria else "sin_categoria" for s in senales),
            tiempos_ms=tiempos,
            ancho=img.width,
            alto=img.height,
            dispositivo=self.dispositivo.descripcion,
        )

    def calentar(self) -> dict[str, float]:
        tiempos = super().calentar()
        crono = Cronometro(sincronizar=lambda: sincronizar(self.dispositivo))
        with crono.etapa("clasificacion"):
            self.embedder.embeber([Image.new("RGB", (64, 64))])
        tiempos["clasificacion"] = tiempos.get("clasificacion", 0.0) + crono.tiempos_ms[
            "clasificacion"]
        return tiempos


def construir_pipeline_seguro(
    config: Config | None = None,
    ruta: Path = RUTA_CAPA,
    fabrica_embedder: Callable[[Dispositivo, str | None, str], object] | None = None,
) -> PipelineSeguro:
    """Como `construir_pipeline`, más SigLIP (`carga['capa_segura']`).

    Lanza `ErrorCapaSegura` (en español, con el comando a correr) si faltan los artefactos o
    SigLIP no está descargado."""
    from roadvision.classifier import ClasificadorGtsrb
    from roadvision.detector import DetectorOwlv2
    from roadvision.fallback import RespaldoClip

    config = config or Config()
    catalogo = cargar_catalogo(config.ruta_catalogo)
    meta = cargar_meta(ruta)
    cabezas = cargar_cabezas(catalogo, ruta)
    if meta.revision is None:
        raise ErrorCapaSegura(
            "meta.json no fija la revisión de SigLIP. Ejecute: " + COMANDO_DESCARGA)
    dispositivo = elegir_dispositivo(config.dispositivo, config.fp16)
    # SigLIP primero: si no está descargado se falla antes de cargar los demás modelos.
    fabrica = fabrica_embedder or (
        lambda d, rev, mid: EmbedderSiglip(mid, d, rev))
    embedder, c_sig = _cargar(dispositivo, lambda: fabrica(dispositivo, meta.revision,
                                                           meta.encoder))
    detector, c_det = _cargar(dispositivo, lambda: DetectorOwlv2(
        config.modelo_detector, config.consultas_detector, dispositivo))
    clasificador, c_cls = _cargar(dispositivo, lambda: ClasificadorGtsrb(
        config.modelo_clasificador, dispositivo))
    respaldo, c_res = _cargar(dispositivo, lambda: RespaldoClip(
        config.modelo_respaldo, dispositivo))
    return PipelineSeguro(
        config, detector, clasificador, respaldo, catalogo, dispositivo,  # type: ignore[arg-type]
        embedder, cabezas, meta.umbrales,
        carga={"detector": c_det, "clasificador": c_cls, "respaldo": c_res,
               "capa_segura": c_sig})


def modo_seguro_disponible(ruta: Path = RUTA_CAPA) -> tuple[bool, str]:
    """(disponible, motivo): artefactos presentes y SigLIP en la caché local de HF."""
    try:
        meta = cargar_meta(ruta)
        cargar_cabezas(cargar_catalogo(Config().ruta_catalogo), ruta)
    except ErrorCapaSegura as exc:
        return False, str(exc)
    from huggingface_hub import try_to_load_from_cache

    # Solo lo que usa el embedder (no el snapshot completo: el script de descarga omite pesos
    # redundantes como pytorch_model.bin cuando existe model.safetensors).
    for archivo in ("config.json", "preprocessor_config.json", "model.safetensors"):
        ruta_hf = try_to_load_from_cache(meta.encoder, archivo, revision=meta.revision)
        if not isinstance(ruta_hf, str):
            return False, f"SigLIP no está descargado. Ejecute: {COMANDO_DESCARGA}"
    return True, ""
