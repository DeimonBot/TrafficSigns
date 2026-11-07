"""Pipeline completo: imagen → detector → clasificador → decisor → Resultado."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from PIL import Image

from roadvision.cajas import filtrar_cajas
from roadvision.catalog import Catalogo, cargar_catalogo
from roadvision.config import Config, Umbrales
from roadvision.contracts import (
    Categoria,
    Clasificador,
    Detector,
    Estado,
    Resultado,
    Respaldo,
    Senal,
)
from roadvision.decision import Decision, Decisor
from roadvision.device import Dispositivo, elegir_dispositivo, sincronizar
from roadvision.imaging import normalizar, recortar
from roadvision.timing import Cronometro

ETAPAS = ("total", "preproceso", "deteccion", "clasificacion", "respaldo")


@dataclass(frozen=True)
class CargaModelo:
    tiempo_ms: float
    vram_mb: float  # aumento de torch.cuda.memory_allocated() al cargar; 0.0 en CPU


class Pipeline:
    def __init__(
        self,
        config: Config,
        detector: Detector,
        clasificador: Clasificador,
        respaldo: Respaldo,
        catalogo: Catalogo,
        dispositivo: Dispositivo,
        carga: dict[str, CargaModelo] | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.clasificador = clasificador
        self.respaldo = respaldo
        self.catalogo = catalogo
        self.dispositivo = dispositivo
        self.carga = carga or {}
        self._decisor = Decisor(catalogo, respaldo)

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
            with crono.etapa("clasificacion"):
                recortes = [recortar(img, d.caja, cfg.margen_recorte) for d in detecciones]
                if recortes:
                    predicciones = self.clasificador.clasificar(recortes)
            with crono.etapa("respaldo"):
                decisiones = self._decisor.decidir(recortes, predicciones, u) if recortes else []

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

    def _senal(self, numero: int, deteccion, recorte: Image.Image, d: Decision) -> Senal:
        cat = self.catalogo
        if d.id_catalogo is not None:
            entrada = cat.entradas[d.id_catalogo]
            nombre, significado = entrada.nombre, entrada.significado
        elif d.estado is Estado.SOLO_CATEGORIA and d.categoria is not None:
            info = cat.categorias[d.categoria]
            nombre, significado = info.nombre, info.significado
        else:
            nombre, significado = cat.nombre_no_reconocida, None
        return Senal(
            numero=numero, caja=deteccion.caja, recorte=recorte, estado=d.estado,
            fuente=d.fuente, id_catalogo=d.id_catalogo, nombre=nombre,
            categoria=d.categoria, significado=significado, confianza=d.confianza,
            puntaje_deteccion=deteccion.puntaje)

    def calentar(self) -> dict[str, float]:
        """Una pasada sintética por cada modelo; devuelve ms por etapa."""
        crono = Cronometro(sincronizar=lambda: sincronizar(self.dispositivo))
        imagen = Image.new("RGB", (640, 480))
        crop = Image.new("RGB", (64, 64))
        textos = [self.catalogo.categorias[Categoria.REGLAMENTARIA].clip,
                  self.catalogo.clip_otro]
        with crono.etapa("deteccion"):
            self.detector.detectar(imagen, self.config.umbrales.deteccion)
        with crono.etapa("clasificacion"):
            self.clasificador.clasificar([crop])
        with crono.etapa("respaldo"):
            self.respaldo.puntuar([crop], textos)
        return dict(crono.tiempos_ms)


def _contar(claves) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for clave in claves:
        conteo[clave] = conteo.get(clave, 0) + 1
    return conteo


def _cargar(dispositivo: Dispositivo, fabrica) -> tuple[object, CargaModelo]:
    cuda = dispositivo.nombre == "cuda"
    antes = torch.cuda.memory_allocated() if cuda else 0
    inicio = time.perf_counter()
    modelo = fabrica()
    sincronizar(dispositivo)
    ms = (time.perf_counter() - inicio) * 1000
    vram = (torch.cuda.memory_allocated() - antes) / 2**20 if cuda else 0.0
    return modelo, CargaModelo(ms, vram)


def construir_pipeline(config: Config | None = None) -> Pipeline:
    from roadvision.classifier import ClasificadorGtsrb
    from roadvision.detector import DetectorOwlv2
    from roadvision.fallback import RespaldoClip

    config = config or Config()
    dispositivo = elegir_dispositivo(config.dispositivo, config.fp16)
    catalogo = cargar_catalogo(config.ruta_catalogo)
    detector, c_det = _cargar(dispositivo, lambda: DetectorOwlv2(
        config.modelo_detector, config.consultas_detector, dispositivo))
    clasificador, c_cls = _cargar(dispositivo, lambda: ClasificadorGtsrb(
        config.modelo_clasificador, dispositivo))
    respaldo, c_res = _cargar(dispositivo, lambda: RespaldoClip(
        config.modelo_respaldo, dispositivo))
    return Pipeline(
        config, detector, clasificador, respaldo, catalogo, dispositivo,  # type: ignore[arg-type]
        carga={"detector": c_det, "clasificador": c_cls, "respaldo": c_res})
