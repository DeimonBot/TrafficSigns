"""Pruebas de `roadvision.metrics` (sin GPU ni torch real)."""
from __future__ import annotations

import io
import logging
from dataclasses import fields

import pytest
from PIL import Image
from rich.console import Console

from roadvision import metrics
from roadvision.contracts import Estado
from roadvision.metrics import (
    COLUMNAS_CSV,
    MetricasAnalisis,
    MuestraSistema,
    MuestreadorSistema,
    calentar_medido,
    construir_metricas,
    escribir_csv,
    leer_csv,
    reporte_terminal,
)
from tests.fakes_metrics import NvmlFalso, PipelineFalso, resultado


def _muestra(t=0.0, cpu=10.0, ram=100.0, gpu=None, vram=None, temp=None, pot=None):
    return MuestraSistema(t, cpu, ram, gpu, vram, temp, pot)


def _fila(**cambios) -> MetricasAnalisis:
    base = dict(
        marca_tiempo="2026-01-01T00:00:00.000+00:00", dispositivo="CPU", precision="fp32",
        calentamiento=False, ancho=640, alto=480, detecciones=0, reconocidas=0,
        aproximadas=0, solo_categoria=0, no_reconocidas=0, ms_total=10.0)
    return MetricasAnalisis(**{**base, **cambios})


@pytest.fixture
def sin_gpu(monkeypatch):
    monkeypatch.setattr(metrics, "_obtener_nvml", lambda: None)


def test_campos_igual_a_columnas():
    assert {f.name for f in fields(MetricasAnalisis)} == set(COLUMNAS_CSV)
    assert tuple(_fila().a_fila()) == COLUMNAS_CSV


def test_etapas_coinciden_con_pipeline():
    pytest.importorskip("torch")
    from roadvision.pipeline import ETAPAS

    assert metrics.ETAPAS == ETAPAS


def test_construir_metricas_conteos_y_tiempos():
    estados = [Estado.RECONOCIDA, Estado.RECONOCIDA, Estado.APROXIMADA,
               Estado.SOLO_CATEGORIA, Estado.NO_RECONOCIDA]
    m = construir_metricas(resultado(estados, 20.0), [], precision="fp16")
    assert (m.reconocidas, m.aproximadas, m.solo_categoria, m.no_reconocidas) == (2, 1, 1, 1)
    assert m.detecciones == 5 == (
        m.reconocidas + m.aproximadas + m.solo_categoria + m.no_reconocidas)
    assert m.ms_total == 20.0 and m.ms_deteccion == 10.0 and m.ms_respaldo == 3.0
    assert m.imagenes_por_s == pytest.approx(1000 / 20.0)
    assert m.precision == "fp16" and not m.calentamiento
    assert (m.ancho, m.alto) == (640, 480)
    assert m.muestras_gpu == 0 and m.gpu_pct_pico is None and m.cpu_pct_prom is None


def test_construir_metricas_pico_y_promedio():
    muestras = [
        _muestra(cpu=0.0, ram=100, gpu=10, vram=500, temp=50, pot=60),
        _muestra(cpu=30.0, ram=150, gpu=50, vram=700, temp=65, pot=100),
        _muestra(cpu=120.0, ram=120, gpu=30, vram=600, temp=60, pot=80),
    ]
    m = construir_metricas(
        resultado([]), muestras, precision="fp32", vram_pico_torch_mb=321.0,
        cpu_pct_prom=120.0)
    assert m.muestras_gpu == 3
    assert m.gpu_pct_pico == 50 and m.gpu_pct_prom == pytest.approx(30)
    assert m.vram_usada_mb == 700 and m.vram_pico_torch_mb == 321.0
    assert m.temperatura_c_max == 65 and m.potencia_w_prom == pytest.approx(80)
    assert m.ram_proceso_mb_pico == 150
    assert m.cpu_pct_prom == 120.0  # viene del kw, no de las muestras


def test_construir_metricas_sin_cpu_pct_prom_queda_none():
    muestras = [_muestra(cpu=50.0), _muestra(cpu=60.0)]
    assert construir_metricas(resultado([]), muestras, precision="fp32").cpu_pct_prom is None


def test_precision_exacta():
    pipe = PipelineFalso()
    for dtype, esperado in [("torch.float16", "fp16"), ("torch.float32", "fp32"),
                            ("torch.bfloat16", "bfloat16")]:
        pipe.dispositivo.dtype = dtype
        assert metrics._precision(pipe) == esperado


def test_construir_metricas_total_cero_sin_imagenes_por_s():
    assert construir_metricas(resultado([], 0.0), [], precision="fp32").imagenes_por_s is None


def test_muestreador_sin_gpu(sin_gpu):
    with MuestreadorSistema(intervalo_s=0.01) as mu:
        pass
    assert not mu.gpu_disponible
    assert len(mu.muestras) >= 2  # entrada + salida
    assert all(m.gpu_pct is None and m.vram_usada_mb is None for m in mu.muestras)
    assert all(m.ram_proceso_mb > 0 for m in mu.muestras)


def test_muestreador_toma_muestras_del_hilo(sin_gpu):
    import time

    with MuestreadorSistema(intervalo_s=0.01) as mu:
        time.sleep(0.15)
    assert len(mu.muestras) > 3
    assert not mu._hilo.is_alive()


def test_muestreador_no_tapa_excepciones(sin_gpu):
    with pytest.raises(ValueError):
        with MuestreadorSistema(intervalo_s=0.01):
            raise ValueError("boom")


def test_muestreador_con_gpu_falsa(monkeypatch):
    monkeypatch.setattr(metrics, "_obtener_nvml", lambda: NvmlFalso())
    with MuestreadorSistema(intervalo_s=0.01) as mu:
        pass
    assert mu.gpu_disponible
    m = construir_metricas(resultado([]), mu.muestras, precision="fp32")
    assert m.muestras_gpu >= 2
    assert m.gpu_pct_pico == 40 and m.vram_usada_mb == 512
    assert m.temperatura_c_max == 60 and m.potencia_w_prom == pytest.approx(90)


def test_muestreador_apaga_gpu_si_nvml_falla(monkeypatch):
    class Roto(NvmlFalso):
        def nvmlDeviceGetUtilizationRates(self, h):
            raise self.NVMLError("fallo")

    monkeypatch.setattr(metrics, "_obtener_nvml", lambda: Roto())
    with MuestreadorSistema(intervalo_s=0.01) as mu:
        pass
    assert not mu.gpu_disponible
    assert all(m.gpu_pct is None for m in mu.muestras)


def test_analizar_medido_con_pipeline_falso(sin_gpu):
    pipe = PipelineFalso([Estado.RECONOCIDA, Estado.APROXIMADA])
    res, m = metrics.analizar_medido(pipe, Image.new("RGB", (10, 10)))
    assert pipe.analizados == 1
    assert res.ancho == 640 and m.detecciones == 2
    assert m.precision == "fp32" and m.vram_pico_torch_mb is None
    assert m.cpu_pct_prom is not None and m.ram_proceso_mb_pico is not None
    assert m.cpu_pct_prom >= 0.0


def test_csv_ida_y_vuelta(tmp_path):
    ruta = tmp_path / "logs" / "metricas.csv"
    escribir_csv(_fila(detecciones=3, reconocidas=3, ms_total=12.5), ruta)
    escribir_csv(_fila(detecciones=1, solo_categoria=1, ms_total=8.0), ruta)
    texto = ruta.read_text(encoding="utf-8").splitlines()
    assert texto[0].split(",") == list(COLUMNAS_CSV)
    assert len(texto) == 3
    df = leer_csv(ruta)
    assert list(df.columns) == list(COLUMNAS_CSV)
    assert len(df) == 2
    assert df["detecciones"].tolist() == [3, 1]
    assert df["ms_total"].tolist() == [12.5, 8.0]
    assert df["gpu_pct_pico"].isna().all()  # None -> celda vacía -> NaN
    assert df["calentamiento"].dtype == bool


def test_leer_csv_excluye_calentamiento_por_defecto(tmp_path):
    ruta = tmp_path / "m.csv"
    escribir_csv(_fila(calentamiento=True, ms_total=None), ruta)
    escribir_csv(_fila(), ruta)
    assert len(leer_csv(ruta)) == 1
    assert not leer_csv(ruta)["calentamiento"].any()
    assert len(leer_csv(ruta, incluir_calentamiento=True)) == 2


def test_leer_csv_inexistente_es_vacio(tmp_path):
    df = leer_csv(tmp_path / "no_existe.csv")
    assert df.empty and list(df.columns) == list(COLUMNAS_CSV)


def test_cabecera_distinta_archiva_el_viejo(tmp_path, caplog):
    ruta = tmp_path / "metricas.csv"
    ruta.write_text("otra,cabecera\n1,2\n", encoding="utf-8", newline="")
    with caplog.at_level(logging.WARNING):
        escribir_csv(_fila(), ruta)
    archivados = list(tmp_path.glob("metricas_*.csv"))
    assert len(archivados) == 1
    assert archivados[0].read_text(encoding="utf-8") == "otra,cabecera\n1,2\n"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    assert lineas[0].split(",") == list(COLUMNAS_CSV)
    assert len(lineas) == 2  # cabecera + 1 fila
    assert "cabecera" in caplog.text
    escribir_csv(_fila(), ruta)  # ya coincide: sigue en el archivo canónico
    assert len(ruta.read_text(encoding="utf-8").splitlines()) == 3
    assert len(list(tmp_path.glob("metricas_*.csv"))) == 1


def test_permission_error_avisa_sin_lanzar(tmp_path, monkeypatch, caplog):
    ruta = tmp_path / "metricas.csv"
    original = type(ruta).open

    def abrir(self, *a, **k):
        raise PermissionError("bloqueado")

    monkeypatch.setattr(type(ruta), "open", abrir)
    with caplog.at_level(logging.WARNING):
        escribir_csv(_fila(), ruta)
    monkeypatch.setattr(type(ruta), "open", original)
    assert "Excel" in caplog.text


def test_registrar_no_rompe_si_falla_escritura(tmp_path, monkeypatch):
    def falla(*a, **k):
        raise OSError("disco lleno")

    monkeypatch.setattr(metrics, "escribir_csv", falla)
    monkeypatch.setattr(metrics, "reporte_terminal", lambda *a, **k: None)
    metrics.registrar(_fila(), tmp_path / "m.csv")  # no debe lanzar


def test_calentar_medido_con_pipeline_falso(sin_gpu):
    pipe = PipelineFalso()
    m = calentar_medido(pipe)
    assert pipe.calentado == 1
    assert m.calentamiento is True
    assert m.ms_total is None and m.ms_preproceso is None and m.imagenes_por_s is None
    assert (m.ms_deteccion, m.ms_clasificacion, m.ms_respaldo) == (12.0, 4.0, 2.0)
    assert m.detecciones == 0 == m.reconocidas == m.aproximadas
    assert (m.ancho, m.alto) == (640, 480)


def test_reporte_terminal_rotula_y_avisa():
    salida = io.StringIO()
    reporte_terminal(_fila(), consola=Console(file=salida, width=120))
    texto = salida.getvalue()
    assert "equivalente" in texto and "dispositivo" in texto.lower()
    assert "VRAM usada pico (dispositivo)" in texto
    assert "no son fiables" in texto  # muestras_gpu = 0 < MIN_MUESTRAS_FIABLES


def test_reporte_terminal_sin_aviso_con_muestras_suficientes():
    salida = io.StringIO()
    reporte_terminal(_fila(muestras_gpu=5), consola=Console(file=salida, width=120))
    assert "no son fiables" not in salida.getvalue()


def test_muestreador_max_muestras_acota_la_memoria():
    from roadvision.metrics import MuestraSistema, MuestreadorSistema

    m = MuestreadorSistema(0.01, max_muestras=3)
    for i in range(10):
        m._muestras.append(MuestraSistema(t_s=float(i), cpu_pct=0.0, ram_proceso_mb=1.0))
    assert [x.t_s for x in m.muestras] == [7.0, 8.0, 9.0]


def test_ruta_csv_es_absoluta_y_apunta_a_logs():
    from roadvision.metrics import RUTA_CSV

    assert RUTA_CSV.is_absolute()
    assert RUTA_CSV.parts[-2:] == ("logs", "metricas.csv")
