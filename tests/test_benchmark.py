"""Pruebas de `scripts/benchmark.py` (sin torch ni GPU)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from roadvision import metrics
from roadvision.contracts import Estado
from roadvision.metrics import MetricasAnalisis, leer_csv
from tests.fakes_metrics import PipelineFalso

_RUTA = Path(__file__).resolve().parent.parent / "scripts" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("benchmark_script", _RUTA)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def _fila(ms_total=10.0, **extra) -> MetricasAnalisis:
    base = dict(
        marca_tiempo="t", dispositivo="CPU", precision="fp32", calentamiento=False,
        ancho=640, alto=480, detecciones=0, reconocidas=0, aproximadas=0, solo_categoria=0,
        no_reconocidas=0, ms_total=ms_total, ms_preproceso=1.0, ms_deteccion=5.0,
        ms_clasificacion=2.0, ms_respaldo=1.0)
    base.update(extra)
    return MetricasAnalisis(**base)


def test_modulo_importable_sin_torch():
    """benchmark y metrics se importan sin cargar torch (proceso limpio: el conftest del
    compañero ya importa torch en el proceso de pytest)."""
    import subprocess
    import sys

    codigo = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(_RUTA.parent.parent)!r})\n"
        f"spec = importlib.util.spec_from_file_location('b', {str(_RUTA)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "import roadvision.metrics\n"
        "assert 'torch' not in sys.modules, 'se importó torch'\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sinteticas_deterministas():
    a, b = bm.imagenes_sinteticas(), bm.imagenes_sinteticas()
    assert len(a) == 3 and a[0].size == (1280, 720)
    assert a[0].tobytes() == b[0].tobytes() and a[0].tobytes() != a[1].tobytes()


def test_cargar_imagenes_ordenadas_y_filtra(tmp_path):
    for nombre in ("b.png", "a.jpg", "c.txt"):
        if nombre.endswith("txt"):
            (tmp_path / nombre).write_text("x")
        else:
            Image.new("RGB", (8, 8)).save(tmp_path / nombre)
    imagenes, sinteticas = bm.cargar_imagenes(tmp_path)
    assert len(imagenes) == 2 and not sinteticas


@pytest.mark.parametrize("carpeta", ["no_existe", "vacia"])
def test_cargar_imagenes_cae_a_sinteticas(tmp_path, carpeta):
    ruta = tmp_path / carpeta
    if carpeta == "vacia":
        ruta.mkdir()
    imagenes, sinteticas = bm.cargar_imagenes(ruta)
    assert sinteticas and len(imagenes) == 3


def test_resumir_p95_solo_con_20_muestras():
    pocas = bm.resumir([_fila(float(i)) for i in range(1, 11)])
    assert pocas["etapas"]["total"]["p95"] is None
    muchas = bm.resumir([_fila(float(i)) for i in range(1, 101)])
    t = muchas["etapas"]["total"]
    assert t["n"] == 100 and t["media"] == pytest.approx(50.5)
    assert t["p50"] == pytest.approx(50.5) and t["p95"] == pytest.approx(95.05)
    assert (t["min"], t["max"]) == (1.0, 100.0)
    assert muchas["imagenes_por_s_equivalente"] == pytest.approx(1000 / 50.5)


def test_resumir_excluye_calentamiento_y_maneja_none():
    filas = [_fila(100.0, calentamiento=True), _fila(10.0, gpu_pct_pico=40.0,
             vram_usada_mb=500.0, potencia_w_prom=30.0), _fila(20.0, gpu_pct_pico=70.0,
             vram_usada_mb=300.0, potencia_w_prom=50.0)]
    r = bm.resumir(filas)
    assert r["n"] == 2 and r["etapas"]["total"]["max"] == 20.0
    assert r["gpu_pct_pico"] == 70.0 and r["vram_usada_mb_pico"] == 500.0
    assert r["potencia_w_prom"] == pytest.approx(40.0)
    assert r["temperatura_c_max"] is None
    vacio = bm.resumir([])
    assert vacio["n"] == 0 and vacio["etapas"]["total"] is None
    assert vacio["imagenes_por_s_equivalente"] is None


def test_correr_con_pipeline_falso_y_csv(tmp_path):
    pipe = PipelineFalso([Estado.RECONOCIDA] if hasattr(Estado, "RECONOCIDA") else [])
    imagenes = [Image.new("RGB", (32, 32))] * 2
    csv_ruta = tmp_path / "m.csv"
    filas = bm.correr(pipe, imagenes, 3, ruta_csv=csv_ruta)
    assert len(filas) == 6 and pipe.analizados == 6 and pipe.calentado == 0
    df = leer_csv(csv_ruta)
    assert len(df) == 6 and not df["calentamiento"].any()


def test_correr_sin_csv_no_crea_archivo(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(metrics, "escribir_csv", lambda *a, **k: llamadas.append(a))
    bm.correr(PipelineFalso(), [Image.new("RGB", (8, 8))], 1)
    assert not llamadas and not list(tmp_path.iterdir())


def test_resumir_rechaza_configuraciones_mezcladas():
    with pytest.raises(ValueError, match="mezcladas"):
        bm.resumir([_fila(precision="fp32"), _fila(precision="fp16")])
    # el calentamiento no cuenta para decidir
    r = bm.resumir([_fila(precision="fp16", calentamiento=True), _fila(precision="fp32")])
    assert r["precision"] == "fp32"


def test_mostrar_resumen_rotula_honestamente(capsys):
    r = bm.resumir([_fila(10.0)])
    bm.mostrar_resumen(r)
    salida = capsys.readouterr().out
    assert "equivalente" in salida and "dispositivo" in salida and "n<20" in salida
    assert "100 % = 1" in salida and "incluye 0.0 ms" in salida


def test_mostrar_resumen_cpu_no_muestra_lecturas_de_gpu(capsys):
    r = bm.resumir([_fila(10.0, gpu_pct_pico=55.0, vram_usada_mb=777.0,
                          temperatura_c_max=61.0, potencia_w_prom=33.0)])
    bm.mostrar_resumen(r)
    salida = capsys.readouterr().out
    assert salida.count("corrida en CPU") == 4
    assert "777" not in salida and "61" not in salida


def _construir_falso(config):
    pipe = PipelineFalso()
    pipe.config = config
    return pipe


def test_main_excluye_calentamiento_y_usa_sinteticas(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(metrics, "reporte_arranque", lambda *a, **k: None)
    ruta = tmp_path / "m.csv"
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    codigo = bm.main(
        ["--carpeta", str(vacia), "--repeticiones", "2", "--csv", str(ruta)],
        construir=_construir_falso)
    salida = capsys.readouterr().out
    assert codigo == 0 and "SINT" in salida
    todo = leer_csv(ruta, incluir_calentamiento=True)
    assert todo["calentamiento"].sum() == 1 and len(todo) == 7
    assert len(leer_csv(ruta)) == 6


def test_main_carpeta_inexistente_da_2(tmp_path, capsys):
    assert bm.main(["--carpeta", str(tmp_path / "no_existe"), "--sin-csv"],
                   construir=_construir_falso) == 2
    assert "no existe" in capsys.readouterr().err


def test_main_sinteticas_no_escriben_csv_del_repo(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(metrics, "reporte_arranque", lambda *a, **k: None)
    escritas = []
    monkeypatch.setattr(metrics, "escribir_csv", lambda *a, **k: escritas.append(a))
    assert bm.main(["--carpeta", str(tmp_path), "--repeticiones", "1"],
                   construir=_construir_falso) == 0
    salida = capsys.readouterr().out
    assert not escritas
    assert "no se escribe el CSV" in salida and "no representativas" in salida


def test_main_sin_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "reporte_arranque", lambda *a, **k: None)
    assert bm.main(["--carpeta", str(tmp_path), "--repeticiones", "1", "--sin-csv"],
                   construir=_construir_falso) == 0


def test_main_cuda_no_disponible_da_2(tmp_path, capsys):
    def sin_cuda(config):
        raise RuntimeError("Se pidió CUDA pero no hay una GPU NVIDIA con CUDA disponible")

    assert bm.main(["--dispositivo", "cuda", "--sin-csv", "--carpeta", str(tmp_path)],
                   construir=sin_cuda) == 2
    assert "CUDA" in capsys.readouterr().err


def test_main_import_error_da_2(tmp_path, capsys):
    def sin_torch(config):
        raise ImportError("No module named 'torch'")

    assert bm.main(["--sin-csv", "--carpeta", str(tmp_path)], construir=sin_torch) == 2
    assert "PyTorch" in capsys.readouterr().err


def test_comparar_precision_con_oom_en_fp32(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(metrics, "reporte_arranque", lambda *a, **k: None)

    def construir(config):
        if not config.fp16:
            raise RuntimeError("CUDA out of memory")
        pipe = PipelineFalso()
        pipe.dispositivo = SimpleNamespace(nombre="cuda", dtype="torch.float16",
                                           descripcion="GPU falsa")
        return pipe

    codigo = bm.main(["--comparar-precision", "--sin-csv", "--repeticiones", "1",
                      "--pausa-s", "0", "--carpeta", str(tmp_path)],
                     construir=construir)
    salida = capsys.readouterr().out
    assert codigo == 0 and "Comparación" in salida and "out of memory" in salida


def test_comparar_precision_en_cpu_se_rechaza(capsys):
    assert bm.main(["--comparar-precision", "--dispositivo", "cpu"]) == 2
    assert "CUDA" in capsys.readouterr().err


def test_sin_fp16_y_comparar_son_excluyentes(tmp_path):
    with pytest.raises(SystemExit) as exc:
        bm.main(["--sin-fp16", "--comparar-precision", "--carpeta", str(tmp_path)])
    assert exc.value.code == 2
