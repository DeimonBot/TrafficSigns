from PIL import Image

from roadvision.cli import main
from roadvision.contracts import Deteccion, Prediccion
from tests.fakes import pipeline_falso


def test_cli_imprime_senales(tmp_path, capsys):
    ruta = tmp_path / "foto.jpg"
    Image.new("RGB", (640, 480)).save(ruta)
    falso = pipeline_falso([Deteccion((100, 100, 160, 160), 0.9)], [Prediccion("Stop", 0.97)])
    assert main([str(ruta)], construir=lambda c: falso) == 0
    out = capsys.readouterr().out
    assert "#1 PARE [reglamentaria] reconocida conf 0.97" in out
    assert "Resumen:" in out and "total=" in out


def test_cli_archivo_inexistente(capsys):
    assert main(["no_existe.jpg"], construir=lambda c: None) == 2
    assert "No existe el archivo" in capsys.readouterr().err


def test_cli_archivo_no_es_imagen(tmp_path, capsys):
    ruta = tmp_path / "texto.jpg"
    ruta.write_text("no soy una imagen")
    assert main([str(ruta)], construir=lambda c: None) == 2
    assert "No se pudo abrir la imagen" in capsys.readouterr().err


def test_cli_pasa_dispositivo(tmp_path):
    ruta = tmp_path / "foto.jpg"
    Image.new("RGB", (64, 64)).save(ruta)
    recibido = {}

    def construir(c):
        recibido["c"] = c
        return pipeline_falso([], [])

    main([str(ruta), "--dispositivo", "cpu", "--sin-fp16"], construir=construir)
    assert (recibido["c"].dispositivo, recibido["c"].fp16) == ("cpu", False)
