from PIL import Image

from roadvision.imaging import normalizar, recortar
from tests.fakes import jpeg_con_orientacion


def test_normalizar_aplica_exif():
    # JPEG 40x20 guardado con EXIF Orientation=6 (rotar 90°)
    assert normalizar(jpeg_con_orientacion(40, 20, 6)).size == (20, 40)


def test_normalizar_transparencia_sobre_blanco():
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
    out = normalizar(img)
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255)


def test_normalizar_grises_y_cmyk():
    assert normalizar(Image.new("L", (8, 8))).mode == "RGB"
    assert normalizar(Image.new("CMYK", (8, 8))).mode == "RGB"


def test_recortar_con_margen():
    img = Image.new("RGB", (200, 100))
    assert recortar(img, (50, 20, 100, 70), 0.10).size == (60, 60)   # (45,15)-(105,75)


def test_recortar_en_borde():
    img = Image.new("RGB", (200, 100))
    assert recortar(img, (0, 0, 30, 30), 0.10).size == (33, 33)     # (0,0)-(33,33)
