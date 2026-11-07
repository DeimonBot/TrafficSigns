# RoadVision — Plan de implementación de los modelos (H0 mínimo + H1 + H2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el núcleo `roadvision/` que recibe una foto y devuelve un `Resultado` con cada señal detectada, clasificada y explicada en español, usando OWLv2 + ViT GTSRB + respaldo CLIP, sin entrenar nada.

**Architecture:** Paquete Python sin dependencias de interfaz. Cada modelo vive detrás de un protocolo (`Detector`, `Clasificador`, `Respaldo`) definido en `contracts.py`; la lógica de decisión (`decision.py`) es pura y se prueba con un respaldo falso; `pipeline.py` orquesta y mide tiempos por etapa. Las pruebas unitarias no descargan modelos; las que sí lo hacen llevan la marca `modelos`.

**Tech Stack:** Python ≥ 3.11, PyTorch + torchvision (instalados aparte), transformers 5.22.2, Pillow, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-roadvision-design.md` (secciones 3, 5.1–5.4, 5.7, 5.8, 6.1–6.2 y 7).

**Alcance:** este plan es la parte de modelos (responsable: DidierParody). Quedan para otros planes: interfaz Streamlit (H3), métricas de GPU con `nvidia-ml-py` y `benchmark.py` (H4), evaluación niveles B y C (H5). Este plan deja listo lo que esos planes consumen: `contracts.py`, `Pipeline.analizar`, `Pipeline.calentar`, `Pipeline.carga` y `Resultado.tiempos_ms`. El dibujo de cajas que el diseño ubica en `imaging.py` lo agrega el plan de interfaz (H3).

**Rama:** `feat/modelos`. Publicar la Tarea 1 apenas esté: el resto del equipo depende de `contracts.py`.

## Global Constraints

- Python ≥ 3.11 (torch 2.15 lo exige); probado en 3.12.
- `torch` y `torchvision` **no** van en `requirements.txt`: se instalan desde pytorch.org (variante CUDA en el PC NVIDIA, CPU en los demás).
- Todo corre en CPU; CUDA si está disponible; fp16 **solo** en CUDA.
- Modelos: detector `google/owlv2-base-patch16-ensemble`, clasificador `kelvinandreas/vit-traffic-sign-GTSRB`, respaldo `openai/clip-vit-base-patch32`.
- Consultas del detector: `"a traffic sign"`, `"a road sign"`.
- Umbrales iniciales: detección 0.15, GTSRB 0.80, CLIP señal 0.50, CLIP categoría 0.50; margen de recorte 10 %; tamaño mínimo de caja 20 px; IoU para duplicados 0.5.
- Textos que ve el usuario: en español. Textos para CLIP y OWLv2: en inglés.
- `roadvision/` no importa `streamlit` ni nada de interfaz.
- No usar Ultralytics.
- Nunca subir pesos ni datos: `.gitignore` incluye `*.pt`, `logs/`, `datasets/` (además de lo que ya tiene).
- Pruebas unitarias: sin red y sin modelos. Pruebas con modelos reales: marca `@pytest.mark.modelos`, excluidas por defecto, se corren con `pytest -m modelos`.
- Ninguna señal sale como "reconocida" sin pasar por las reglas de la sección 5.3 del diseño (es la anti-meta principal).

## Review Focus

1. **Foto sin ninguna señal** → `Resultado` con listas y conteos vacíos, sin llamar a CLIP, sin error. Prueba: Tarea 9, `test_sin_detecciones`.
2. **Foto de celular con rotación EXIF** → las cajas se refieren a la imagen ya rotada y `Resultado.ancho/alto` reflejan la rotación. Pruebas: Tarea 2 `test_normalizar_aplica_exif`, Tarea 9 `test_exif_rotado`.
3. **PNG con transparencia, imagen en escala de grises o JPEG CMYK** → se normaliza a RGB; lo transparente queda blanco. Pruebas: Tarea 2 `test_normalizar_transparencia_sobre_blanco`, `test_normalizar_grises_y_cmyk`.
4. **Señal pegada al borde o diminuta** → el recorte con margen no se sale de la imagen; cajas con lado < 20 px se descartan. Pruebas: Tarea 2 `test_recortar_en_borde`, `test_filtrar_umbral_tamano_y_duplicados`.
5. **La misma señal detectada por las dos consultas** → una sola señal. Pruebas: Tarea 2 `test_filtrar_umbral_tamano_y_duplicados`, Tarea 9 `test_duplicados_y_pequenas`.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `requirements.txt`, `pytest.ini`, `.gitignore` | Dependencias, marca `modelos`, exclusiones |
| `roadvision/__init__.py` | Paquete |
| `roadvision/contracts.py` | Tipos compartidos y protocolos (lo que consumen UI y métricas) |
| `roadvision/config.py` | `Config`, `Umbrales`, `RAIZ` |
| `roadvision/cajas.py` | `iou`, `filtrar_cajas` |
| `roadvision/imaging.py` | `normalizar`, `recortar` |
| `roadvision/device.py` | `Dispositivo`, `elegir_dispositivo`, `sincronizar` |
| `roadvision/timing.py` | `Cronometro` por etapa |
| `roadvision/catalog.py` + `data/catalogo_senales.yaml` | Catálogo de señales y su validación |
| `roadvision/decision.py` | `Decisor`: reglas de la sección 5.3 |
| `roadvision/classifier.py` | `ClasificadorGtsrb` (ViT) |
| `roadvision/detector.py` | `DetectorOwlv2` |
| `roadvision/fallback.py` | `RespaldoClip` |
| `roadvision/pipeline.py` | `Pipeline`, `construir_pipeline` |
| `roadvision/cli.py` | `python -m roadvision.cli foto.jpg` |
| `roadvision/evaluacion.py` + `scripts/evaluate.py` | Evaluación nivel A (H5 agrega B y C) |
| `scripts/download_models.py`, `scripts/check_gpu.py` | Descarga y verificación del entorno |
| `tests/fakes.py` | Detector, clasificador y respaldo falsos; `pipeline_falso` |
| `tests/conftest.py` | Fixtures de modelos reales y muestras GTSRB |
| `tests/fixtures/gtsrb_etiquetas.json`, `tests/fixtures/catalogo_prueba.yaml` | Datos fijos de prueba |

---

### Task 1: Esqueleto, contratos y configuración

**Files:**
- Create: `requirements.txt` (reemplaza el vacío), `pytest.ini`, `roadvision/__init__.py`, `roadvision/contracts.py`, `roadvision/config.py`, `tests/test_config.py`
- Modify: `.gitignore` (agregar `*.pt`, `logs/`, `datasets/`)

**Interfaces:**
- Produces (`roadvision/contracts.py`):
  - `Caja = tuple[int, int, int, int]` — `(x1, y1, x2, y2)` en píxeles de la imagen normalizada.
  - `class Categoria(str, Enum)`: `REGLAMENTARIA="reglamentaria"`, `PREVENTIVA="preventiva"`, `INFORMATIVA="informativa"`.
  - `class Fuente(str, Enum)`: `GTSRB="gtsrb"`, `CLIP="clip"`, `NINGUNA="ninguna"`.
  - `class Estado(str, Enum)`: `RECONOCIDA="reconocida"`, `APROXIMADA="aproximada"`, `SOLO_CATEGORIA="solo_categoria"`, `NO_RECONOCIDA="no_reconocida"`.
  - `@dataclass(frozen=True) Deteccion(caja: Caja, puntaje: float)`
  - `@dataclass(frozen=True) Prediccion(etiqueta: str, probabilidad: float)`
  - `@dataclass Senal(numero: int, caja: Caja, recorte: Image.Image, estado: Estado, fuente: Fuente, id_catalogo: str | None, nombre: str, categoria: Categoria | None, significado: str | None, confianza: float, puntaje_deteccion: float)`
  - `@dataclass Resultado(senales: list[Senal], conteo_por_nombre: dict[str, int], conteo_por_categoria: dict[str, int], tiempos_ms: dict[str, float], ancho: int, alto: int, dispositivo: str)`
  - `class Detector(Protocol)`: `detectar(self, imagen: Image.Image, umbral: float) -> list[Deteccion]`
  - `class Clasificador(Protocol)`: atributo `etiquetas: list[str]` (ordenadas por id de clase); `clasificar(self, recortes: list[Image.Image]) -> list[Prediccion]`
  - `class Respaldo(Protocol)`: `puntuar(self, recortes: list[Image.Image], textos: list[str]) -> list[list[float]]` — una fila por recorte, probabilidades (softmax) sobre `textos`.
- Produces (`roadvision/config.py`): `RAIZ: Path` (raíz del repo), `@dataclass(frozen=True) Umbrales(deteccion=0.15, gtsrb=0.80, clip_senal=0.50, clip_categoria=0.50)`, `@dataclass(frozen=True) Config(...)` con los campos de la prueba.

- [ ] **Step 1: Crear rama y archivos de proyecto**

```bash
git checkout -b feat/modelos
```
`requirements.txt`: comentario indicando que torch/torchvision se instalan aparte (README) y `transformers==5.22.2`, `pillow==13.0.0`, `pyyaml==6.0.3`, `pytest==9.2.0`. `pytest.ini`: `testpaths = tests`, `markers = modelos: usa modelos reales descargados de Hugging Face`, `addopts = -m "not modelos"`.

- [ ] **Step 2: Instalar entorno local (CPU en este PC)**

Run: `python -m venv .venv`, activar (`.venv\Scripts\activate` en Windows), `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, luego `pip install -r requirements.txt`.
Expected: `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"` imprime `2.x … 5.22.2`.

- [ ] **Step 3: Escribir la prueba que falla**

```python
# tests/test_config.py
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
```

- [ ] **Step 4: Verificar que falla** — Run: `pytest tests/test_config.py -v` — Expected: FAIL (`ModuleNotFoundError: roadvision`).

- [ ] **Step 5: Implementar `contracts.py` y `config.py`** según Interfaces. `RAIZ = Path(__file__).resolve().parent.parent`; `ruta_catalogo = RAIZ / "data" / "catalogo_senales.yaml"`.

- [ ] **Step 6: Verificar que pasa** — Run: `pytest -v` — Expected: 2 passed.

- [ ] **Step 7: Commit y publicar**

```bash
git add requirements.txt pytest.ini .gitignore roadvision tests/test_config.py docs/superpowers/plans/2026-11-06-roadvision-modelos.md
git commit -m "feat: contratos y configuración del núcleo de modelos"
git push -u origin feat/modelos
```

---

### Task 2: Cajas e imágenes

**Files:**
- Create: `roadvision/cajas.py`, `roadvision/imaging.py`, `tests/test_cajas.py`, `tests/test_imaging.py`

**Interfaces:**
- Consumes: `Caja`, `Deteccion` (Tarea 1).
- Produces:
  - `iou(a: Caja, b: Caja) -> float`
  - `filtrar_cajas(detecciones: list[Deteccion], umbral: float, tamano_minimo: int, iou_max: float) -> list[Deteccion]` — quita puntaje < umbral, quita cajas con `min(ancho, alto) < tamano_minimo`, y aplica supresión de no-máximos voraz por puntaje (descarta la de menor puntaje si IoU > iou_max). Devuelve en orden de puntaje descendente.
  - `normalizar(imagen: Image.Image) -> Image.Image` — aplica `ImageOps.exif_transpose`; si tiene transparencia, la compone sobre blanco; devuelve modo `"RGB"`.
  - `recortar(imagen: Image.Image, caja: Caja, margen: float) -> Image.Image` — amplía cada lado en `margen × ancho` (horizontal) y `margen × alto` (vertical), redondeando al entero más cercano, y recorta a los límites de la imagen.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
# tests/test_cajas.py
def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

def test_filtrar_umbral_tamano_y_duplicados():
    dets = [Deteccion((100, 100, 150, 150), 0.9),
            Deteccion((102, 101, 152, 149), 0.6),   # misma señal, otra consulta
            Deteccion((300, 300, 310, 330), 0.9),   # 10 px de ancho
            Deteccion((400, 100, 460, 160), 0.1)]   # bajo el umbral
    assert filtrar_cajas(dets, umbral=0.15, tamano_minimo=20, iou_max=0.5) == [
        Deteccion((100, 100, 150, 150), 0.9)]

# tests/test_imaging.py
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
```
`jpeg_con_orientacion(ancho, alto, orientacion)` es un helper del archivo de prueba: crea la imagen, `exif = Image.Exif(); exif[0x0112] = orientacion`, la guarda como JPEG en memoria y la reabre.

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_cajas.py tests/test_imaging.py -v` — Expected: FAIL (funciones no definidas).
- [ ] **Step 3: Implementar `cajas.py` e `imaging.py`** según Interfaces.
- [ ] **Step 4: Verificar que pasan** — Run: `pytest -v` — Expected: todo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: utilidades de cajas e imágenes"` (agregando los 4 archivos).

---

### Task 3: Dispositivo, cronómetro y verificación de GPU

**Files:**
- Create: `roadvision/device.py`, `roadvision/timing.py`, `scripts/check_gpu.py`, `tests/test_device.py`, `tests/test_timing.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Dispositivo(nombre: str, dtype: torch.dtype, descripcion: str)` — `nombre` es `"cuda"` o `"cpu"`; `descripcion` es el nombre de la GPU o `"CPU"`.
  - `elegir_dispositivo(preferido: str = "auto", fp16: bool = True) -> Dispositivo` — `preferido` ∈ {`"auto"`, `"cuda"`, `"cpu"`}, otro valor → `ValueError`. `"cuda"` sin GPU → `RuntimeError` con un mensaje en español que contiene "CUDA". fp16 solo si el resultado es CUDA.
  - `sincronizar(dispositivo: Dispositivo) -> None` — `torch.cuda.synchronize()` en CUDA; nada en CPU.
  - `class Cronometro`: `__init__(self, sincronizar: Callable[[], None] | None = None)`; `etapa(self, nombre: str)` context manager que llama a `sincronizar` antes y después, mide con `time.perf_counter()` y **suma** a `tiempos_ms[nombre]`; atributo `tiempos_ms: dict[str, float]`.
  - `scripts/check_gpu.py`: imprime versión de torch, `torch.version.cuda`, si hay CUDA, nombre de GPU, VRAM total (GB) y el resultado de una multiplicación de matrices en fp16 en GPU. Código de salida 0 si CUDA funciona; si no, imprime "No se detectó una GPU NVIDIA con CUDA" y sale con 1. (La parte de `nvidia-ml-py` la agrega el plan de métricas.)

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
# tests/test_device.py  (usa monkeypatch sobre torch.cuda)
def test_auto_sin_cuda_usa_cpu_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = elegir_dispositivo("auto", fp16=True)
    assert (d.nombre, d.dtype, d.descripcion) == ("cpu", torch.float32, "CPU")

def test_auto_con_cuda_usa_fp16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "GPU de prueba")
    d = elegir_dispositivo("auto", fp16=True)
    assert (d.nombre, d.dtype, d.descripcion) == ("cuda", torch.float16, "GPU de prueba")
    assert elegir_dispositivo("auto", fp16=False).dtype == torch.float32

def test_cuda_pedido_sin_gpu_falla(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        elegir_dispositivo("cuda")

def test_valor_invalido():
    with pytest.raises(ValueError):
        elegir_dispositivo("tpu")

# tests/test_timing.py
def test_cronometro_suma_y_sincroniza():
    llamadas = []
    c = Cronometro(sincronizar=lambda: llamadas.append(1))
    with c.etapa("a"):
        time.sleep(0.01)
    with c.etapa("a"):
        pass
    assert c.tiempos_ms["a"] >= 10
    assert len(llamadas) == 4

def test_cronometro_sin_sincronizar():
    c = Cronometro()
    with c.etapa("b"):
        pass
    assert c.tiempos_ms["b"] >= 0
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_device.py tests/test_timing.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `device.py`, `timing.py` y `scripts/check_gpu.py`.**
- [ ] **Step 4: Verificar** — Run: `pytest -v` — Expected: todo PASS. Run: `python scripts/check_gpu.py` — Expected en este PC (sin NVIDIA): mensaje "No se detectó una GPU NVIDIA con CUDA" y código de salida 1.
- [ ] **Step 5: Commit** — `git commit -m "feat: selección de dispositivo, cronómetro y check_gpu"`.

---

### Task 4: Catálogo de señales

**Files:**
- Create: `roadvision/catalog.py`, `data/catalogo_senales.yaml`, `tests/fixtures/gtsrb_etiquetas.json`, `tests/fixtures/catalogo_prueba.yaml`, `tests/test_catalog.py`

**Interfaces:**
- Consumes: `Categoria` (Tarea 1), `Config().ruta_catalogo`.
- Produces:
  - `@dataclass(frozen=True) EntradaCatalogo(id: str, nombre: str, categoria: Categoria, significado: str, gtsrb: tuple[str, ...], clip: str | None, velocidad_kmh: int | None)`
  - `@dataclass(frozen=True) InfoCategoria(nombre: str, significado: str, clip: str)`
  - `@dataclass(frozen=True) Catalogo(entradas: dict[str, EntradaCatalogo], categorias: dict[Categoria, InfoCategoria], clip_otro: str, nombre_no_reconocida: str, gtsrb_sin_equivalente: frozenset[str])` con métodos:
    - `por_etiqueta_gtsrb(self, etiqueta: str) -> EntradaCatalogo | None`
    - `velocidades(self) -> list[EntradaCatalogo]` — entradas con `velocidad_kmh` y `clip`.
    - `con_clip(self) -> list[EntradaCatalogo]` — entradas con `clip`.
  - `cargar_catalogo(ruta: Path) -> Catalogo` — lanza `ErrorCatalogo(ValueError)` si: hay ids repetidos; una categoría no es válida; una etiqueta GTSRB está en dos entradas; una etiqueta está mapeada y también en `gtsrb_sin_equivalente`; falta alguna de las 3 categorías; falta la entrada `velocidad_no_verificada` o no es reglamentaria.
  - `etiquetas_sin_cubrir(catalogo: Catalogo, etiquetas: Iterable[str]) -> list[str]` — etiquetas ni mapeadas ni en `gtsrb_sin_equivalente`.

Formato del YAML (claves de primer nivel): `no_reconocida` (texto), `clip_otro` (texto en inglés), `categorias` (mapa `reglamentaria|preventiva|informativa` → `nombre`, `significado`, `clip`), `gtsrb_sin_equivalente` (lista), `senales` (lista de entradas con los campos de `EntradaCatalogo`; `gtsrb`, `clip` y `velocidad_kmh` opcionales).

Contenido obligatorio de `data/catalogo_senales.yaml` (lo demás lo completa el implementador con el Manual de Señalización Vial 2015, sección 5.4 del diseño):
- `no_reconocida: "Señal no reconocida"`; `clip_otro: "a photo of something that is not a traffic sign"`.
- `categorias.*.nombre`: "Señal reglamentaria (tipo no identificado)", "Señal preventiva (tipo no identificado)", "Señal informativa (tipo no identificado)"; `clip` según la sección 5.4 del diseño.
- `pare` ← `Stop`; `ceda_el_paso` ← `Yield`.
- `velocidad_20/30/50/60/70/80/100/120` ← `Speed limit (N km/h)` correspondiente, con `velocidad_kmh` y `clip: "a circular speed limit road sign showing the number N"`.
- `velocidad_40` y `velocidad_90` sin `gtsrb`, con `velocidad_kmh` y el mismo patrón de `clip`.
- `velocidad_no_verificada`: nombre "Velocidad máxima — valor no verificado", reglamentaria, sin `gtsrb`, sin `clip`, sin `velocidad_kmh`.
- `gtsrb_sin_equivalente` incluye al menos: `Priority road`, `End of speed limit (80km/h)`, `End speed + passing limits`, `End of no passing`, `End no passing veh > 3.5 tons`.

`tests/fixtures/gtsrb_etiquetas.json` (copiado del `config.json` del modelo el 2026-11-06, en orden de id):

```json
["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", "No passing veh over 3.5 tons", "Right-of-way at intersection", "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5 tons prohibited", "No entry", "General caution", "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", "End speed + passing limits", "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", "End of no passing", "End no passing veh > 3.5 tons"]
```

`tests/fixtures/catalogo_prueba.yaml` (catálogo mínimo para Tareas 5, 9 y 10, con textos CLIP cortos para que las pruebas sean legibles): `pare` ← `Stop` (reglamentaria, sin clip); `velocidad_30` ← `Speed limit (30km/h)`, clip `v30`, 30; `velocidad_50` ← `Speed limit (50km/h)`, clip `v50`, 50; `velocidad_60` ← `Speed limit (60km/h)`, clip `v60`, 60; `velocidad_40` sin gtsrb, clip `v40`, 40; `velocidad_no_verificada`; `resalto` (preventiva) ← `Bumpy road`, clip `resalto`; categorías con clip `cat-reg`, `cat-prev`, `cat-inf`; `clip_otro: otro`; `gtsrb_sin_equivalente: [Priority road]`.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
# tests/test_catalog.py
def test_catalogo_real_cubre_las_43_etiquetas():
    cat = cargar_catalogo(Config().ruta_catalogo)
    etiquetas = json.loads((FIXTURES / "gtsrb_etiquetas.json").read_text("utf-8"))
    assert len(etiquetas) == 43
    assert etiquetas_sin_cubrir(cat, etiquetas) == []

def test_catalogo_real_entradas_minimas():
    cat = cargar_catalogo(Config().ruta_catalogo)
    assert cat.por_etiqueta_gtsrb("Stop").id == "pare"
    assert cat.por_etiqueta_gtsrb("Yield").id == "ceda_el_paso"
    assert {e.velocidad_kmh for e in cat.velocidades()} >= {20, 30, 40, 50, 60, 70, 80, 90, 100, 120}
    assert cat.entradas["velocidad_40"].gtsrb == () and cat.entradas["velocidad_90"].gtsrb == ()
    assert {"Priority road", "End of speed limit (80km/h)", "End speed + passing limits",
            "End of no passing", "End no passing veh > 3.5 tons"} <= cat.gtsrb_sin_equivalente
    assert all(e.significado.strip() for e in cat.entradas.values())

def test_rechaza_etiqueta_en_dos_entradas(tmp_path): ...          # ErrorCatalogo
def test_rechaza_categoria_invalida(tmp_path): ...                # ErrorCatalogo
def test_rechaza_mapeada_y_sin_equivalente(tmp_path): ...         # ErrorCatalogo
def test_requiere_velocidad_no_verificada(tmp_path): ...          # ErrorCatalogo
def test_catalogo_de_prueba_carga():
    cat = cargar_catalogo(FIXTURES / "catalogo_prueba.yaml")
    assert [e.velocidad_kmh for e in cat.velocidades()] == [30, 50, 60, 40]
```
Cada prueba de rechazo escribe en `tmp_path` una copia de `catalogo_prueba.yaml` con un solo defecto y verifica `pytest.raises(ErrorCatalogo)`.

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_catalog.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `catalog.py` y escribir ambos YAML.** `velocidades()` y `con_clip()` respetan el orden del archivo.
- [ ] **Step 4: Verificar** — Run: `pytest -v` — Expected: todo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: catálogo de señales colombianas con equivalencias GTSRB"`.

---

### Task 5: Decisor (reglas de la sección 5.3)

**Files:**
- Create: `roadvision/decision.py`, `tests/fakes.py`, `tests/test_decision.py`

**Interfaces:**
- Consumes: `Prediccion`, `Estado`, `Fuente`, `Categoria`, `Respaldo` (Tarea 1); `Umbrales` (Tarea 1); `Catalogo` (Tarea 4).
- Produces:
  - `@dataclass(frozen=True) Decision(estado: Estado, fuente: Fuente, id_catalogo: str | None, categoria: Categoria | None, confianza: float)`
  - `class Decisor`: `__init__(self, catalogo: Catalogo, respaldo: Respaldo)`; `decidir(self, recortes: list[Image.Image], predicciones: list[Prediccion], umbrales: Umbrales) -> list[Decision]` (misma longitud y orden que la entrada).
- Produces (`tests/fakes.py`, reutilizado en Tareas 9 y 10): `FakeRespaldo(puntajes: dict[int, dict[str, float]] | None = None, por_defecto: dict[str, float] | None = None)` — identifica cada recorte por su ancho en píxeles; el puntaje de un texto es `puntajes[ancho][texto]`, si no `por_defecto[texto]`, si no `0.01`; cada fila se normaliza para sumar 1; guarda `llamadas: list[tuple[int, int]]` = (número de recortes, número de textos). Helper `recorte(n) -> Image.Image` de tamaño `(n, n)`.

Reglas (por recorte, en este orden):
1. Si la etiqueta está en `gtsrb_sin_equivalente`, o `probabilidad < umbrales.gtsrb`, o no tiene entrada → **respaldo** (regla 3).
2. Si la entrada tiene `velocidad_kmh` → **verificación de velocidad**: CLIP sobre los textos de `catalogo.velocidades()`; sea `mejor` la entrada de mayor probabilidad `p`:
   - `mejor.velocidad_kmh == entrada.velocidad_kmh` → `RECONOCIDA, GTSRB, entrada.id, confianza=probabilidad_gtsrb`.
   - `mejor.gtsrb == ()` y `p >= umbrales.clip_senal` → `APROXIMADA, CLIP, mejor.id, confianza=p`.
   - si no → `SOLO_CATEGORIA, GTSRB, "velocidad_no_verificada", REGLAMENTARIA, confianza=probabilidad_gtsrb`.
   Si no tiene `velocidad_kmh` → `RECONOCIDA, GTSRB, entrada.id, entrada.categoria, confianza=probabilidad_gtsrb`.
3. **Respaldo**: CLIP sobre los textos de `catalogo.con_clip()`; si la mejor `p >= umbrales.clip_senal` → `APROXIMADA, CLIP, id, su categoría, p`. Si no, CLIP sobre `[clip de reglamentaria, preventiva, informativa, clip_otro]`; si gana una categoría con `p >= umbrales.clip_categoria` → `SOLO_CATEGORIA, CLIP, None, esa categoría, p`. Si no (gana "otro" o no alcanza) → `NO_RECONOCIDA, NINGUNA, None, None, 0.0`.

Por cada llamada a `decidir`, `respaldo.puntuar` se llama **a lo sumo una vez por conjunto de textos** (velocidades, luego señales, luego categorías), solo con los recortes que lo necesitan, y nunca con una lista vacía.

- [ ] **Step 1: Escribir las pruebas que fallan** (catálogo `catalogo_prueba.yaml`, `U = Umbrales()`)

```python
def test_stop_confiable_no_usa_clip():
    assert decisor.decidir([recorte(1)], [Prediccion("Stop", 0.97)], U) == [
        Decision(Estado.RECONOCIDA, Fuente.GTSRB, "pare", Categoria.REGLAMENTARIA, 0.97)]
    assert respaldo.llamadas == []

def test_priority_road_se_ignora_y_va_a_clip():      # puntajes {1: {"resalto": 9}}
    [d] = decisor.decidir([recorte(1)], [Prediccion("Priority road", 0.99)], U)
    assert (d.estado, d.fuente, d.id_catalogo, d.categoria) == (
        Estado.APROXIMADA, Fuente.CLIP, "resalto", Categoria.PREVENTIVA)

def test_baja_confianza_queda_en_categoria():         # {1: {"cat-prev": 5}}
    [d] = decisor.decidir([recorte(1)], [Prediccion("Stop", 0.5)], U)
    assert (d.estado, d.fuente, d.id_catalogo, d.categoria) == (
        Estado.SOLO_CATEGORIA, Fuente.CLIP, None, Categoria.PREVENTIVA)

def test_velocidad_confirmada():                      # {1: {"v50": 5}}
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (50km/h)", 0.95)], U)
    assert d == Decision(Estado.RECONOCIDA, Fuente.GTSRB, "velocidad_50", Categoria.REGLAMENTARIA, 0.95)

def test_velocidad_que_gtsrb_no_tiene():              # {1: {"v40": 5}}
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (30km/h)", 0.93)], U)
    assert (d.estado, d.fuente, d.id_catalogo) == (Estado.APROXIMADA, Fuente.CLIP, "velocidad_40")

def test_velocidad_en_desacuerdo():                   # {1: {"v60": 5}}
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (30km/h)", 0.93)], U)
    assert d == Decision(Estado.SOLO_CATEGORIA, Fuente.GTSRB, "velocidad_no_verificada",
                         Categoria.REGLAMENTARIA, 0.93)

def test_velocidad_clip_inseguro():                   # {1: {"v40": 0.02}} → p = 0.4
    [d] = decisor.decidir([recorte(1)], [Prediccion("Speed limit (30km/h)", 0.93)], U)
    assert d.id_catalogo == "velocidad_no_verificada"

def test_otro_gana_no_reconocida():                   # {1: {"otro": 5}}
    [d] = decisor.decidir([recorte(1)], [Prediccion("Priority road", 0.99)], U)
    assert d == Decision(Estado.NO_RECONOCIDA, Fuente.NINGUNA, None, None, 0.0)

def test_lote_agrupa_llamadas():   # {2: {"v30": 5}, 3: {"resalto": 5}}
    decisor.decidir([recorte(1), recorte(2), recorte(3)],
                    [Prediccion("Stop", 0.97), Prediccion("Speed limit (30km/h)", 0.93),
                     Prediccion("Priority road", 0.99)], U)
    assert respaldo.llamadas == [(1, 4), (1, 5)]

def test_umbral_gtsrb_por_llamada():                  # {1: {"otro": 5}}
    p = [Prediccion("Stop", 0.85)]
    assert decisor.decidir([recorte(1)], p, Umbrales(gtsrb=0.80))[0].estado == Estado.RECONOCIDA
    assert decisor.decidir([recorte(1)], p, Umbrales(gtsrb=0.90))[0].estado == Estado.NO_RECONOCIDA
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_decision.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `Decisor` y `tests/fakes.py` (solo `FakeRespaldo` y `recorte` por ahora).**
- [ ] **Step 4: Verificar** — Run: `pytest -v` — Expected: todo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: decisor con verificación de velocidad y respaldo CLIP"`.

---

### Task 6: Clasificador GTSRB y descarga de modelos

**Files:**
- Create: `roadvision/classifier.py`, `scripts/download_models.py`, `tests/conftest.py`, `tests/test_classifier_modelo.py`

**Interfaces:**
- Consumes: `Dispositivo` (Tarea 3), `Prediccion` (Tarea 1), `Config` (Tarea 1).
- Produces:
  - `class ClasificadorGtsrb`: `__init__(self, modelo_id: str, dispositivo: Dispositivo)`; `etiquetas: list[str]` (de `id2label`, en orden de id; si alguna es `None` → `ValueError`); `clasificar(self, recortes: list[Image.Image], tamano_lote: int = 32) -> list[Prediccion]` (top-1 tras softmax; lista vacía → `[]`). Cumple `Clasificador`.
  - `scripts/download_models.py`: `descargar(ids: Sequence[str]) -> None` con `huggingface_hub.snapshot_download` para los 3 ids de `Config`; imprime `OK <id>` por modelo; volver a correrlo no descarga de nuevo.
  - `tests/conftest.py` (fixtures de sesión, solo usadas por pruebas `modelos`): `dispositivo` (= `elegir_dispositivo("auto")`), `clasificador`, `muestra_gtsrb(clase: int) -> Image.Image` (primera imagen del test de GTSRB con esa clase, vía `torchvision.datasets.GTSRB(root=RAIZ / "datasets", split="test", download=True)`), `escena_con_pare(ancho, alto, x, y, lado) -> tuple[Image.Image, Caja]` (fondo gris `(128,128,128)` con `muestra_gtsrb(14)` redimensionada a `lado×lado` pegada en `(x, y)`; devuelve la caja `(x, y, x+lado, y+lado)`).

Carga: `AutoImageProcessor` + `AutoModelForImageClassification`; el `preprocessor_config.json` del modelo declara `ViTFeatureExtractor`, así que si `AutoImageProcessor` no lo acepta en transformers 5.22.2, usar `ViTImageProcessor` con 224×224, media 0.5 y desviación 0.5. Modelo en `dispositivo.nombre` y `dispositivo.dtype`, `eval()`, inferencia bajo `torch.inference_mode()`.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
# tests/test_classifier_modelo.py
pytestmark = pytest.mark.modelos

def test_etiquetas_en_orden(clasificador):
    assert clasificador.etiquetas == json.loads((FIXTURES / "gtsrb_etiquetas.json").read_text("utf-8"))

def test_clasifica_pare_de_gtsrb(clasificador, muestra_gtsrb):
    [p] = clasificador.clasificar([muestra_gtsrb(14)])
    assert p.etiqueta == "Stop" and p.probabilidad >= 0.8

def test_lista_vacia(clasificador):
    assert clasificador.clasificar([]) == []
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest -m modelos tests/test_classifier_modelo.py -v` — Expected: FAIL (`ClasificadorGtsrb` no existe).
- [ ] **Step 3: Implementar `classifier.py`, `download_models.py` y `conftest.py`.**
- [ ] **Step 4: Descargar y verificar** — Run: `python scripts/download_models.py` — Expected: 3 líneas `OK …`. Run: `pytest -m modelos tests/test_classifier_modelo.py -v` — Expected: 3 passed (la primera vez descarga el test de GTSRB a `datasets/`). Run: `pytest -v` — Expected: las pruebas unitarias siguen pasando y las `modelos` no se ejecutan.
- [ ] **Step 5: Commit** — `git commit -m "feat: clasificador ViT GTSRB y descarga de modelos"`.

---

### Task 7: Detector OWLv2

**Files:**
- Create: `roadvision/detector.py`, `tests/test_detector_modelo.py`

**Interfaces:**
- Consumes: `Dispositivo` (Tarea 3), `Deteccion` (Tarea 1), `iou` (Tarea 2), fixtures de Tarea 6.
- Produces: `class DetectorOwlv2`: `__init__(self, modelo_id: str, consultas: Sequence[str], dispositivo: Dispositivo)`; `detectar(self, imagen: Image.Image, umbral: float) -> list[Deteccion]` — cajas de todas las consultas con puntaje ≥ umbral, en enteros y dentro de la imagen; sin eliminar duplicados (eso lo hace el pipeline). Cumple `Detector`.

Enfoque: `Owlv2Processor` + `Owlv2ForObjectDetection`; `processor(text=[list(consultas)], images=imagen, return_tensors="pt")`; `processor.post_process_grounded_object_detection(outputs, threshold=umbral, target_sizes=[(alto, ancho)])`. En transformers 5.22.2 ese post-proceso ya compensa el relleno a cuadrado de OWLv2 (`_scale_boxes`), así que se pasa el tamaño original; la segunda prueba lo comprueba.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
pytestmark = pytest.mark.modelos

def test_detecta_senal_pegada(detector, escena_con_pare):
    img, caja = escena_con_pare(800, 600, 420, 120, 128)
    dets = detector.detectar(img, umbral=0.10)
    assert max(iou(d.caja, caja) for d in dets) >= 0.3

def test_imagen_no_cuadrada_cajas_en_su_sitio(detector, escena_con_pare):
    img, caja = escena_con_pare(1200, 400, 1000, 250, 128)
    dets = detector.detectar(img, umbral=0.10)
    assert max(iou(d.caja, caja) for d in dets) >= 0.3
    assert all(0 <= d.caja[0] < d.caja[2] <= 1200 and 0 <= d.caja[1] < d.caja[3] <= 400 for d in dets)

def test_imagen_lisa_sin_detecciones(detector):
    assert detector.detectar(Image.new("RGB", (640, 480), (128, 128, 128)), umbral=0.15) == []
```
Agregar la fixture de sesión `detector` en `conftest.py`.

- [ ] **Step 2: Verificar que fallan** — Run: `pytest -m modelos tests/test_detector_modelo.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `DetectorOwlv2`.**
- [ ] **Step 4: Verificar** — Run: `pytest -m modelos tests/test_detector_modelo.py -v` — Expected: 3 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: detector genérico de señales con OWLv2"`.

---

### Task 8: Respaldo CLIP

**Files:**
- Create: `roadvision/fallback.py`, `tests/test_fallback_modelo.py`

**Interfaces:**
- Consumes: `Dispositivo` (Tarea 3), fixtures de Tarea 6.
- Produces: `class RespaldoClip`: `__init__(self, modelo_id: str, dispositivo: Dispositivo)`; `puntuar(self, recortes: list[Image.Image], textos: list[str]) -> list[list[float]]` — softmax de `logits_per_image` sobre los textos, como floats de Python; `recortes == []` → `[]`. Cumple `Respaldo`. Usa `CLIPModel` + `CLIPProcessor`.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
pytestmark = pytest.mark.modelos

def test_filas_suman_uno_y_pare_prefiere_octogono(respaldo, muestra_gtsrb):
    [fila] = respaldo.puntuar([muestra_gtsrb(14)],
                              ["a red octagonal stop sign", "a yellow diamond-shaped warning road sign"])
    assert sum(fila) == pytest.approx(1.0, abs=1e-3)
    assert fila[0] > fila[1]

def test_vacio(respaldo):
    assert respaldo.puntuar([], ["x"]) == []
```
Agregar la fixture de sesión `respaldo` en `conftest.py`.

- [ ] **Step 2: Verificar que fallan** — Run: `pytest -m modelos tests/test_fallback_modelo.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `RespaldoClip`.**
- [ ] **Step 4: Verificar** — Run: `pytest -m modelos tests/test_fallback_modelo.py -v` — Expected: 2 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: respaldo CLIP para señales fuera de GTSRB"`.

---

### Task 9: Pipeline

**Files:**
- Create: `roadvision/pipeline.py`, `tests/test_pipeline.py`, `tests/test_pipeline_modelo.py`
- Modify: `tests/fakes.py` (agregar `FakeDetector`, `FakeClasificador`, `pipeline_falso`)

**Interfaces:**
- Consumes: todo lo anterior.
- Produces:
  - `@dataclass(frozen=True) CargaModelo(tiempo_ms: float, vram_mb: float)` — `vram_mb` es el aumento de `torch.cuda.memory_allocated()` al cargar (0.0 en CPU).
  - `class Pipeline`: `__init__(self, config: Config, detector: Detector, clasificador: Clasificador, respaldo: Respaldo, catalogo: Catalogo, dispositivo: Dispositivo, carga: dict[str, CargaModelo] | None = None)` (crea su propio `Decisor`); atributos `carga`, `dispositivo`; `analizar(self, imagen: Image.Image, umbrales: Umbrales | None = None) -> Resultado` (por defecto `config.umbrales`); `calentar(self) -> dict[str, float]` (llama una vez a cada modelo con una entrada sintética y devuelve ms con claves `deteccion`, `clasificacion`, `respaldo`).
  - `construir_pipeline(config: Config | None = None) -> Pipeline` — elige dispositivo con `config.dispositivo`/`config.fp16`, carga catálogo y los 3 modelos reales registrando `carga` con claves `detector`, `clasificador`, `respaldo`.
  - `tests/fakes.py`: `FakeDetector(detecciones: list[Deteccion])` (guarda la última imagen recibida), `FakeClasificador(predicciones: list[Prediccion], etiquetas: list[str] | None = None)` (consume `predicciones` en orden a lo largo de todas sus llamadas, volviendo al inicio cuando se acaban; acumula en `recibidos: int` cuántos recortes recibió), `pipeline_falso(detecciones, predicciones, respaldo=None) -> Pipeline` (con `catalogo_prueba.yaml` y dispositivo CPU).

`analizar` (dentro de `Cronometro(sincronizar=lambda: sincronizar(dispositivo))`, todo bajo la etapa `total`):
- `preproceso`: `normalizar`.
- `deteccion`: `detector.detectar(img, u.deteccion)` y `filtrar_cajas(..., u.deteccion, config.tamano_minimo_caja, config.iou_duplicados)`.
- `clasificacion`: `recortar(img, caja, config.margen_recorte)` por caja y `clasificador.clasificar(recortes)` (si no hay recortes, no se llama).
- `respaldo`: `decisor.decidir(...)`.
- Las señales se ordenan por `(x1, y1)` y se numeran desde 1. `nombre`/`significado`: de la entrada del catálogo; en `SOLO_CATEGORIA` sin id, de `catalogo.categorias[categoria]`; en `NO_RECONOCIDA`, `catalogo.nombre_no_reconocida` y `significado=None`.
- `conteo_por_nombre`: cuenta todas las señales por `nombre`. `conteo_por_categoria`: por `categoria.value`, o `"sin_categoria"` si no tiene.
- `tiempos_ms` siempre trae `total`, `preproceso`, `deteccion`, `clasificacion`, `respaldo` (0.0 si no se ejecutó). `dispositivo = dispositivo.descripcion`.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
# tests/test_pipeline.py
def test_sin_detecciones():
    respaldo = FakeRespaldo()
    p = pipeline_falso([], [], respaldo)
    r = p.analizar(Image.new("RGB", (640, 480)))
    assert (r.senales, r.conteo_por_nombre, r.conteo_por_categoria) == ([], {}, {})
    assert respaldo.llamadas == []
    assert set(r.tiempos_ms) == {"total", "preproceso", "deteccion", "clasificacion", "respaldo"}
    assert (r.ancho, r.alto) == (640, 480)

def test_dos_pare_y_una_no_reconocida():
    dets = [Deteccion((300, 50, 360, 110), 0.9), Deteccion((100, 50, 160, 110), 0.8),
            Deteccion((500, 50, 560, 110), 0.7)]
    preds = [Prediccion("Stop", 0.97), Prediccion("Stop", 0.95), Prediccion("Priority road", 0.99)]
    r = pipeline_falso(dets, preds, FakeRespaldo(por_defecto={"otro": 5})).analizar(Image.new("RGB", (640, 480)))
    assert [s.numero for s in r.senales] == [1, 2, 3]
    assert [s.caja[0] for s in r.senales] == [100, 300, 500]
    assert [s.nombre for s in r.senales] == ["PARE", "PARE", "Señal no reconocida"]
    assert r.conteo_por_nombre == {"PARE": 2, "Señal no reconocida": 1}
    assert r.conteo_por_categoria == {"reglamentaria": 2, "sin_categoria": 1}

def test_duplicados_y_pequenas():
    dets = [Deteccion((100, 100, 150, 150), 0.9), Deteccion((102, 101, 152, 149), 0.6),
            Deteccion((300, 300, 310, 330), 0.9)]
    clasif = FakeClasificador([Prediccion("Stop", 0.97)])
    r = pipeline_con(dets, clasif).analizar(Image.new("RGB", (640, 480)))
    assert len(r.senales) == 1 and clasif.recibidos == 1

def test_umbral_de_deteccion_por_llamada():
    dets = [Deteccion((100, 100, 150, 150), 0.20), Deteccion((300, 100, 350, 150), 0.12)]
    p = pipeline_falso(dets, [Prediccion("Stop", 0.97)] * 2)
    assert len(p.analizar(IMG, Umbrales(deteccion=0.15)).senales) == 1
    assert len(p.analizar(IMG, Umbrales(deteccion=0.10)).senales) == 2

def test_exif_rotado():
    r = pipeline_falso([], []).analizar(jpeg_con_orientacion(400, 200, 6))
    assert (r.ancho, r.alto) == (200, 400)

def test_calentar_usa_cada_modelo():
    respaldo = FakeRespaldo()
    p = pipeline_falso([], [Prediccion("Stop", 0.9)], respaldo)
    assert set(p.calentar()) == {"deteccion", "clasificacion", "respaldo"}
    assert len(respaldo.llamadas) == 1

# tests/test_pipeline_modelo.py
pytestmark = pytest.mark.modelos

def test_pipeline_real_reconoce_pare(escena_con_pare):
    p = construir_pipeline()
    img, caja = escena_con_pare(800, 600, 420, 120, 128)
    r = p.analizar(img)
    assert any(s.id_catalogo == "pare" and iou(s.caja, caja) >= 0.3 for s in r.senales)
    assert set(p.carga) == {"detector", "clasificador", "respaldo"}
```
`pipeline_con(dets, clasificador)` es una variante local de `pipeline_falso` que recibe el clasificador; `IMG = Image.new("RGB", (640, 480))`; `jpeg_con_orientacion` se mueve de `test_imaging.py` a `tests/fakes.py` para compartirlo.

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_pipeline.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `pipeline.py` y completar `tests/fakes.py`.**
- [ ] **Step 4: Verificar** — Run: `pytest -v` — Expected: todo PASS. Run: `pytest -m modelos tests/test_pipeline_modelo.py -v` — Expected: 1 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: pipeline completo detector → clasificador → decisor"`.

---

### Task 10: Comando de línea

**Files:**
- Create: `roadvision/cli.py` (se ejecuta como `python -m roadvision.cli`), `tests/test_cli.py`

**Interfaces:**
- Consumes: `Config`, `construir_pipeline`, `Pipeline`, `Resultado` (Tareas 1 y 9).
- Produces: `main(argv: Sequence[str] | None = None, construir: Callable[[Config], Pipeline] = construir_pipeline) -> int`. Argumentos: `imagen` (ruta), `--dispositivo {auto,cpu,cuda}` (por defecto `auto`), `--sin-fp16`. La existencia del archivo se verifica **antes** de construir el pipeline: si no existe, escribe en stderr `No existe el archivo: <ruta>` y devuelve 2. Si existe, imprime una línea por señal con el formato `#{numero} {nombre} [{categoria o 'sin categoría'}] {estado} conf {confianza:.2f}`, debajo el significado si lo hay, luego `Resumen: ` con `conteo_por_nombre`, luego `Tiempos (ms): total=… preproceso=… deteccion=… clasificacion=… respaldo=…`, y devuelve 0.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
def test_cli_imprime_senales(tmp_path, capsys):
    ruta = tmp_path / "foto.jpg"; Image.new("RGB", (640, 480)).save(ruta)
    falso = pipeline_falso([Deteccion((100, 100, 160, 160), 0.9)], [Prediccion("Stop", 0.97)])
    assert main([str(ruta)], construir=lambda c: falso) == 0
    out = capsys.readouterr().out
    assert "#1 PARE [reglamentaria] reconocida conf 0.97" in out
    assert "Resumen:" in out and "total=" in out

def test_cli_archivo_inexistente(capsys):
    assert main(["no_existe.jpg"], construir=lambda c: None) == 2
    assert "No existe el archivo" in capsys.readouterr().err

def test_cli_pasa_dispositivo(tmp_path):
    ruta = tmp_path / "foto.jpg"; Image.new("RGB", (64, 64)).save(ruta)
    recibido = {}
    def construir(c):
        recibido["c"] = c
        return pipeline_falso([], [])
    main([str(ruta), "--dispositivo", "cpu", "--sin-fp16"], construir=construir)
    assert (recibido["c"].dispositivo, recibido["c"].fp16) == ("cpu", False)
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_cli.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `cli.py`** (con `if __name__ == "__main__": raise SystemExit(main())`).
- [ ] **Step 4: Verificar** — Run: `pytest -v` — Expected: todo PASS. Run: `python -m roadvision.cli <una foto propia con señales>` — Expected: lista de señales, resumen y tiempos (en CPU tarda varios segundos).
- [ ] **Step 5: Commit** — `git commit -m "feat: comando de línea para analizar una foto"`.

---

### Task 11: Evaluación nivel A, README y lista de verificación

**Files:**
- Create: `roadvision/evaluacion.py`, `scripts/evaluate.py`, `tests/test_evaluacion.py`
- Modify: `README.md` (secciones de esta parte; el resto del equipo agrega las suyas)

**Interfaces:**
- Consumes: `Clasificador` (Tarea 1), `ClasificadorGtsrb` (Tarea 6), `elegir_dispositivo` (Tarea 3).
- Produces:
  - `@dataclass(frozen=True) ReporteA(n: int, exactitud: float, exactitud_por_etiqueta: dict[str, float], imagenes_por_segundo: float)`
  - `evaluar_clasificador(clasificador: Clasificador, muestras: Iterable[tuple[Image.Image, int]], tamano_lote: int = 64) -> ReporteA` — la etiqueta correcta de una muestra es `clasificador.etiquetas[clase]`.
  - `scripts/evaluate.py`: `--nivel a` (única opción en este plan; H5 agrega `b` y `c`), `--limite N`, `--dispositivo`, `--sin-fp16`. Usa el test de GTSRB en `datasets/`. Imprime `n`, `exactitud` con 4 decimales, las 5 etiquetas con peor exactitud e imágenes por segundo.

- [ ] **Step 1: Escribir la prueba que falla**

```python
def test_evaluar_clasificador():
    clasif = FakeClasificador([Prediccion("A", 0.9), Prediccion("B", 0.9), Prediccion("A", 0.9)],
                              etiquetas=["A", "B"])
    img = Image.new("RGB", (32, 32))
    r = evaluar_clasificador(clasif, [(img, 0), (img, 1), (img, 1)], tamano_lote=2)
    assert r.n == 3
    assert r.exactitud == pytest.approx(2 / 3)
    assert r.exactitud_por_etiqueta == {"A": 1.0, "B": 0.5}
    assert r.imagenes_por_segundo > 0
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_evaluacion.py -v` — Expected: FAIL.
- [ ] **Step 3: Implementar `evaluacion.py` y `scripts/evaluate.py`.**
- [ ] **Step 4: Verificar con datos reales** — Run: `python scripts/evaluate.py --nivel a --limite 2000 --dispositivo cpu` — Expected: `exactitud ≥ 0.95`. Si da cerca de 0.02, el orden de etiquetas del modelo no coincide con los ids de GTSRB: detenerse y revisar antes de seguir.
- [ ] **Step 5: Escribir en `README.md`**:
  - **Instalación (modelos):** Python 3.11/3.12 y `venv`; torch+torchvision desde el selector de pytorch.org (CUDA en el PC NVIDIA, CPU en los demás); `pip install -r requirements.txt`; `python scripts/download_models.py`; `python scripts/check_gpu.py`.
  - **Uso:** `python -m roadvision.cli foto.jpg [--dispositivo cpu|cuda] [--sin-fp16]`.
  - **Pruebas:** `pytest` (rápidas, sin modelos) y `pytest -m modelos` (con modelos reales).
  - **Lista de verificación en el PC NVIDIA:** (1) `check_gpu.py` sale con código 0; (2) `pytest -m modelos` pasa con `--dispositivo` CUDA; (3) `evaluate.py --nivel a --dispositivo cuda` sobre el test completo da ≥ 0.95; (4) repetir con `--sin-fp16` y comprobar que la exactitud no cambia más de 0.5 puntos.
  - **Créditos:** los tres modelos con su licencia (OWLv2 Apache-2.0, ViT GTSRB MIT, CLIP según su ficha) y GTSRB.
- [ ] **Step 6: Verificar todo** — Run: `pytest -v` y `pytest -m modelos -v` — Expected: todo PASS.
- [ ] **Step 7: Commit, publicar y abrir PR**

```bash
git add roadvision/evaluacion.py scripts/evaluate.py tests/test_evaluacion.py README.md
git commit -m "feat: evaluación nivel A del clasificador e instrucciones de instalación"
git push
gh pr create --base main --head feat/modelos --title "feat: núcleo de modelos de RoadVision (H1–H2)"
```
