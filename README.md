# RoadVision

Reconocimiento de señales de tránsito en fotos. El núcleo (`roadvision/`) detecta las
señales con OWLv2, las clasifica con un ViT entrenado en GTSRB, usa CLIP como respaldo
para señales que GTSRB no cubre y decide el resultado final de cada señal.

Este README tiene las secciones de la parte de **modelos**. Las demás partes del
equipo (interfaz, métricas) agregan sus propias secciones más abajo.

## Modelos: instalación

Requisitos: Python 3.11 o 3.12.

1. Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   # Windows (PowerShell): .venv\Scripts\Activate.ps1
   # Linux/macOS:          source .venv/bin/activate
   ```

2. Instalar PyTorch y torchvision.
   - Sin GPU NVIDIA (CPU):

     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```

   - PC con GPU NVIDIA: elegir la variante CUDA en el selector de
     https://pytorch.org/get-started/locally/ y ejecutar el comando que indica.

3. Instalar el resto de dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Descargar los modelos (una sola vez, se guardan en la caché de Hugging Face):

   ```bash
   python scripts/download_models.py
   ```

5. Comprobar la GPU (sale con código 0 si PyTorch ve CUDA y fp16 funciona):

   ```bash
   python scripts/check_gpu.py
   ```

Los datasets (GTSRB, usado por la evaluación) se descargan en `datasets/` dentro del
repositorio, que está ignorado por git. Para usar otra carpeta, definir la variable de
entorno `ROADVISION_DATASETS`.

## Modelos: uso

Analizar una foto desde la línea de comandos:

```bash
python -m roadvision.cli foto.jpg [--dispositivo cpu|cuda] [--sin-fp16]
```

Imprime, por cada señal, su número, nombre, categoría, estado, confianza y significado,
además de un resumen y los tiempos por etapa. Sin `--dispositivo` se usa CUDA si está
disponible; `--sin-fp16` desactiva la media precisión.

## Modelos: pruebas

```bash
pytest              # rápidas, sin modelos
pytest -m modelos   # con los modelos reales (requiere haberlos descargado)
```

## Modelos: evaluación (nivel A)

Exactitud del clasificador sobre el conjunto de prueba oficial de GTSRB (12 630
imágenes; la primera ejecución lo descarga):

```bash
python scripts/evaluate.py --nivel a [--limite N] [--dispositivo cpu|cuda] [--sin-fp16]
```

Imprime `n`, la exactitud con 4 decimales, las 5 etiquetas con peor exactitud y las
imágenes por segundo.

## Lista de verificación en el PC NVIDIA

1. `python scripts/check_gpu.py` termina con código 0.
2. `pytest -m modelos` pasa usando CUDA (dispositivo `auto` con GPU disponible).
3. `python scripts/evaluate.py --nivel a --dispositivo cuda` sobre el test completo da
   exactitud >= 0.95.
4. Repetir el punto 3 con `--sin-fp16`: la exactitud no debe cambiar más de 0.5 puntos.

## Créditos

- Detector: [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)
  (`google/owlv2-base-patch16-ensemble`), licencia Apache-2.0.
- Clasificador: [ViT GTSRB](https://huggingface.co/kelvinandreas/vit-traffic-sign-GTSRB)
  (`kelvinandreas/vit-traffic-sign-GTSRB`), licencia MIT.
- Respaldo: [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
  (`openai/clip-vit-base-patch32`), ver la ficha del modelo.
- Datos: GTSRB (German Traffic Sign Recognition Benchmark, Institut für Neuroinformatik,
  Ruhr-Universität Bochum), ver términos del benchmark.
