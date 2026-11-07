# Guía: instalar y probar RoadVision en el PC con GPU NVIDIA

> **Para Claude (o quien ejecute esta guía):** este documento son instrucciones para ejecutar
> en orden, en el PC que tiene la GPU NVIDIA. El código del núcleo de modelos ya está
> terminado y probado en un PC **sin** GPU; lo que falta es comprobar que todo funciona en
> **CUDA** y medir su rendimiento. Al final hay que entregar un informe de resultados.

## Reglas para quien ejecuta

1. Ejecuta los pasos **en orden**. Si un paso falla, intenta lo indicado en
   [Solución de problemas](#solución-de-problemas); si sigue fallando, **detente** y
   anota el error completo en el informe.
2. **No modifiques** el código de `roadvision/`, `scripts/` ni `tests/` para que una
   prueba pase. Si algo falla, se reporta; los arreglos los hace el equipo.
3. **No hagas push a `main`** ni fusiones nada. El informe se entrega en una rama aparte
   y solo después de que la persona dueña del PC lo apruebe (paso 10).
4. Los comandos de esta guía llaman siempre al Python del entorno virtual por su ruta
   (`.venv\Scripts\python.exe`) en vez de "activar" el entorno: así funcionan aunque cada
   comando se ejecute en una terminal nueva.
5. Guarda la salida de cada paso: la vas a necesitar para el informe.

Los comandos están para **Windows (PowerShell)**. En Linux, cambia
`.venv\Scripts\python.exe` por `.venv/bin/python` y las barras `\` por `/`.

---

## Paso 0. Requisitos previos

Comprueba cada uno y anota la salida:

```powershell
nvidia-smi
git --version
python --version
```

- `nvidia-smi` debe mostrar la GPU. Anota el **modelo**, la **VRAM total**, la versión del
  **driver** y la **"CUDA Version"** de la esquina superior derecha: esa es la versión
  máxima de CUDA que soporta el driver.
- Python debe ser **3.11 o 3.12** (el proyecto se probó en 3.12). Si `python --version`
  da otra versión, instala 3.12 desde python.org y usa `py -3.12` en lugar de `python`
  en el paso 2.
- Espacio libre: unos **4 GB** (PyTorch con CUDA, ~1,5 GB de modelos y ~100 MB de datos).

Si `nvidia-smi` no existe o no muestra la GPU, **detente**: hay que instalar o actualizar
el driver de NVIDIA antes de seguir.

## Paso 1. Obtener el código

Clona el repositorio en una carpeta que **no** esté sincronizada con OneDrive, Google Drive
o Dropbox (sincronizar el entorno virtual y los modelos es muy lento). Por ejemplo
`C:\proyectos`:

```powershell
cd C:\proyectos
git clone https://github.com/DeimonBot/TrafficSigns.git
cd TrafficSigns
git checkout main
git pull
```

Si el repositorio ya estaba clonado, basta con `git checkout main` y `git pull` dentro de él.

Comprueba que el código de modelos está presente:

```powershell
dir roadvision
```

Debe listar, entre otros, `pipeline.py`, `detector.py`, `classifier.py` y `fallback.py`.

## Paso 2. Crear el entorno virtual

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

## Paso 3. Instalar PyTorch con CUDA

1. Abre https://pytorch.org/get-started/locally/ y elige: *Stable*, tu sistema operativo,
   *Pip*, *Python* y una versión de **CUDA menor o igual** a la "CUDA Version" que mostró
   `nvidia-smi` en el paso 0.
2. La página muestra un comando del estilo
   `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`.
   Ejecútalo con el Python del entorno, cambiando `pip3` por `.venv\Scripts\python.exe -m pip`:

   ```powershell
   .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
   ```

   (reemplaza `cuXXX` por el que indique la página; no inventes el número).

3. Verifica que PyTorch ve la GPU:

   ```powershell
   .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

   Debe terminar en `True`. Si dice `False` o la versión termina en `+cpu`, mira
   [Solución de problemas](#solución-de-problemas).

## Paso 4. Instalar el resto de dependencias

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` **no** incluye torch a propósito; no lo reinstales desde ahí.

## Paso 5. Descargar los modelos

```powershell
.venv\Scripts\python.exe scripts\download_models.py
```

Debe imprimir tres líneas `OK` (OWLv2, ViT GTSRB y CLIP). Descarga cerca de 1,5 GB a la
caché de Hugging Face del usuario; volver a ejecutarlo no descarga de nuevo.

## Paso 6. Verificar la GPU

```powershell
.venv\Scripts\python.exe scripts\check_gpu.py
echo "codigo de salida: $LASTEXITCODE"
```

**Esperado:** muestra la GPU, su VRAM y `matmul fp16 en GPU: OK ...`, y el código de
salida es **0**.

## Paso 7. Pruebas automáticas

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pytest -m modelos -v
```

**Esperado:**
- La primera (rápida, sin modelos) da **58 passed** (y 10 deselected).
- La segunda carga los modelos reales **en la GPU** (sin opciones, el proyecto usa CUDA si
  está disponible) y da **10 passed**. La primera vez descarga el conjunto de prueba de
  GTSRB a la carpeta `datasets\` del repositorio (ignorada por git).

Anota los tiempos que imprime pytest.

## Paso 8. Evaluación del clasificador en el test completo de GTSRB

Corre la evaluación dos veces, con y sin media precisión (fp16):

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --nivel a --dispositivo cuda
.venv\Scripts\python.exe scripts\evaluate.py --nivel a --dispositivo cuda --sin-fp16
```

Cada una procesa las 12 630 imágenes. **Esperado:**
- `exactitud` **≥ 0.95** en ambas (en CPU, con las primeras 2000 imágenes, dio 0.9505).
- La diferencia entre las dos exactitudes es de **0.005 o menos** (medio punto).
- Anota `imágenes por segundo` de cada una.

## Paso 9. Probar el comando de línea con una escena de prueba

1. Crea el archivo `escena_prueba.py` **fuera** del repositorio (por ejemplo en
   `C:\proyectos\escena_prueba.py`) con este contenido. Pega una señal PARE del test de
   GTSRB sobre un fondo gris:

   ```python
   import sys
   from pathlib import Path

   from PIL import Image
   from torchvision.datasets import GTSRB

   carpeta_datos = Path(sys.argv[1])
   destino = Path(sys.argv[2])
   datos = GTSRB(root=carpeta_datos, split="test", download=True)
   for imagen, clase in datos:
       if clase == 14:  # Stop
           break

   escena = Image.new("RGB", (800, 600), (128, 128, 128))
   escena.paste(imagen.convert("RGB").resize((128, 128)), (420, 120))
   escena.save(destino, quality=95)
   print(f"escena guardada en {destino}")
   ```

2. Desde la carpeta del repositorio, genérala (el primer argumento es la carpeta de datos
   que usó el paso 7, `datasets`) y analízala con la CLI, en fp16 y en fp32:

   ```powershell
   .venv\Scripts\python.exe C:\proyectos\escena_prueba.py datasets C:\proyectos\escena_pare.jpg
   .venv\Scripts\python.exe -m roadvision.cli C:\proyectos\escena_pare.jpg --dispositivo cuda
   .venv\Scripts\python.exe -m roadvision.cli C:\proyectos\escena_pare.jpg --dispositivo cuda --sin-fp16
   ```

   **Esperado** en ambas: una línea `#1 PARE [reglamentaria] reconocida conf ...` con su
   significado, la línea `Calentamiento (ms, no incluido en los tiempos): ...` y la línea
   `Tiempos (ms): total=... preproceso=... deteccion=... clasificacion=... respaldo=...`.
   Anota las dos líneas de tiempos de cada ejecución. En el PC sin GPU el total era de
   12 a 14 segundos, casi todo en `deteccion`; en GPU debería ser mucho menor.

3. Mientras corre la CLI (o justo después), toma una foto del estado de la GPU:

   ```powershell
   nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
   ```

4. **Opcional, muy valioso:** si tienes fotos reales de calles colombianas con señales,
   analiza 3 a 5 con `-m roadvision.cli <foto>` y anota para cada una qué señales había de
   verdad y qué respondió el programa. No subas las fotos al repositorio.

## Paso 10. Informe de resultados

1. Crea el archivo `docs/resultados/AAAA-MM-DD-pc-nvidia.md` (con la fecha de hoy) usando
   la [plantilla del informe](#plantilla-del-informe). Rellena **todo** con las salidas
   reales; si un paso falló, pega el error completo y lo que intentaste.
2. Muestra el informe a la persona dueña del PC. **Solo con su aprobación**:

   ```powershell
   git checkout -b resultados/pc-nvidia
   git add docs/resultados
   git commit -m "docs: resultados de la prueba en el PC con NVIDIA"
   git push -u origin resultados/pc-nvidia
   ```

   y abre un pull request hacia `main` con el título
   "Resultados de la prueba en el PC con NVIDIA" (con `gh pr create` si está instalado, o
   desde el enlace que imprime `git push`). **No lo fusiones**: lo revisa el equipo.

---

## Plantilla del informe

```markdown
# Resultados en el PC con NVIDIA — AAAA-MM-DD

## Equipo
- GPU / VRAM / driver / "CUDA Version" de nvidia-smi:
- Sistema operativo:
- Python:
- torch / torch.version.cuda / torchvision:
- transformers:

## Pasos
| Paso | Resultado | Notas |
|---|---|---|
| 6. check_gpu.py (código de salida) | | |
| 7. pytest (rápidas) | | |
| 7. pytest -m modelos | | tiempo: |
| 8. evaluación fp16: exactitud / img/s | | |
| 8. evaluación fp32: exactitud / img/s | | |
| 8. diferencia fp16 − fp32 | | debe ser ≤ 0.005 |
| 9. CLI fp16: resultado y tiempos | | |
| 9. CLI fp32: resultado y tiempos | | |

## Salidas completas
(pegar aquí las salidas de los pasos 6, 8 y 9 y la de nvidia-smi del paso 9.3)

## Fotos reales (opcional)
| Foto | Señales reales | Lo que respondió RoadVision |
|---|---|---|

## Problemas encontrados
(errores, advertencias raras, pasos que hubo que repetir)
```

---

## Solución de problemas

| Síntoma | Qué hacer |
|---|---|
| `torch.cuda.is_available()` da `False` y la versión termina en `+cpu` | Se instaló el PyTorch de CPU. Desinstala (`.venv\Scripts\python.exe -m pip uninstall -y torch torchvision`) y repite el paso 3 con el comando CUDA de pytorch.org. |
| `False` aunque la versión trae CUDA | El driver es más viejo que el CUDA del paquete: elige en pytorch.org una variante de CUDA más baja (≤ "CUDA Version" de `nvidia-smi`) o actualiza el driver de NVIDIA. |
| `pip` no encuentra una versión de torch para tu Python | Usa Python 3.12 (paso 0). |
| `CUDA out of memory` | Cierra otros programas que usen la GPU (juegos, navegador con aceleración) y repite. Si persiste, anota el error en el informe junto con la VRAM de tu GPU. Ten en cuenta que `--sin-fp16` (fp32) usa **más** memoria que fp16. |
| Error al descargar modelos o GTSRB (red, tiempo de espera) | Vuelve a ejecutar el mismo comando: retoma lo que falta. |
| Mensaje de Hugging Face sobre "unauthenticated requests" | Es solo un aviso, no un error. Se puede ignorar. |
| Los acentos salen raros en la consola | Es la codificación de la consola de Windows; no afecta los resultados. |
| Activar el entorno da error de "execution policy" | No hace falta activarlo: esta guía llama a `.venv\Scripts\python.exe` directamente. |
| Una prueba de `pytest -m modelos` falla | No toques el código. Copia la salida completa al informe y sigue con los demás pasos. |
