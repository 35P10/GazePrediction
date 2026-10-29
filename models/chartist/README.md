# Chartist

`model.py` carga los cuatro pesos PPO y construye directamente `LLMPolicy` del repositorio oficial.
Sus métodos, prompts, memoria, movimientos y terminación permanecen sin cambios.
La adaptación se limita a las rutas locales y la conexión LLM propia.

```python
import json
from models import load_predictor
from models.chartist import ChartistInput

cases = [ChartistInput(**item) for item in json.loads((DATA / 'meta.json').read_text())]
predictor = load_predictor('chartist')  # Usa OPENAI_API_KEY; también acepta api_key= o client=.
meta = cases[0]
prediction = predictor.predict(meta, meta_json=DATA / 'meta.json', folder=DATA / 'images')
predictions = predictor.run(cases, meta_json=DATA / 'meta.json', folder=DATA / 'images')
```

Cada predicción contiene `points` en 320 × 320, `answer`, `actions` y `log`. El `log` guarda:
- **cada llamada al LLM:** número de decisión, mensajes enviados, respuesta, modelo servido, tokens, tiempo y error, si lo hubo;
- **el estado final del controlador:** celdas de mirada, memoria y resumen de memoria;
- **lo que imprime el código oficial y la duración.**

Si una predicción falla, el log parcial queda en `predictor.last_log`. El registro envuelve el cliente del LLM
y no modifica el código oficial.
`run()` devuelve una lista en el orden de los metadatos recibidos. La clase acepta `api_key`,
`client` y `device` (CPU por defecto). No solicita claves interactivamente ni escribe resultados.
La memoria conserva su valor original de treinta elementos. El máximo de decisiones se fija en **diez**
(`MAX_STEPS = 10`, pasado al constructor oficial), no en las cinco del release: con cinco, un recorrido
no puede pasar de 25 fijaciones, y el paper reporta medias de 46–49 (DE 3–7), compatibles con un tope de 50.
Es una inferencia a partir de la Tabla 1; los autores no publican ese valor.

La entrada oficial es `LLMPolicy.setup(meta, meta_json, folder)`. `ChartistInput` tipa `meta`
con el formato de los registros publicados: `id`, `image`, `task`, `chart_type`, `question`
y `text_positions`. `ChartistText` describe cada texto OCR y su caja `[left, top, right, bottom]`
en la imagen de 320 × 320; las puntuaciones OCR y de similitud son opcionales.
`meta_json` y `folder` son rutas `str` o `Path`, recibidas por separado como en el código oficial.
Estos tipos son diccionarios tipados; no convierten ni validan datos en ejecución.

## Validar `text_positions` con PaddleOCR

[ocr.py](ocr.py) recalcula `text_positions` como `data_gen.py`:

1. Ejecuta PaddleOCR sobre `chart_<id>_origin.png`.
2. Escala cada caja a 320 × 320.
3. Calcula la similitud de cada texto con la pregunta usando `stsb-roberta-base-v2`.

Con esto se comprueba que los datos nuevos tienen el mismo formato y contenido que esperan las políticas.
Sobre el `meta.json` oficial reproduce las 40 condiciones: mismas cajas, textos y posiciones, y puntuaciones
con diferencias menores de 3·10⁻⁶.

PaddlePaddle no funciona en el entorno del modelo (Python 3.8 en Apple Silicon), así que usa su propio entorno:

```bash
uv venv .venv-chartist-ocr -p 3.10
uv pip install -p .venv-chartist-ocr/bin/python -r models/chartist/requirements-ocr.txt
.venv-chartist-ocr/bin/python -m models.chartist.ocr DATA/meta.json DATA/images
```

Desde Python, `validate_text_positions(meta, origin_image)` devuelve la lista de diferencias (vacía si coincide),
y `ocr_text_positions(origin_image, question)` genera `text_positions` para una imagen nueva.

## Archivos oficiales

`official_reference/model/` contiene los diez archivos Python y cuatro pesos oficiales intactos.
Su procedencia está en `official_source.json`. Los datos y scripts publicados están en
`reproductions/chartist_taskvis/official_reference/` desde la raíz del proyecto.

La evaluación frente al paper está en
[01_reproduce_chartist.ipynb](../../reproductions/chartist_taskvis/01_reproduce_chartist.ipynb).
