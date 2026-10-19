# Mike-LLM v0.1

Proyecto educativo para entrenar un Transformer decoder-only desde pesos aleatorios en Apple Silicon (PyTorch MPS).

## Objetivo
Construir, entrenar y evaluar un modelo de lenguaje pequeño sin usar pesos preentrenados. El tokenizador se entrenará también desde cero.

## Fases
1. Verificar Python, PyTorch y MPS.
2. Reunir corpus de texto con licencia apropiada y separar entrenamiento/validación.
3. Entrenar tokenizador y definir arquitectura GPT pequeña (~10–15M parámetros).
4. Entrenar con checkpoints, métricas y semilla reproducible.
5. Implementar inferencia autoregresiva.
6. Evaluar pérdida, perplejidad, muestras y velocidad.

## Inicio en macOS
```bash
cd ~/Projects
git clone <URL_DEL_REPOSITORIO> mike-llm
cd mike-llm
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/check_device.py
```

No usar datos privados ni corpus sin licencia. Entrenar desde cero requiere tiempo y un corpus adecuado; la calidad inicial será limitada.
