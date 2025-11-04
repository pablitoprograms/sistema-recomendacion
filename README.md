# Comparación de sistemas de recomendación

Proyecto de ejemplo para comparar un recomendador **colaborativo** y uno **basado en contenido** usando el dataset **MovieLens 100k**.

## Contenido
- `app.py` - App Streamlit interactiva con gráficas Plotly.
- `recommender/` - Implementación de modelos y utilidades.
- `requirements.txt` - Dependencias con versiones exactas.
- `data/` - Se descargará el dataset MovieLens 100k aquí la primera vez que se ejecute la app.
- `results/` - Resultados generados (ej. `comparison_results.csv`).

## Cómo usar (rápido)
1. Crear un entorno virtual (recomendado):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\\Scripts\\activate    # Windows (PowerShell)
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar la app Streamlit:
   ```bash
   streamlit run app.py
   ```

La primera vez la app descargará el dataset MovieLens 100k automáticamente (desde GroupLens).

## Notas
- El recomendador basado en contenido usa TF-IDF (título + géneros).
- El recomendador colaborativo usa SVD de la librería `surprise`.
- Las métricas mostradas incluyen RMSE (para el colaborativo) y Precision@K (aproximada para ambos modelos).
- Si quieres incluir el dataset dentro del repositorio (para uso sin conexión), descarga `ml-100k.zip` desde GroupLens y descomprímelo en `data/ml-100k/`.
