import streamlit as st
import pandas as pd
import plotly.express as px
from recommender.content_based import ContentRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.utils import download_movielens, load_movielens_ratings, load_movielens_movies, evaluate, precision_at_k

st.set_page_config(layout="wide", page_title="Comparación de Recomendadores")

st.title("Comparación: Recomendador Colaborativo vs Basado en Contenido")

# Ensure dataset present (downloads if necessary)
with st.spinner("Comprobando dataset MovieLens 100k..."):
    data_dir = download_movielens()

ratings = load_movielens_ratings(data_dir)
movies = load_movielens_movies(data_dir)

st.sidebar.header("Configuración")
model_choice = st.sidebar.selectbox("Selecciona modelo", ["Comparar ambos", "Colaborativo", "Contenido"])
k = st.sidebar.slider("K para Precision@K", 1, 20, 10)
run_btn = st.sidebar.button("Ejecutar comparación")

st.markdown("### Muestra del dataset")
st.dataframe(movies.head())

if run_btn:
    st.info("Entrenando y evaluando modelos. Esto puede tardar unos segundos...")
    if model_choice in ("Comparar ambos","Contenido"):
        cb = ContentRecommender(movies, ratings)
        cb.fit()
        cb_metrics, cb_topk = cb.evaluate(k=k)
    else:
        cb_metrics, cb_topk = None, None

    if model_choice in ("Comparar ambos","Colaborativo"):
        col = CollaborativeRecommender(ratings)
        col.fit()
        col_metrics, col_topk = col.evaluate(k=k)
    else:
        col_metrics, col_topk = None, None

    # Mostrar métricas
    rows = []
    if cb_metrics:
        rows.append({"Modelo":"Contenido", **cb_metrics})
    if col_metrics:
        rows.append({"Modelo":"Colaborativo", **col_metrics})
    if rows:
        dfm = pd.DataFrame(rows)
        st.markdown("### Métricas comparativas")
        st.table(dfm.set_index("Modelo"))

        # Guardar resultados
        out_path = "results/comparison_results.csv"
        dfm.to_csv(out_path, index=False)
        st.success(f"Resultados guardados en {out_path}")

        # Gráficas interactivas
        st.markdown("### Gráficas interactivas")
        fig = px.bar(dfm, x="Modelo", y="RMSE", title="RMSE por Modelo", text="RMSE")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(dfm, x="Modelo", y="Precision@"+str(k), title=f"Precision@{k} por Modelo", text="Precision@"+str(k))
        st.plotly_chart(fig2, use_container_width=True)

    # Mostrar recomendaciones de ejemplo (top 10 para un usuario aleatorio)
    user_id = ratings['userId'].sample(1).iloc[0]
    st.markdown(f"### Recomendaciones de ejemplo para el usuario {user_id}")
    cols = st.columns(2)
    if cb_topk is not None:
        cols[0].write("Contenido (top 10)")
        cols[0].write(cb_topk.get(user_id, [])[:10])
    if col_topk is not None:
        cols[1].write("Colaborativo (top 10)")
        cols[1].write(col_topk.get(user_id, [])[:10])
