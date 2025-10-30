# -*- coding: utf-8 -*-
# Clustering Interactivo con K-Means y PCA (con controles para init, max_iter, n_init y random_state)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# -------------------- Configuración de la app --------------------
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("Aprendizaje no supervizado: k-means")
st.subheader("By Carlos Alberto Castro Luna 744849")
st.subheader("cargar datos")

# -------------------- Subir archivo --------------------
st.sidebar.header("📂 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("⚙️ Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        # Parám. básicos
        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)
        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Parámetros avanzados de KMeans")

        # init
        init_method = st.sidebar.selectbox(
            "init (método de inicialización)",
            options=["k-means++", "random"],
            index=0
        )

        # max_iter
        max_iter = st.sidebar.number_input(
            "max_iter (iteraciones máximas)",
            min_value=10, max_value=5000, value=300, step=10
        )

        # n_init (auto o entero)
        n_init_mode = st.sidebar.selectbox(
            "n_init (veces a reiniciar la inicialización)",
            options=["auto", "entero"],
            index=0,
            help="Desde scikit-learn 1.4 se recomienda 'auto'."
        )
        if n_init_mode == "entero":
            n_init_value = st.sidebar.number_input(
                "n_init (entero)",
                min_value=1, max_value=1000, value=10, step=1
            )
            n_init_param = int(n_init_value)
        else:
            n_init_param = "auto"  # si tu versión no soporta 'auto', verás un aviso abajo

        # random_state (None o entero)
        use_rs = st.sidebar.checkbox("Fijar random_state (reproducibilidad)", value=True)
        if use_rs:
            random_state = st.sidebar.number_input(
                "random_state (entero)",
                min_value=0, max_value=10_000, value=0, step=1
            )
            random_state = int(random_state)
        else:
            random_state = None

        # -------------------- Datos y modelo --------------------
        if len(selected_cols) < 2:
            st.error("Selecciona al menos **dos** columnas numéricas para continuar.")
            st.stop()

        X = data[selected_cols].copy()

        # Ajustar n_components si excede el número de características seleccionadas
        max_pca = min(len(selected_cols), 3)
        if n_components > max_pca:
            st.warning(f"PCA en {n_components}D no es posible con {len(selected_cols)} columnas. Se usará {max_pca}D.")
            n_components = max_pca

        # Instanciar y entrenar KMeans con manejo de 'n_init=auto' para versiones antiguas
        def build_kmeans(n_clusters, init, max_iter, n_init, random_state):
            try:
                return KMeans(
                    n_clusters=n_clusters,
                    init=init,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_state
                )
            except TypeError:
                # Fallback si 'auto' no es soportado (versiones < 1.4)
                if isinstance(n_init, str) and n_init.lower() == "auto":
                    st.info("ℹ️ Tu versión de scikit-learn no soporta 'n_init=\"auto\"'. Se usará n_init=10.")
                    return KMeans(
                        n_clusters=n_clusters,
                        init=init,
                        max_iter=max_iter,
                        n_init=10,
                        random_state=random_state
                    )
                raise

        kmeans = build_kmeans(
            n_clusters=k,
            init=init_method,
            max_iter=int(max_iter),
            n_init=n_init_param,
            random_state=random_state
        )
        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        # -------------------- PCA --------------------
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster'].astype(str)

        # -------------------- Visualización antes del clustering --------------------
        st.subheader("📊 Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df,
                x='PCA1', y='PCA2',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                pca_df,
                x='PCA1', y='PCA2', z='PCA3',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # -------------------- Visualización después del clustering --------------------
        st.subheader(f"🎯 Datos agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df,
                x='PCA1', y='PCA2',
                color='Cluster',
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df,
                x='PCA1', y='PCA2', z='PCA3',
                color='Cluster',
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # -------------------- Centroides --------------------
        st.subheader("📍 Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        # -------------------- Parámetros elegidos --------------------
        st.markdown("#### Parámetros del modelo")
        resumen = {
            "init": init_method,
            "max_iter": int(max_iter),
            "n_init": n_init_param,
            "random_state": random_state if random_state is not None else "None",
            "k": k,
            "columnas": selected_cols
        }
        st.json(resumen)

        # -------------------- Método del Codo --------------------
        st.subheader("📈 Método del Codo (Elbow Method)")
        if st.button("Calcular número óptimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = build_kmeans(
                    n_clusters=i,
                    init=init_method,
                    max_iter=int(max_iter),
                    n_init=n_init_param,
                    random_state=random_state
                )
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(list(K), inertias, marker='o')
            ax2.set_title('Método del Codo')
            ax2.set_xlabel('Número de Clusters (k)')
            ax2.set_ylabel('Inercia (SSE)')
            ax2.grid(True)
            st.pyplot(fig2)

        # -------------------- Descarga de resultados --------------------
        st.subheader("💾 Descargar datos con clusters asignados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("👉 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write(
        """
**Ejemplo de formato:**

| Ingreso_Anual | Gasto_Tienda | Edad |
|---------------|--------------|------|
| 45000         | 350          | 28   |
| 72000         | 680          | 35   |
| 28000         | 210          | 22   |
"""
    )
