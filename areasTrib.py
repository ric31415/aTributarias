import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Configuración de la app ---
st.set_page_config(page_title="Áreas tributarias")
st.title("Cálculo de áreas tributarias")

# --- Parámetros de la malla ---
st.sidebar.header("Discretización")
dx = st.sidebar.number_input(
    "dx:", min_value=1e-4, max_value=1.0,
    value=0.005, step=0.001, format="%.4f"
)
dy = st.sidebar.number_input(
    "dy:", min_value=1e-4, max_value=1.0,
    value=0.005, step=0.001, format="%.4f"
)

# --- Entrada de Vértices ---
st.sidebar.header("Vértices del Polígono")
input_v = st.sidebar.radio(
    "Método de ingreso (vértices):", ["Manual", "Archivo"], key="input_v"
)
vertices = []
if input_v == "Archivo":
    up_v = st.sidebar.file_uploader(
        "Suba un CSV o Excel (.xlsx) con columnas 'x' y 'y' para vértices",
        type=["csv", "xls", "xlsx"], key="vert_file"
    )
    if up_v:
        # Leer según extensión
        if up_v.name.lower().endswith((".xls", ".xlsx")):
            df_v = pd.read_excel(up_v)
        else:
            df_v = pd.read_csv(up_v)
        if {"x", "y"}.issubset(df_v.columns):
            vertices = list(zip(df_v['x'], df_v['y']))
        else:
            st.sidebar.error("El archivo debe tener columnas 'x' y 'y'.")
else:
    txt_v = st.sidebar.text_area(
        "Vertices 'x,y' separados por espacios:",
        value="0,0 20,0 20,20 0,20"
    )
    try:
        vertices = [tuple(map(float, p.split(','))) for p in txt_v.split()]
    except:
        st.sidebar.error("Formato inválido. Use 'x1,y1 x2,y2 ...'")

# --- Entrada de Columnas ---
st.sidebar.header("Columnas")
input_c = st.sidebar.radio(
    "Método de ingreso (columnas):", ["Manual", "Archivo"], key="input_c"
)
columnas = []
if input_c == "Archivo":
    up_c = st.sidebar.file_uploader(
        "Suba un CSV o Excel (.xlsx) con columnas 'x' y 'y' para columnas",
        type=["csv", "xls", "xlsx"], key="col_file"
    )
    if up_c:
        if up_c.name.lower().endswith((".xls", ".xlsx")):
            df_c = pd.read_excel(up_c)
        else:
            df_c = pd.read_csv(up_c)
        if {"x", "y"}.issubset(df_c.columns):
            columnas = list(zip(df_c['x'], df_c['y']))
        else:
            st.sidebar.error("El archivo debe tener columnas 'x' y 'y'.")
else:
    txt_c = st.sidebar.text_area(
        "Columnas 'x,y' separados por espacios:",
        value=(
            "0,0 0,8 0,14 0,20 "
            "5,0 5,8 5,14 5,20 "
            "12,0 12,8 12,14 12,20 "
            "20,0 20,8 20,14 20,20"
        )
    )
    try:
        columnas = [tuple(map(float, p.split(','))) for p in txt_c.split()]
    except:
        st.sidebar.error("Formato inválido. Use 'x1,y1 x2,y2 ...'")

# --- Botón de cálculo ---
calcular = st.sidebar.button("Calcular")

if calcular:
    if len(vertices) < 3 or len(columnas) < 1:
        st.warning("Ingrese al menos 3 vértices y 1 columna para procesar.")
    else:
        # Polígono cerrado y malla
        poly = Path(vertices + [vertices[0]])
        x_coords, y_coords = zip(*vertices)
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        x = np.arange(xmin, xmax + dx, dx)
        y = np.arange(ymin, ymax + dy, dy)
        X, Y = np.meshgrid(x, y)
        pts = np.vstack([X.ravel(), Y.ravel()]).T

        # Filtrar e indexar
        mask = poly.contains_points(pts, radius=-1e-9)
        dists = cdist(pts[mask], columnas)
        idxs = np.full(pts.shape[0], np.nan)
        idxs_mask = np.argmin(dists, axis=1)
        idxs[mask] = idxs_mask
        Z = idxs.reshape(Y.shape).astype(int)

        # Áreas
        counts = np.bincount(idxs_mask, minlength=len(columnas))
        areas = (counts * dx * dy).round(4)
        total_area = areas.sum().round(4)

        # Tabla
        with st.spinner('Generando tabla de áreas...'):
            st.subheader("Áreas tributarias de columnas")
            df_res = pd.DataFrame({'Columna': np.arange(len(columnas)), 'Área': areas})
            st.dataframe(df_res)
            st.markdown(f"**El área total de discretización es {total_area}**")

        # Gráfico
        with st.spinner('Generando gráfico de áreas...'):
            st.subheader("Mapa de áreas tributarias")
            fig, ax = plt.subplots(figsize=(8, 8))
            cmap = cm.get_cmap('rainbow', len(columnas))
            norm = mcolors.BoundaryNorm(np.arange(-0.5, len(columnas)+0.5, 1), len(columnas))
            pcm = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto', alpha=0.5)
            ax.add_patch(plt.Polygon(vertices, closed=True, fill=False, edgecolor='black', linewidth=1.0))
            for i, (cx, cy) in enumerate(columnas):
                ax.scatter(cx, cy, c='k', marker='s', s=30)
                ax.text(cx, cy + (ymax-ymin)*0.02, f"{i}: {areas[i]:.2f}", ha='center', va='bottom', fontsize=8)
            dxw, dyh = xmax-xmin, ymax-ymin
            ax.set_xlim(xmin-0.1*dxw, xmax+0.1*dxw)
            ax.set_ylim(ymin-0.1*dyh, ymax+0.1*dyh)
            ax.set_aspect('equal'); ax.set_xlabel('X'); ax.set_ylabel('Y')
            cbar = fig.colorbar(pcm, ax=ax, boundaries=np.arange(-0.5, len(columnas)+0.5,1), ticks=[])
            st.pyplot(fig)
            st.snow()
else:
    st.info("Presione 'Calcular' en la barra lateral para ejecutar el análisis.")
