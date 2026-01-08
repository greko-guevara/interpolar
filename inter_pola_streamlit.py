import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, Rbf
from scipy.stats import kurtosis, skew
import skgstat as skg
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

st.set_page_config(page_title="Inter-Pola", layout="wide")

st.title("🔢 InterPola")
st.subheader("Software académico para el análisis de métodos de interpolación")
st.caption("Versión 1.0")


st.sidebar.header("⚙️ Hiperparámetros")

pix = st.sidebar.number_input(
    "Tamaño de la malla",
    min_value=20,
    max_value=500,
    value=100,
    step=10
)

color = st.sidebar.selectbox(
    "Colormap",
    ["Reds", "Blues", "Greens", "Greys"]
)


st.header("📂 Cargar datos")

uploaded_file = st.file_uploader(
    "Suba archivo CSV o Excel",
    type=["csv", "xls", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Datos cargados correctamente")
    st.dataframe(df)


def calcular_cu(df, media):
    df1 = df.sort_values("z")
    df1["est"] = abs(df1["z"] - media)
    return (1 - df1["est"].sum() / (media * len(df1))) * 100

def calcular_du(df, media):
    df1 = df.sort_values("z")
    return df1.iloc[:len(df1)//4]["z"].mean() / media * 100


if uploaded_file:
    st.header("🧮 Métodos de interpolación")

    metodo = st.selectbox(
        "Seleccione método",
        ["Vecino más cercano", "Lineal", "IDW", "RBF", "Kriging"]
    )

    if st.button("Ejecutar interpolación"):
        x = df["x"].values
        y = df["y"].values
        z = df["z"].values

        xi = np.linspace(x.min(), x.max(), pix)
        yi = np.linspace(y.min(), y.max(), pix)
        XI, YI = np.meshgrid(xi, yi)

        if metodo == "Vecino más cercano":
            ZI = griddata((x, y), z, (XI, YI), method="nearest")

            fig, ax = plt.subplots()
            c = ax.contourf(XI, YI, ZI, cmap=color)
            ax.scatter(x, y, c=z, edgecolor="k")
            fig.colorbar(c)

            st.pyplot(fig)
