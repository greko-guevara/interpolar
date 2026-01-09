# ==========================================
# INTER-POLAR | Streamlit Edition
# Migración desde Tkinter
# Prof. Gregory Guevara
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, Rbf
from scipy.stats import kurtosis, skew
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import skgstat as skg

# ------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------
st.set_page_config(
    page_title="Inter-Polar | Interpolación",
    layout="wide"
)

st.title("🌍 Inter-Polar – Métodos de Interpolación")
st.caption("Versión Streamlit | Geoestadística aplicada")

# ------------------------------------------
# SIDEBAR – CARGA DE DATOS
# ------------------------------------------
st.sidebar.header("1️⃣ Archivo de datos")

uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo CSV o Excel",
    type=["csv", "xls", "xlsx"]
)

df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required_cols = {"x", "y", "z", "punto"}
    if not required_cols.issubset(df.columns):
        st.error("El archivo debe contener las columnas: x, y, z, punto")
        st.stop()

# ------------------------------------------
# ESTADÍSTICAS DESCRIPTIVAS
# ------------------------------------------
if df is not None:
    st.subheader("📊 Estadísticas descriptivas")

    col1, col2, col3 = st.columns(3)

    col1.metric("Puntos", len(df))
    col1.metric("Mínimo Z", f"{df['z'].min():.3f}")
    col1.metric("Máximo Z", f"{df['z'].max():.3f}")

    col2.metric("Media Z", f"{df['z'].mean():.3f}")
    col2.metric("Desv. Std", f"{df['z'].std():.3f}")
    col2.metric("Varianza", f"{df['z'].var():.3f}")

    col3.metric("Curtosis", f"{kurtosis(df['z']):.3f}")
    col3.metric("Asimetría", f"{skew(df['z']):.3f}")

    st.dataframe(df, use_container_width=True)

# ------------------------------------------
# SIDEBAR – PARÁMETROS
# ------------------------------------------
st.sidebar.header("2️⃣ Parámetros generales")

grid_size = st.sidebar.slider("Tamaño de malla", 30, 200, 100)
colormap = st.sidebar.selectbox(
    "Mapa de color",
    ["viridis", "plasma", "inferno", "magma", "cividis"]
)

st.sidebar.header("3️⃣ Método de interpolación")

metodo = st.sidebar.selectbox(
    "Seleccione método",
    ["Nearest", "Linear", "IDW", "RBF", "Kriging"]
)

# --- Hiperparámetros ---
idw_power = None
rbf_func = None
kriging_type = None
kriging_model = None

if metodo == "IDW":
    idw_power = st.sidebar.selectbox("Power (IDW)", [1, 2, 3, 4], index=1)

if metodo == "RBF":
    rbf_func = st.sidebar.selectbox(
        "Función RBF",
        [
            "linear",
            "thin_plate_spline",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "gaussian",
        ]
    )

if metodo == "Kriging":
    kriging_type = st.sidebar.selectbox(
        "Tipo de Kriging", ["Ordinary", "Universal"]
    )
    kriging_model = st.sidebar.selectbox(
        "Modelo de variograma",
        ["linear", "power", "gaussian", "spherical", "exponential"]
    )

# ------------------------------------------
# INTERPOLACIÓN
# ------------------------------------------
if df is not None and st.button("▶ Ejecutar interpolación"):

    x = df["x"].values
    y = df["y"].values
    z = df["z"].values

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    zi = None
    titulo = ""

    try:
        # ---- GRIDDATA ----
        if metodo == "Nearest":
            zi = griddata((x, y), z, (XI, YI), method="nearest")
            titulo = "Vecinos Próximos"

        elif metodo == "Linear":
            zi = griddata((x, y), z, (XI, YI), method="linear")
            titulo = "Interpolación Lineal"

        # ---- IDW (RBF) ----
        elif metodo == "IDW":
            rbf = Rbf(x, y, z, function="inverse_multiquadric")
            zi = rbf(XI, YI)
            titulo = f"IDW (Power={idw_power})"

        # ---- RBF ----
        elif metodo == "RBF":
            rbf = Rbf(x, y, z, function=rbf_func)
            zi = rbf(XI, YI)
            titulo = f"RBF ({rbf_func})"

        # ---- KRIGING ----
        elif metodo == "Kriging":
            if kriging_type == "Ordinary":
                k = OrdinaryKriging(
                    x,
                    y,
                    z,
                    variogram_model=modelo_variograma,
                    variogram_parameters=variogram_params,
                    verbose=False,
                    enable_plotting=False,
    )

            else:
                k = UniversalKriging(
                    x, y, z,
                    variogram_model=kriging_model,
                    verbose=False,
                    enable_plotting=False
                )

            zi, _ = k.execute("grid", xi, yi)
            titulo = f"Kriging {kriging_type} ({kriging_model})"

        # ---- FIX CRÍTICO: NaN ----
        zi = np.nan_to_num(zi, nan=np.nanmean(z))

        # ------------------------------------------
        # VISUALIZACIÓN
        # ------------------------------------------
        st.subheader(f"🗺 Resultado – {titulo}")

        fig, ax = plt.subplots(figsize=(8, 6))

        levels = np.linspace(np.nanmin(zi), np.nanmax(zi), 20)
        c = ax.contourf(XI, YI, zi, levels=levels, cmap=colormap)

        sc = ax.scatter(x, y, c=z, cmap=colormap, edgecolor="black", s=50)

        for _, row in df.iterrows():
            ax.annotate(
                row["punto"],
                (row["x"], row["y"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(titulo)
        ax.grid(alpha=0.4)

        plt.colorbar(c, ax=ax, label="Z interpolado")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error durante la interpolación: {e}")

# ------------------------------------------
# VARIOGRAMAS (VERSIÓN ESTABLE)
# ------------------------------------------
if df is not None and st.checkbox("📈 Ver análisis de variogramas"):

    st.subheader("Análisis comparativo de variogramas")

    coords = df[["x", "y"]].values
    values = df["z"].values

    # Validación mínima
    if len(values) < 6:
        st.warning("Se requieren al menos 6 puntos para calcular variogramas.")
        st.stop()

    models = ["spherical", "exponential", "gaussian"]

    fig, axs = plt.subplots(1, len(models), figsize=(14, 4))
    if len(models) == 1:
        axs = [axs]

    for ax, model in zip(axs, models):
        try:
            V = skg.Variogram(
                coords,
                values,
                model=model,
                n_lags=6,
                normalize=False,
                maxlag="median",
            )

            # Datos experimentales
            ax.scatter(
                V.bins,
                V.experimental,
                color="black",
                label="Experimental",
            )

            # Modelo ajustado
            h = np.linspace(0, V.bins.max(), 100)
            ax.plot(h, V.fitted_model(h), label="Modelo")

            ax.set_title(f"Modelo: {model}")
            ax.set_xlabel("Distancia")
            ax.set_ylabel("Semivarianza")
            ax.legend()
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error\n{model}",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()
    st.pyplot(fig)


st.subheader("Selección de variograma para Kriging")

# ------------------------------------------
# seleccion de variograma
# ------------------------------------------

modelo_variograma = st.selectbox(
    "Modelo de variograma a utilizar",
    ["spherical", "exponential", "gaussian"]
)

V_sel = skg.Variogram(
    coords,
    values,
    model=modelo_variograma,
    n_lags=6,
    normalize=False,
    maxlag="median",
)
st.write("Parámetros del variograma seleccionado")
st.json(variogram_params)
