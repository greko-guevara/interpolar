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

    coords = df[["x", "y"]].values
    values = df["z"].values

# ------------------------------------------
# VARIOGRAMAS
# ------------------------------------------
variogram_params = None
modelo_variograma = None

if df is not None and st.checkbox("📈 Ver análisis de variogramas"):

    st.subheader("Análisis comparativo de variogramas")

    models = ["spherical", "exponential", "gaussian"]

    fig, axs = plt.subplots(1, len(models), figsize=(14, 4))

    for ax, model in zip(axs, models):
        V = skg.Variogram(
            coords,
            values,
            model=model,
            n_lags=6,
            normalize=False,
            maxlag="median",
        )

        ax.scatter(V.bins, V.experimental, color="black", label="Experimental")

        h = np.linspace(0, V.bins.max(), 100)
        ax.plot(h, V.fitted_model(h), label="Modelo")

        ax.set_title(model)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Semivarianza")
        ax.grid(alpha=0.3)
        ax.legend()

    st.pyplot(fig)

    st.subheader("Selección de variograma para Kriging")


    st.subheader("🤖 Selección automática del mejor variograma")

    models = ["spherical", "exponential", "gaussian"]
    rmse_results = {}

    for model in models:
        try:
            V = skg.Variogram(
                coords,
                values,
                model=model,
                n_lags=6,
                normalize=False,
                maxlag="median",
            )

            rmse_results[model] = V.rmse

        except Exception:
            rmse_results[model] = np.inf


    rmse_df = (
        pd.DataFrame.from_dict(rmse_results, orient="index", columns=["RMSE"])
        .sort_values("RMSE")
    )

    st.dataframe(
        rmse_df.style
        .highlight_min(color="#b6f5c9")
        .format("{:.4f}"),
        use_container_width=True
    )

    modelo_variograma = rmse_df.index[0]

    st.success(
        f"✅ Variograma óptimo detectado automáticamente: **{modelo_variograma.capitalize()}**"
    )

    modo_variograma = st.radio(
    "Modo de selección de variograma",
    ["Automático (RMSE)", "Manual"],
    horizontal=True
    )

if modo_variograma == "Manual":
    modelo_variograma = st.selectbox(
        "Modelo de variograma",
        models
    )


    modelo_variograma = st.selectbox(
        "Modelo de variograma a utilizar",
        models
    )

    V_sel = skg.Variogram(
        coords,
        values,
        model=modelo_variograma,
        n_lags=6,
        normalize=False,
        maxlag="median",
    )

    variogram_params = {
        "range": float(V_sel.parameters[0]),
        "sill": float(V_sel.parameters[1]),
        "nugget": float(V_sel.parameters[2]) if len(V_sel.parameters) > 2 else 0.0,
    }

    st.markdown("### 📌 Variograma seleccionado")

    c1, c2, c3 = st.columns(3)

    c1.metric("Modelo", modelo_variograma.capitalize())
    c2.metric("Rango", f"{variogram_params['range']:.3f}")
    c3.metric("Sill", f"{variogram_params['sill']:.3f}")

    st.markdown(
        f"""
        **Interpretación rápida:**
        - 📏 **Rango**: hasta ~{variogram_params['range']:.2f} unidades existe correlación espacial  
        - 📈 **Sill**: varianza estructural ≈ {variogram_params['sill']:.2f}  
        - 🔹 **Nugget**: {'despreciable' if variogram_params['nugget']==0 else variogram_params['nugget']}
        """
    )


# ------------------------------------------
# SIDEBAR – PARÁMETROS DE INTERPOLACIÓN
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

idw_power = None
rbf_func = None
kriging_type = None

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

# ------------------------------------------
# INTERPOLACIÓN
# ------------------------------------------
if df is not None and st.button("▶ Ejecutar interpolación"):

    x, y, z = df["x"].values, df["y"].values, df["z"].values

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    if metodo == "Nearest":
        zi = griddata((x, y), z, (XI, YI), method="nearest")
        titulo = "Vecinos Próximos"

    elif metodo == "Linear":
        zi = griddata((x, y), z, (XI, YI), method="linear")
        titulo = "Interpolación Lineal"

    elif metodo == "IDW":
        rbf = Rbf(x, y, z, function="inverse_multiquadric")
        zi = rbf(XI, YI)
        titulo = f"IDW (Power={idw_power})"

    elif metodo == "RBF":
        rbf = Rbf(x, y, z, function=rbf_func)
        zi = rbf(XI, YI)
        titulo = f"RBF ({rbf_func})"

    elif metodo == "Kriging":
        if variogram_params is None:
            st.error("Debe calcular y seleccionar un variograma primero.")
            st.stop()

        if kriging_type == "Ordinary":
            k = OrdinaryKriging(
                x, y, z,
                variogram_model=modelo_variograma,
                variogram_parameters=variogram_params,
                verbose=False,
                enable_plotting=False,
            )
        else:
            k = UniversalKriging(
                x, y, z,
                variogram_model=modelo_variograma,
                variogram_parameters=variogram_params,
                verbose=False,
                enable_plotting=False,
            )

        zi, _ = k.execute("grid", xi, yi)
        titulo = f"Kriging {kriging_type} ({modelo_variograma})"

    zi = np.nan_to_num(zi, nan=np.nanmean(z))

    st.subheader(f"🗺 Resultado – {titulo}")

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(XI, YI, zi, 20, cmap=colormap)
    ax.scatter(x, y, c=z, cmap=colormap, edgecolor="black", s=50)

    for _, r in df.iterrows():
        ax.annotate(r["punto"], (r["x"], r["y"]), fontsize=8)

    plt.colorbar(c, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)

    st.pyplot(fig)
