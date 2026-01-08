import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, Rbf
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Interpolación Espacial",
    layout="wide"
)

st.title("Interpolación Espacial – Versión Streamlit")

# =========================
# 1. CARGA DE DATOS
# =========================
st.subheader("1. Cargar datos")

uploaded_file = st.file_uploader(
    "Cargue archivo CSV con columnas: X, Y, Z",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Cargue un archivo para continuar")
    st.stop()

df = pd.read_csv(uploaded_file)

if not all(col in df.columns for col in ["X", "Y", "Z"]):
    st.error("El archivo debe contener columnas: X, Y, Z")
    st.stop()

x = df["X"].values
y = df["Y"].values
z = df["Z"].values

st.success(f"{len(df)} puntos cargados correctamente")

# =========================
# 2. MALLA DE INTERPOLACIÓN
# =========================
st.subheader("2. Configuración de malla")

resolucion = st.slider("Resolución de la malla", 30, 300, 100)

xi = np.linspace(x.min(), x.max(), resolucion)
yi = np.linspace(y.min(), y.max(), resolucion)
xi, yi = np.meshgrid(xi, yi)

# =========================
# 3. MÉTODO DE INTERPOLACIÓN
# =========================
st.subheader("3. Ejecutar Interpolación")

metodo = st.radio(
    "Seleccione el método",
    (
        "Geométrica Lineal",
        "Vecinos Próximos",
        "IDW",
        "RBF",
        "Kriging"
    )
)

# ---- Parámetros dinámicos ----
idw_power = None
rbf_func = None
kriging_type = None
kriging_model = None

if metodo == "IDW":
    idw_power = st.selectbox("Power (IDW)", [1, 2, 3, 4], index=1)

if metodo == "RBF":
    rbf_func = st.selectbox(
        "Función RBF",
        [
            "linear",
            "thin_plate_spline",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "gaussian"
        ]
    )

if metodo == "Kriging":
    kriging_type = st.selectbox("Tipo de Kriging", ["Ordinary", "Universal"])
    kriging_model = st.selectbox(
        "Modelo de variograma",
        ["linear", "power", "gaussian", "spherical", "exponential"]
    )

# =========================
# FUNCIÓN CENTRAL (MISMA IDEA QUE TKINTER)
# =========================
def ejecutar_interpolacion(
    metodo,
    idw_power=None,
    rbf_func=None,
    kriging_type=None,
    kriging_model=None
):
    if metodo == "Geométrica Lineal":
        zi = griddata((x, y), z, (xi, yi), method="linear")

    elif metodo == "Vecinos Próximos":
        zi = griddata((x, y), z, (xi, yi), method="nearest")

    elif metodo == "IDW":
        zi = np.zeros_like(xi)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                dist = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
                dist[dist == 0] = 1e-10
                w = 1 / dist**idw_power
                zi[i, j] = np.sum(w * z) / np.sum(w)

    elif metodo == "RBF":
        rbf = Rbf(x, y, z, function=rbf_func)
        zi = rbf(xi, yi)

    elif metodo == "Kriging":
        if kriging_type == "Ordinary":
            OK = OrdinaryKriging(
                x, y, z,
                variogram_model=kriging_model,
                verbose=False,
                enable_plotting=False
            )
            zi, _ = OK.execute("grid", xi[0, :], yi[:, 0])

        else:
            UK = UniversalKriging(
                x, y, z,
                variogram_model=kriging_model,
                drift_terms=["regional_linear"],
                verbose=False,
                enable_plotting=False
            )
            zi, _ = UK.execute("grid", xi[0, :], yi[:, 0])

    else:
        zi = None

    return zi

# =========================
# 4. EJECUTAR
# =========================
if st.button("▶ Ejecutar Interpolación"):
    with st.spinner("Interpolando..."):
        zi = ejecutar_interpolacion(
            metodo,
            idw_power=idw_power,
            rbf_func=rbf_func,
            kriging_type=kriging_type,
            kriging_model=kriging_model
        )

    st.success("Interpolación completada")

    # =========================
    # 5. GRÁFICO
    # =========================
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(xi, yi, zi, levels=20)
    ax.scatter(x, y, c=z, edgecolor="k", s=40)
    ax.set_title(f"Método: {metodo}")
    plt.colorbar(c, ax=ax)

    st.pyplot(fig)
