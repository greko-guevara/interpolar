# ==========================================
# INTER-POLAR | Streamlit Edition
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

with st.expander("📘 Ayuda teórica — Fundamentos de Interpolación Espacial", expanded=False):

    st.markdown("""
## Introducción a la Interpolación de Datos  
**Prof. Gregory Guevara**  
**Universidad EARTH**  
**Enero 2026**

La **interpolación de datos** es un proceso matemático que permite estimar valores desconocidos
en un dominio espacial o temporal continuo a partir de un conjunto finito de observaciones
discretas.

Formalmente, se busca construir una función:

""")

    st.latex(r"""
    f : \mathbb{R}^n \rightarrow \mathbb{R}
    """)

    st.markdown("""
que represente adecuadamente un fenómeno natural observado en puntos específicos del espacio.

La interpolación es ampliamente utilizada en:
- Hidrología
- Geofísica
- Meteorología
- Ciencias del suelo
- Sistemas de Información Geográfica (SIG)

El objetivo principal es **generar superficies continuas** que preserven la estructura espacial
del fenómeno estudiado.
""")

# =====================================================
    st.markdown("## 1. Interpolación Lineal")

    st.markdown("**Fundamento matemático**")

    st.markdown("Dado un intervalo definido por dos puntos conocidos:")

    st.latex(r"""
    (x_1, y_1), \; (x_2, y_2)
    """)

    st.markdown("la interpolación lineal se define como:")

    st.latex(r"""
    y(x) = y_1 + \frac{y_2 - y_1}{x_2 - x_1}(x - x_1)
    """)

    st.markdown("""
**Ventajas**
- Implementación trivial
- Bajo costo computacional

**Desventajas**
- No representa curvaturas
- Genera superficies angulosas
""")

# =====================================================
    st.markdown("## 2. Vecinos Próximos (Nearest Neighbor)")

    st.markdown("**Fundamento matemático**")

    st.latex(r"""
    \hat{Z}(x) = Z(x_i)
    """)

    st.markdown("donde el punto seleccionado cumple:")

    st.latex(r"""
    x_i = \arg \min_{x_j \in S} \| x - x_j \|
    """)

    st.markdown("""
Este método asigna al punto interpolado el valor del punto observado más cercano.

**Características**
- Superficies discontinuas
- Alta dependencia de la densidad muestral
""")

# =====================================================
    st.markdown("## 3. Inverso de la Distancia Ponderado (IDW)")

    st.markdown("**Fundamento matemático**")

    st.latex(r"""
    \hat{Z}(x) =
    \frac{\sum_{i=1}^{n} Z(x_i)\, d(x,x_i)^{-p}}
    {\sum_{i=1}^{n} d(x,x_i)^{-p}}
    """)

    st.markdown("donde:")

    st.latex(r"""
    d(x,x_i) = \text{distancia euclidiana}, \quad p > 0
    """)

    st.markdown("""
El parámetro \\(p\\) controla la influencia espacial de los puntos vecinos.

**Observaciones**
- Valores altos de \\(p\\) aumentan el peso de puntos cercanos
- No modela estructura espacial global
""")

# =====================================================
    st.markdown("## 4. Funciones de Base Radial (RBF)")

    st.markdown("**Fundamento matemático**")

    st.latex(r"""
    f(x,y) = \sum_{i=1}^{n} \lambda_i
    \, \phi \left( \| (x,y) - (x_i,y_i) \| \right)
    """)

    st.markdown("""
Las RBF generan superficies suaves mediante combinaciones lineales
de funciones radiales centradas en los puntos de muestreo.
""")

    st.markdown("### Funciones base radiales más utilizadas")

    st.latex(r"""
    \begin{aligned}
    \text{Lineal:} \quad & \phi(r) = r \\
    \text{Cúbica:} \quad & \phi(r) = r^3 \\
    \text{Quíntica:} \quad & \phi(r) = r^5 \\
    \text{Gaussiana:} \quad & \phi(r) = e^{-(\varepsilon r)^2} \\
    \text{Multicuadrática:} \quad & \phi(r) = \sqrt{1 + (\varepsilon r)^2}
    \end{aligned}
    """)

# =====================================================
    st.markdown("## 5. Kriging Ordinario")

    st.markdown("**Estimador lineal insesgado**")

    st.latex(r"""
    \hat{Z}(u) = \sum_{i=1}^{n} \lambda_i Z(u_i)
    """)

    st.markdown("""
Los pesos \\(\\lambda_i\\) se obtienen a partir del **variograma experimental**,
garantizando:
- Insesgadez
- Varianza mínima del error
""")

# =====================================================
    st.markdown("## 6. Kriging Universal")

    st.markdown("**Modelo con tendencia**")

    st.latex(r"""
    \hat{Z}(u) = \sum_{i=1}^{n} \lambda_i Z(u_i)
    + \sum_{j=1}^{m} \mu_j f_j(u)
    """)

    st.markdown("""
Este método incorpora una tendencia determinística mediante funciones auxiliares
(polynomial drift).
""")

# =====================================================
    st.markdown("## Comparación conceptual de métodos")

    st.latex(r"""
    \begin{array}{lccc}
    \textbf{Método} & \textbf{Precisión} & \textbf{Suavidad} & \textbf{Costo} \\
    \hline
    \text{Lineal} & \text{Baja} & \text{Baja} & \text{Muy bajo} \\
    \text{Vecinos} & \text{Baja} & \text{Muy baja} & \text{Muy bajo} \\
    \text{IDW} & \text{Media} & \text{Media} & \text{Bajo} \\
    \text{RBF} & \text{Alta} & \text{Alta} & \text{Medio} \\
    \text{Kriging} & \text{Muy alta} & \text{Muy alta} & \text{Alto}
    \end{array}
    """)

    st.info("""
**Conclusión**

No existe un método universalmente superior.  
La selección depende de:
- la estructura espacial de los datos,
- el objetivo del análisis,
- la disponibilidad computacional y estadística.
""")



# ------------------------------------------
# SESSION STATE (FUENTE ÚNICA DE VERDAD)
# ------------------------------------------
if "modelo_variograma" not in st.session_state:
    st.session_state.modelo_variograma = None

if "variogram_params" not in st.session_state:
    st.session_state.variogram_params = None

if "modo_variograma" not in st.session_state:
    st.session_state.modo_variograma = "Automático (RMSE)"

# ------------------------------------------
# SIDEBAR – CARGA DE DATOS
# ------------------------------------------
st.sidebar.header("1️⃣ Archivo de datos")

uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo CSV o Excel, " \
    "**Columnas requeridas = {x,y,z,punto}**",
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
if df is not None and st.checkbox("📈 Ver análisis de variogramas"):

    st.subheader("Análisis comparativo de variogramas")

    modelos = ["spherical", "exponential", "gaussian"]

    fig, axs = plt.subplots(1, len(modelos), figsize=(14, 4))

    for ax, model in zip(axs, modelos):
        V = skg.Variogram(
            coords, values,
            model=model,
            n_lags=6,
            maxlag="median",
            normalize=False,
        )
        ax.scatter(V.bins, V.experimental, color="black")
        h = np.linspace(0, V.bins.max(), 100)
        ax.plot(h, V.fitted_model(h))
        ax.set_title(model.capitalize())
        ax.grid(alpha=0.3)

    st.pyplot(fig)

    # ------------------------------
    # SELECCIÓN AUTOMÁTICA
    # ------------------------------
    rmse = {}
    for model in modelos:
        try:
            V = skg.Variogram(coords, values, model=model, n_lags=6, maxlag="median")
            rmse[model] = V.rmse
        except Exception:
            rmse[model] = np.inf

    rmse_df = pd.DataFrame.from_dict(rmse, orient="index", columns=["RMSE"]).sort_values("RMSE")
    st.dataframe(rmse_df.style.highlight_min(color="#b6f5c9"))

    mejor_modelo = rmse_df.index[0]

    # Guardar automático
    V_auto = skg.Variogram(coords, values, model=mejor_modelo, n_lags=6, maxlag="median")

    st.session_state.modelo_variograma = mejor_modelo
    st.session_state.variogram_params = {
        "range": float(V_auto.parameters[0]),
        "sill": float(V_auto.parameters[1]),
        "nugget": float(V_auto.parameters[2]) if len(V_auto.parameters) > 2 else 0.0,
    }

    st.success(f"Variograma automático óptimo: **{mejor_modelo.capitalize()}**")

    # ------------------------------
    # MODO DE SELECCIÓN
    # ------------------------------
    st.session_state.modo_variograma = st.radio(
        "Modo de selección",
        ["Automático (RMSE)", "Manual"],
        horizontal=True
    )

    if st.session_state.modo_variograma == "Manual":
        modelo_manual = st.selectbox("Modelo de variograma", modelos)

        V_man = skg.Variogram(coords, values, model=modelo_manual, n_lags=6, maxlag="median")

        st.session_state.modelo_variograma = modelo_manual
        st.session_state.variogram_params = {
            "range": float(V_man.parameters[0]),
            "sill": float(V_man.parameters[1]),
            "nugget": float(V_man.parameters[2]) if len(V_man.parameters) > 2 else 0.0,
        }

    # ------------------------------
    # MOSTRAR PARÁMETROS
    # ------------------------------
    vp = st.session_state.variogram_params
    if vp is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelo", st.session_state.modelo_variograma.capitalize())
        c2.metric("Rango", f"{vp['range']:.3f}")
        c3.metric("Sill", f"{vp['sill']:.3f}")

# ------------------------------------------
# SIDEBAR – INTERPOLACIÓN
# ------------------------------------------
st.sidebar.header("2️⃣ Parámetros")

grid_size = st.sidebar.slider("Tamaño de malla", 30, 200, 100)
colormap = st.sidebar.selectbox(
    "Mapa de color",
    ["viridis", "plasma", "inferno", "magma", "cividis"]
)

metodo = st.sidebar.selectbox(
    "Método",
    ["Nearest", "Linear", "IDW", "RBF", "Kriging"]
)


if metodo == "RBF":
    rbf_func = st.sidebar.selectbox(
        "Función RBF",
        ["linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "gaussian"]
    )

if metodo == "Kriging":
    kriging_type = st.sidebar.selectbox("Tipo de Kriging", ["Ordinary", "Universal"])

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
        titulo = "IDW"

    elif metodo == "RBF":
        rbf = Rbf(x, y, z, function=rbf_func)
        zi = rbf(XI, YI)
        titulo = f"RBF ({rbf_func})"

    elif metodo == "Kriging":
        if st.session_state.variogram_params is None:
            st.error("Debe calcular el variograma primero.")
            st.stop()

        if kriging_type == "Ordinary":
            k = OrdinaryKriging(
                x, y, z,
                variogram_model=st.session_state.modelo_variograma,
                variogram_parameters=st.session_state.variogram_params,
            )
        else:
            k = UniversalKriging(
                x, y, z,
                variogram_model=st.session_state.modelo_variograma,
                variogram_parameters=st.session_state.variogram_params,
            )

        zi, _ = k.execute("grid", xi, yi)
        titulo = f"Kriging {kriging_type}"

    zi = np.nan_to_num(zi, nan=np.nanmean(z))

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(XI, YI, zi, 20, cmap=colormap)
    ax.scatter(x, y, c=z, cmap=colormap, edgecolor="black")
    plt.colorbar(c, ax=ax)
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

