import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Prueba de Hipótesis - Proporción", layout="wide")

st.title("Prueba de Hipótesis: Proporción de Aprobados")
st.markdown("Herramienta interactiva para contraste de hipótesis bilateral sobre una proporción poblacional ($p$).")

# Barra lateral con los parámetros del problema
st.sidebar.header("Parámetros del Ejercicio")

p0 = st.sidebar.number_input("Proporción afirmada por la institución (p₀):", min_value=0.01, max_value=0.99, value=0.60, step=0.05)
n = st.sidebar.slider("Tamaño de la muestra (n):", min_value=30, max_value=500, value=200, step=10)
x = st.sidebar.slider("Cantidad de aprobados observados (x):", min_value=0, max_value=n, value=110, step=1)
alpha = st.sidebar.selectbox("Nivel de significancia (α):", [0.01, 0.05, 0.10], index=1)

# Cálculos estadísticos
p_hat = x / n
se = np.sqrt(p0 * (1 - p0) / n)
z_stat = (p_hat - p0) / se
z_critico = stats.norm.ppf(1 - alpha / 2)
p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Columnas de métricas en pantalla
col1, col2, col3, col4 = st.columns(4)
col1.metric("Proporción Muestral (p̂)", f"{p_hat:.2%}")
col2.metric("Estadístico Z Calculado", f"{z_stat:.4f}")
col3.metric("Valores Críticos (Z)", f"±{z_critico:.2f}")
col4.metric("p-valor", f"{p_val:.4f}")

# Criterio de decisión
st.subheader("Criterio y Decisión Estadística")
if abs(z_stat) > z_critico:
    st.error(f"**Rechazar H₀:** Como |Z_calc| = {abs(z_stat):.4f} > {z_critico:.2f} (o p-valor = {p_val:.4f} < {alpha}), existe evidencia suficiente para afirmar que la proporción difiere del {p0:.0%}.")
else:
    st.success(f"**No Rechazar H₀:** Como |Z_calc| = {abs(z_stat):.4f} ≤ {z_critico:.2f} (o p-valor = {p_val:.4f} ≥ {alpha}), no hay evidencia suficiente para afirmar que la proporción real difiere del {p0:.0%}.")

# Gráfico pedagógico de la Campana de Gauss
st.subheader("Visualización de la Campana de Gauss y Zonas de Decisión")

z_vals = np.linspace(-4, 4, 1000)
pdf_vals = stats.norm.pdf(z_vals)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(z_vals, pdf_vals, color="#1f77b4", lw=2, label="Distribución Normal Estándar N(0,1)")

# Sombreado de regiones críticas (rechazo)
ax.fill_between(z_vals, 0, pdf_vals, where=(z_vals <= -z_critico), color="red", alpha=0.35, label=f"Región de Rechazo (α = {alpha})")
ax.fill_between(z_vals, 0, pdf_vals, where=(z_vals >= z_critico), color="red", alpha=0.35)

# Zona de no rechazo
ax.fill_between(z_vals, 0, pdf_vals, where=((z_vals > -z_critico) & (z_vals < z_critico)), color="green", alpha=0.10, label="Región de No Rechazo")

# Líneas críticas y estadístico
ax.axvline(-z_critico, color="darkred", linestyle="--", lw=1.5, label=f"Z crítico = ±{z_critico:.2f}")
ax.axvline(z_critico, color="darkred", linestyle="--", lw=1.5)
ax.axvline(z_stat, color="black", linestyle="-", lw=2.5, label=f"Z muestra = {z_stat:.2f}")

ax.set_ylim(bottom=0)
ax.set_xlabel("Puntaje Z", fontsize=11)
ax.set_ylabel("Densidad de Probabilidad", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", fontsize=9)

st.pyplot(fig)
