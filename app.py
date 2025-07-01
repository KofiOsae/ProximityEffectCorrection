import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import io

st.set_page_config(layout="wide")

# === Load Data ===
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        df = df.rename(columns={'Centre': 'CD'})
        required_cols = ['Dose', 'PatternDensity', 'Target CD', 'CD']
        if not all(col in df.columns for col in required_cols):
            st.error(f"File must contain columns: {required_cols}")
            return None
        return df.dropna(subset=['Dose', 'PatternDensity', 'CD'])
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# === Models ===
def simulate_cd(dose, density, alpha, beta, eta):
    effective_dose = dose * (1 + eta * (density / 100))
    return 300 / (1 + np.exp(-(effective_dose - 600) / 50))

def loss(params, df):
    alpha, beta, eta = params
    if alpha <= 0 or beta <= 0 or not (0 < eta < 1): return np.inf
    preds = [simulate_cd(row['Dose'], row['PatternDensity'], alpha, beta, eta) for _, row in df.iterrows()]
    return np.sum((df['CD'] - preds) ** 2)

def linear_model(x, m, c): return m * x + c
def quadratic_model(x, a, b, c): return a * x**2 + b * x + c

# === App ===
st.title("PEC Isofocal Dose Analyzer")

uploaded_file = st.file_uploader("Upload Experimental Data (.csv, .xlsx)", type=["csv", "xlsx"])
target_cd = st.number_input("Target CD (nm)", value=200.0)

plot_3d = st.checkbox("Show 3D Polynomial Surface Plot")
plot_2d = st.checkbox("Show 2D Polynomial Contour Plot")
plot_isofocal = st.checkbox("Show Isofocal Dose Plot", value=True)
plot_contour = st.checkbox("Show Double-Gaussian Contour", value=True)
plot_psf = st.checkbox("Show Double-Gaussian PSF", value=True)
plot_cd_stats = st.checkbox("Show Mean/Std Plot of CD")

fit_option = st.selectbox("Isofocal Fit Type", ['None', 'Linear', 'Quadratic', 'Both'])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())

        X = df[['Dose', 'PatternDensity']].values
        y = df['CD'].values
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X, y)

        result = minimize(loss, [10, 1000, 0.6], args=(df,), bounds=[(1, 50), (100, 3000), (0.01, 0.99)])
        alpha_fit, beta_fit, eta_fit = result.x

        # Isofocal Experimental Points
        iso_densities, iso_doses = [], []
        for density, group in df.groupby('PatternDensity'):
            closest = group.iloc[(group['CD'] - target_cd).abs().argmin()]
            iso_densities.append(density)
            iso_doses.append(closest['Dose'])

        iso_densities_np = np.array(iso_densities)
        iso_doses_np = np.array(iso_doses)

        # Isofocal Plot
        if plot_isofocal:
            fig, ax = plt.subplots()
            ax.scatter(iso_densities, iso_doses, color='red', label="Experimental Isofocal Points")
            ax.set_xlabel("Pattern Density (%)")
            ax.set_ylabel("Dose (µC/cm²)")
            ax.set_title(f"Isofocal Dose vs Pattern Density (CD = {target_cd} nm)")

            if fit_option in ['Linear', 'Both']:
                coeffs_lin = np.polyfit(iso_densities, iso_doses, 1)
                ax.plot(iso_densities_np, np.polyval(coeffs_lin, iso_densities_np), label="Linear Fit", color='green')
            if fit_option in ['Quadratic', 'Both']:
                coeffs_quad = np.polyfit(iso_densities, iso_doses, 2)
                ax.plot(iso_densities_np, np.polyval(coeffs_quad, iso_densities_np), label="Quadratic Fit", color='blue')

            ax.legend()
            st.pyplot(fig)

        # CD Stats
        if plot_cd_stats:
            mean_cd = df.groupby("PatternDensity")["CD"].mean()
            std_cd = df.groupby("PatternDensity")["CD"].std()
            fig, ax = plt.subplots()
            ax.errorbar(mean_cd.index, mean_cd.values, yerr=std_cd.values, fmt='o', label="CD (mean ± std)", capsize=5)
            ax.axhline(target_cd, color='red', linestyle='--', label=f"Target CD = {target_cd} nm")
            ax.set_xlabel("Pattern Density (%)")
            ax.set_ylabel("CD (nm)")
            ax.set_title("Corner CD Distribution with Target CD")
            ax.legend()
            st.pyplot(fig)

        # More plots and logic as needed...
