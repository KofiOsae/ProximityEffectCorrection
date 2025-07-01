# PEC Streamlit App - v2 with Improvements

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import norm
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

        result = minimize(lambda p: np.sum((simulate_cd(df['Dose'], df['PatternDensity'], *p) - df['CD'])**2), [10, 1000, 0.6], bounds=[(1, 50), (100, 3000), (0.01, 0.99)])
        alpha_fit, beta_fit, eta_fit = result.x

        iso_densities, iso_doses = [], []
        for density, group in df.groupby('PatternDensity'):
            closest = group.iloc[(group['CD'] - target_cd).abs().argmin()]
            iso_densities.append(density)
            iso_doses.append(closest['Dose'])

        iso_densities_np = np.array(iso_densities)
        iso_doses_np = np.array(iso_doses)

        # Fit linear/quadratic models
        lin_params, quad_params, lin_cov, quad_cov = None, None, None, None
        if len(iso_doses_np) > 2:
            lin_params, lin_cov = np.polyfit(iso_densities_np, iso_doses_np, 1, cov=True)
            quad_params, quad_cov = np.polyfit(iso_densities_np, iso_doses_np, 2, cov=True)

        if plot_isofocal:
            fig, ax = plt.subplots()
            ax.scatter(iso_densities, iso_doses, color='red', label="Experimental Isofocal Points")
            x_vals = np.linspace(0, 100, 100)

            if fit_option in ['Linear', 'Both'] and lin_params is not None:
                y_pred = np.polyval(lin_params, x_vals)
                std_err = np.sqrt(np.sum((np.polyval(lin_params, iso_densities_np) - iso_doses_np) ** 2) / (len(iso_doses_np) - 2))
                ci = 1.96 * std_err
                ax.plot(x_vals, y_pred, color='green', label="Linear Fit")
                ax.fill_between(x_vals, y_pred - ci, y_pred + ci, color='green', alpha=0.2, label="95% CI")

            if fit_option in ['Quadratic', 'Both'] and quad_params is not None:
                y_pred = np.polyval(quad_params, x_vals)
                std_err = np.sqrt(np.sum((np.polyval(quad_params, iso_densities_np) - iso_doses_np) ** 2) / (len(iso_doses_np) - 3))
                ci = 1.96 * std_err
                ax.plot(x_vals, y_pred, color='blue', label="Quadratic Fit")
                ax.fill_between(x_vals, y_pred - ci, y_pred + ci, color='blue', alpha=0.2, label="95% CI")

            ax.set_xlabel("Pattern Density (%)")
            ax.set_ylabel("Dose (µC/cm²)")
            ax.set_title(f"Isofocal Dose vs Pattern Density (CD = {target_cd} nm)")
            ax.legend()
            st.pyplot(fig)

            # Constant bias at dose that gives target CD (per model)
            if quad_params is not None:
                dose_q = np.polyval(quad_params, 50)
                cd_q = model.predict([[dose_q, 50]])[0]
                st.info(f"Quadratic Model Bias at CD={target_cd} nm: {cd_q - target_cd:.2f} nm")
            if lin_params is not None:
                dose_l = np.polyval(lin_params, 50)
                cd_l = model.predict([[dose_l, 50]])[0]
                st.info(f"Linear Model Bias at CD={target_cd} nm: {cd_l - target_cd:.2f} nm")

        if plot_cd_stats:
            mean_cd = df['CD'].mean()
            std_cd = df['CD'].std()
            fig, ax = plt.subplots()
            ax.bar(['Measured CD'], [mean_cd], yerr=[std_cd], capsize=5)
            ax.axhline(y=target_cd, color='red', linestyle='--', label=f'Target CD = {target_cd} nm')
            ax.set_title("Mean and Std Deviation of Measured CD")
            ax.legend()
            st.pyplot(fig)

        if plot_3d:
            from mpl_toolkits.mplot3d import Axes3D
            dose_vals = np.linspace(df['Dose'].min(), df['Dose'].max(), 50)
            density_vals = np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 50)
            D, P = np.meshgrid(dose_vals, density_vals)
            Z = model.predict(np.column_stack([D.ravel(), P.ravel()])).reshape(D.shape)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(D, P, Z, cmap='viridis', alpha=0.7)
            ax.set_xlabel("Dose")
            ax.set_ylabel("Pattern Density")
            ax.set_zlabel("CD")
            st.pyplot(fig)

        if plot_2d:
            dose_vals = np.linspace(df['Dose'].min(), df['Dose'].max(), 100)
            density_vals = np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 100)
            D, P = np.meshgrid(dose_vals, density_vals)
            Z = model.predict(np.column_stack([D.ravel(), P.ravel()])).reshape(D.shape)
            fig, ax = plt.subplots(figsize=(8, 6))
            cp = ax.contourf(D, P, Z, levels=20, cmap='viridis')
            plt.colorbar(cp, label='Predicted CD')
            ax.set_title("2D Contour of Polynomial Model")
            st.pyplot(fig)

        if plot_psf:
            r = np.linspace(0, max(alpha_fit * 5, beta_fit * 2), 300)
            psf = ((1 - eta_fit) / (np.pi * alpha_fit ** 2)) * np.exp(-r**2 / alpha_fit**2) + \
                  (eta_fit / (np.pi * beta_fit ** 2)) * np.exp(-r**2 / beta_fit**2)
            psf /= np.max(psf)
            fig, ax = plt.subplots()
            ax.plot(r, psf)
            ax.set_title("Normalized Double-Gaussian PSF")
            ax.set_xlabel("Distance")
            ax.set_ylabel("Normalized Intensity")
            st.pyplot(fig)

        if plot_contour:
            D, P = np.meshgrid(np.linspace(df['Dose'].min(), df['Dose'].max(), 50),
                               np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 50))
            Z = np.array([[simulate_cd(d, p, alpha_fit, beta_fit, eta_fit) for d in D[0]] for p in P[:, 0]])
            fig, ax = plt.subplots()
            cs = ax.contourf(D, P, Z, cmap='viridis')
            plt.colorbar(cs)
            ax.set_title("CD Contour - Double-Gaussian Model")
            st.pyplot(fig)
