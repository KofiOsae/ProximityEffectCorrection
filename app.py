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

        fig_col1, fig_col2 = st.columns(2)

        # 3D Plot
        if plot_3d:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            dose_range = np.linspace(df['Dose'].min(), df['Dose'].max(), 50)
            density_range = np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 50)
            D, P = np.meshgrid(dose_range, density_range)
            Z = model.predict(np.column_stack((D.ravel(), P.ravel()))).reshape(D.shape)
            ax.plot_surface(D, P, Z, cmap='viridis', alpha=0.7)
            ax.set_xlabel("Dose")
            ax.set_ylabel("Pattern Density")
            ax.set_zlabel("CD")
            st.pyplot(fig)

        # 2D Contour
        if plot_2d:
            dose_range = np.linspace(df['Dose'].min(), df['Dose'].max(), 100)
            density_range = np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 100)
            D, P = np.meshgrid(dose_range, density_range)
            Z = model.predict(np.column_stack((D.ravel(), P.ravel()))).reshape(D.shape)
            fig, ax = plt.subplots(figsize=(8, 6))
            cp = ax.contourf(D, P, Z, levels=20, cmap='viridis')
            plt.colorbar(cp, label='Predicted CD')
            ax.set_xlabel("Dose")
            ax.set_ylabel("Pattern Density")
            st.pyplot(fig)

        # Isofocal Fit Plot
        if plot_isofocal:
            fig, ax = plt.subplots()
            ax.scatter(iso_densities, iso_doses, color='red', label="Experimental Isofocal Points")

            if fit_option in ['Linear', 'Both']:
                coeffs_lin = np.polyfit(iso_densities, iso_doses, 1)
                ax.plot(iso_densities_np, np.polyval(coeffs_lin, iso_densities_np), label="Linear Fit", color='green')
            if fit_option in ['Quadratic', 'Both']:
                coeffs_quad = np.polyfit(iso_densities, iso_doses, 2)
                ax.plot(iso_densities_np, np.polyval(coeffs_quad, iso_densities_np), label="Quadratic Fit", color='blue')

            ax.set_xlabel("Pattern Density (%)")
            ax.set_ylabel("Dose (µC/cm²)")
            ax.set_title(f"Isofocal Dose vs Pattern Density (CD = {target_cd} nm)")
            ax.legend()
            st.pyplot(fig)

        # Contour from Double-Gaussian
        if plot_contour:
            D, P = np.meshgrid(np.linspace(df['Dose'].min(), df['Dose'].max(), 50),
                               np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 50))
            Z = np.array([[simulate_cd(d, p, alpha_fit, beta_fit, eta_fit) for d in D[0]] for p in P[:, 0]])
            fig, ax = plt.subplots()
            cs = ax.contourf(D, P, Z, cmap='viridis')
            plt.colorbar(cs)
            ax.set_title("Double-Gaussian CD Contour")
            ax.set_xlabel("Dose")
            ax.set_ylabel("Pattern Density")
            st.pyplot(fig)

        # PSF Plot
        if plot_psf:
            r = np.linspace(0, max(alpha_fit * 5, beta_fit * 2), 300)
            psf = ((1 - eta_fit) / (np.pi * alpha_fit ** 2)) * np.exp(-r**2 / alpha_fit**2) + \
                  (eta_fit / (np.pi * beta_fit ** 2)) * np.exp(-r**2 / beta_fit**2)
            psf /= np.max(psf)
            fig, ax = plt.subplots()
            ax.plot(r, psf)
            ax.set_xlabel("Distance (nm)")
            ax.set_ylabel("Normalized PSF")
            ax.set_title("Double-Gaussian PSF")
            st.pyplot(fig)

        # Corrected Dose Calculation
        st.subheader("Corrected Dose Calculator")
        density_input = st.number_input("Pattern Density (%)", value=50.0)
        model_type = st.selectbox("Model", ["Quadratic", "Linear", "Double-Gaussian"])

        dose = None
        if st.button("Calculate"):
            if model_type == "Quadratic":
                coeffs = np.polyfit(iso_densities, iso_doses, 2)
                dose = np.polyval(coeffs, density_input)
            elif model_type == "Linear":
                coeffs = np.polyfit(iso_densities, iso_doses, 1)
                dose = np.polyval(coeffs, density_input)
            elif model_type == "Double-Gaussian":
                func = lambda dose: simulate_cd(dose, density_input, alpha_fit, beta_fit, eta_fit) - target_cd
                sol = root(func, x0=np.mean(df['Dose']))
                if sol.success:
                    dose = sol.x[0]
            if dose is not None:
                st.success(f"Corrected Dose for {density_input}% = {dose:.2f} µC/cm²")
            else:
                st.error("Failed to calculate corrected dose.")
