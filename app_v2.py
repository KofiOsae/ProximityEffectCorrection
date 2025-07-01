# --- Colab-compatible full workflow --- Do not touch this
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, root # Import curve_fit and root for numerical solve
from scipy.interpolate import griddata # Keep griddata if needed, though not explicitly used in integrated parts
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
from sklearn.metrics import r2_score, mean_squared_error # Import metrics
from ipywidgets import interact, Dropdown, Checkbox, VBox, FileUpload, Output, FloatText, Button, Label # Import ipywidgets, Output, and FloatText
from IPython.display import display # Import display for widgets
from sklearn.preprocessing import PolynomialFeatures # Import PolynomialFeatures
from sklearn.linear_model import LinearRegression # Import LinearRegression
from sklearn.pipeline import make_pipeline # Import make_pipeline
import ipywidgets as widgets # Import ipywidgets with the alias 'widgets'


# === Load Experimental Data ===
# --- Data Loading from File (Excel or Text) ---
def load_data(uploaded_file):
    """Loads data from an uploaded file (Excel or text)."""
    if not uploaded_file:
        print("No file uploaded.")
        return None

    file_info = list(uploaded_file.values())[0]
    file_name = file_info['metadata']['name']
    content = file_info['content']

    try:
        # Read the content into a BytesIO object
        from io import BytesIO
        file_io = BytesIO(content)

        if file_name.endswith('.xlsx'):
            df = pd.read_excel(file_io)
        elif file_name.endswith('.csv') or file_name.endswith('.txt'):
            # Try different encodings
            try:
                df = pd.read_csv(file_io, encoding='utf-8')
            except UnicodeDecodeError:
                file_io.seek(0) # Reset pointer
                df = pd.read_csv(file_io, encoding='latin-1')
        else:
            print("Error: Unsupported file format. Please provide an Excel (.xlsx) or text (.csv, .txt) file.")
            return None

        # Check if required columns exist and rename 'Centre' to 'CD' for consistency
        required_cols = ['Dose', 'PatternDensity', 'Target CD', 'Centre']
        if not all(required_col in df.columns for required_col in required_cols):
             print(f"Error: Input file must contain the following columns: {required_cols}")
             return None


        df = df.rename(columns={'Centre': 'CD'})

        return df.dropna(subset=['Dose', 'PatternDensity', 'CD']) # Drop rows with NaN in key columns
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def simulate_cd(dose, density, alpha, beta, eta):
    gamma = (1 - eta) / (np.pi * alpha**2) + eta / (np.pi * beta**2)
    effective_dose = dose * (1 + eta * (density / 100))
    cd = 300 / (1 + np.exp(-(effective_dose - 600) / 50)) # Placeholder sigmoid, could be fitted
    return cd

def loss(params, df_fit):
    alpha, beta, eta = params
    if alpha <= 0 or beta <= 0 or not (0 < eta < 1):
        return np.inf
    error = 0
    for i, row in df_fit.iterrows():
        cd_pred = simulate_cd(row['Dose'], row['PatternDensity'], alpha, beta, eta)
        error += (cd_pred - row['CD'])**2
    return error

# === Fit Analytical Model to Experimental Isofocal Trend ===

# Define functions for linear and quadratic models
def linear_model(x, m, c):
    """Linear model: y = mx + c"""
    return m * x + c

def quadratic_model(x, a, b, c):
    """Quadratic model: y = ax^2 + bx + c"""
    return a * x**2 + b * x + c


# --- Main Analysis and Plotting Function ---
def run_analysis(uploaded_file, target_cd_val, show_poly_3d, show_poly_2d, show_isofocal, show_contour, show_psf, isofocal_fit_type):
    """Runs the analysis and generates plots based on user selections."""

    # Make variables available globally for the example usage section and corrected dose calculation
    global params_poly, alpha_fit, beta_fit, eta_fit, params_linear, params_quadratic, experimental_densities, experimental_isofocal_doses, poly_model_3d, CD_target_val # Make CD_target_val global
    # Make the out widget global so the button callback can access it
    global out


    out.clear_output()
    with out:
        df = load_data(uploaded_file)

        if df is None or df.empty:
            print("Failed to load data or data is empty. Analysis cannot proceed.")
            # Set fitted parameters to None if data loading fails
            params_poly = None
            alpha_fit, beta_fit, eta_fit = None, None, None
            params_linear = None
            params_quadratic = None
            experimental_densities = []
            experimental_isofocal_doses = []
            poly_model_3d = None
            CD_target_val = target_cd_val # Still set the global target CD
            # Clear previous plots by creating new empty figures
            plt.close('all')
            return

        print("Experimental Data (first 5 rows):")
        # Display DataFrame with reduced decimal places
        with pd.option_context('display.float_format', '{:.2f}'.format):
            display(df.head())
        print(f"\nTarget CD: {target_cd_val} nm")
        print(f"\nNumber of valid data points loaded: {len(df)}")


        # Update CD_target for use within this function and plots
        CD_target_val = target_cd_val


        # === Fit 2nd Degree Polynomial Model (using sklearn pipeline) ===
        print("\nFitting 2nd Degree Polynomial Model...")
        # Prepare data for sklearn (features X, target y)
        X_sklearn = df[['Dose', 'PatternDensity']].values
        y_sklearn = df['CD'].values

        poly_model_3d = None
        try:
            poly_model_3d = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            poly_model_3d.fit(X_sklearn, y_sklearn)
            # Note: params_poly from curve_fit is no longer used directly
            print("2nd Degree Polynomial Model Fitted (using sklearn).")
            # You can access coefficients if needed via poly_model_3d.named_steps['linearregression'].coef_
        except Exception as e:
            print(f"Error fitting polynomial model (sklearn): {e}")
            poly_model_3d = None # Ensure None on failure


        # === Optimize Double Gaussian PSF Parameters ===
        print("\nOptimizing Double-Gaussian PSF Parameters...")
        alpha_fit, beta_fit, eta_fit = None, None, None
        x0 = [10.0, 1000.0, 0.6]
        bounds = [(1, 50), (100, 3000), (0.01, 0.99)]
        try:
            result = minimize(loss, x0, args=(df,), bounds=bounds) # Pass df to loss function
            alpha_fit, beta_fit, eta_fit = result.x
            print(f"Fitted PSF Parameters:")
            print(f"  Alpha (forward range): {alpha_fit:.2f} nm")
            print(f"  Beta (backscatter range): {beta_fit:.2f} nm")
            print(f"  Eta (energy ratio): {eta_fit:.3f}")
        except Exception as e:
            print(f"Error during Double-Gaussian optimization: {e}")
            alpha_fit, beta_fit, eta_fit = None, None, None # Ensure None on failure


        # === Experimental Isofocal Dose Calculation ===
        print("\nCalculating Experimental Isofocal Doses...")
        grouped_densities = df.groupby('PatternDensity')
        experimental_isofocal_doses = []
        experimental_densities = []
        best_doses_per_density = {}

        for density, group in grouped_densities:
            if not group.empty:
                # Find the index of the CD value closest to CD_target_val
                closest_cd_index = np.abs(group['CD'] - CD_target_val).argmin()
                # Get the corresponding Dose value
                isofocal_dose = group.iloc[closest_cd_index]['Dose']
                experimental_densities.append(density)
                experimental_isofocal_doses.append(isofocal_dose)
                best_doses_per_density[density] = isofocal_dose

        print("Experimental Isofocal Points Calculated.")
        print(f"\nBest Dose for each Pattern Density (to achieve target CD = {CD_target_val} nm):")
        if best_doses_per_density:
            # Sort by density for cleaner output
            for density in sorted(best_doses_per_density.keys()):
                dose = best_doses_per_density[density]
                print(f"  Density {density}%: {dose:.2f} µC/cm²")
        else:
            print("  Could not determine best doses from experimental data.")


        # === Fit Analytical Model to Experimental Isofocal Trend ===
        params_linear = None
        params_quadratic = None
        r2_linear = -np.inf
        r2_quadratic = -np.inf
        mse_linear = np.inf
        mse_quadratic = np.inf

        if experimental_densities and experimental_isofocal_doses:
            experimental_densities_np = np.array(experimental_densities)
            experimental_isofocal_doses_np = np.array(experimental_isofocal_doses)

            print("\nFitting Analytical Models to Experimental Isofocal Trend...")

            # Linear Fit
            try:
                params_linear, covariance_linear = curve_fit(linear_model, experimental_densities_np, experimental_isofocal_doses_np)
                m_fit, c_fit_linear = params_linear
                predicted_isofocal_linear = linear_model(experimental_densities_np, *params_linear)
                r2_linear = r2_score(experimental_isofocal_doses_np, predicted_isofocal_linear)
                mse_linear = mean_squared_error(experimental_isofocal_doses_np, predicted_isofocal_linear)
                print("Linear Model Fitted.")
                print(f"  m: {m_fit:.4f}, c: {c_fit_linear:.4f}")
                print(f"  R-squared: {r2_linear:.4f}, MSE: {mse_linear:.4f}")

            except RuntimeError as e:
                print(f"Error fitting linear model: {e}")
                params_linear = None # Ensure None on failure


            # Quadratic Fit
            try:
                initial_guess_quadratic = [0, m_fit if params_linear is not None else 1, c_fit_linear if params_linear is not None else 500]
                params_quadratic, covariance_quadratic = curve_fit(quadratic_model, experimental_densities_np, experimental_isofocal_doses_np, p0=initial_guess_quadratic)
                a_fit, b_fit, c_fit_quadratic = params_quadratic
                predicted_isofocal_quadratic = quadratic_model(experimental_densities_np, *params_quadratic)
                r2_quadratic = r2_score(experimental_isofocal_doses_np, predicted_isofocal_quadratic)
                mse_quadratic = mean_squared_error(experimental_isofocal_doses_np, predicted_isofocal_quadratic)
                print("\nQuadratic Model Fitted.")
                print(f"  a: {a_fit:.8f}, b: {b_fit:.4f}, c: {c_fit_quadratic:.4f}")
                print(f"  R-squared: {r2_quadratic:.4f}, MSE: {mse_quadratic:.4f}")

            except RuntimeError as e:
                print(f"Error fitting quadratic model: {e}")
                params_quadratic = None # Ensure params_quadratic is None on failure

            print("\n--- Model Comparison ---")
            if r2_linear > r2_quadratic:
                print("Linear model provides a better fit (higher R-squared).")
            elif r2_quadratic > r2_linear:
                print("Quadratic model provides a better fit (higher R-squared).")
            else:
                print("Models have similar R-squared or fitting failed.")

            if mse_linear < mse_quadratic:
                print("Linear model provides a better fit (lower MSE).")
            elif mse_quadratic < mse_linear:
                print("Quadratic model provides a better fit (lower MSE).")
            else:
                print("Models have similar MSE or fitting failed.")
        else:
             print("\nSkipping analytical model fitting due to insufficient experimental isofocal points.")


        # === Calculate and Display Extracted Parameters ===

        print("\n--- Extracted Process Parameters ---")

        # 1. Process Blur (from Double-Gaussian Model) - Already shown above, so skipping explicit print here.
        # Check if fitted parameters are available to confirm fitting was attempted
        if alpha_fit is None or beta_fit is None or eta_fit is None:
            print("\n1. Process Blur: Double-Gaussian PSF parameters not available (fitting failed).")
        else:
             print("\n1. Process Blur: See 'Optimizing Double Gaussian PSF Parameters' section above for fitted values.")


        # 2. Constant Bias (Estimated)
        constant_bias_poly = None
        constant_bias_gauss = None

        # Use the isofocal dose at 0% density from the best isofocal fit
        isofocal_dose_at_0_density = None
        isofocal_fit_model_used = 'None' # Initialize with 'None'
        if params_quadratic is not None:
            isofocal_dose_at_0_density = quadratic_model(0, *params_quadratic)
            isofocal_fit_model_used = 'Quadratic Isofocal Fit'
        elif params_linear is not None:
            isofocal_dose_at_0_density = linear_model(0, *params_linear)
            isofocal_fit_model_used = 'Linear Isofocal Fit'


        print("\n2. Constant Bias (Estimated at 0% Density):")
        if isofocal_dose_at_0_density is not None:
            print(f"   - Using Isofocal Dose at 0% Density from {isofocal_fit_model_used}: {isofocal_dose_at_0_density:.2f} µC/cm²")

            # Estimate using Polynomial Model
            if poly_model_3d is not None:
                try:
                    # Ensure the input to predict is a 2D array
                    predicted_cd_at_0_density_poly = poly_model_3d.predict(np.array([[isofocal_dose_at_0_density, 0]]))[0]
                    constant_bias_poly = predicted_cd_at_0_density_poly - CD_target_val
                    print(f"   - Estimated Bias (using Polynomial CD Prediction Model): {constant_bias_poly:.2f} nm (Predicted CD: {predicted_cd_at_0_density_poly:.2f} nm)")
                except Exception as e:
                    print(f"   - Error estimating constant bias with Polynomial model: {e}")

            # Estimate using Double-Gaussian Model
            if alpha_fit is not None and beta_fit is not None and eta_fit is not None and CD_target_val is not None:
                try:
                    # Use simulate_cd to get the predicted CD at 0% density and the calculated isofocal dose
                    predicted_cd_at_0_density_gauss = simulate_cd(isofocal_dose_at_0_density, 0, alpha_fit, beta_fit, eta_fit)
                    if np.isfinite(predicted_cd_at_0_density_gauss):
                         constant_bias_gauss = predicted_cd_at_0_density_gauss - CD_target_val
                         print(f"   - Estimated Bias (using Double-Gaussian CD Prediction Model): {constant_bias_gauss:.2f} nm (Predicted CD: {predicted_cd_at_0_density_gauss:.2f} nm)")
                    else:
                         print("   - Could not estimate constant bias with Double-Gaussian model: Predicted CD is non-finite.")
                except Exception as e:
                    print(f"   - Error estimating constant bias with Double-Gaussian model: {e}")
        else:
            print("   - Could not estimate constant bias as no analytical isofocal fit was successful to determine the dose at 0% density.")


        # 3. Density Pattern Dependent Lateral Bias (Illustrated via Button)
        print("\n3. Density Pattern Dependent Lateral Bias (Illustrated via Button):")
        print("   - Click the 'Calculate Lateral Bias Illustration' button below to see predicted CD and bias at various densities.")

        # Define the function to perform the density bias illustration
        def illustrate_density_bias(reference_dose, reference_dose_source):
            density_range_illustration = np.linspace(0, 100, 5) # Check CD at 0, 25, 50, 75, 100% density

            print(f"\n   - Illustrating Predicted CD vs Density at a Reference Dose of {reference_dose:.2f} µC/cm² (from {reference_dose_source}):")

            # Illustrate using Polynomial Model
            if poly_model_3d is not None:
                try:
                    X_illustration_poly = np.column_stack([np.full_like(density_range_illustration, reference_dose), density_range_illustration])
                    predicted_cds_poly = poly_model_3d.predict(X_illustration_poly)
                    print("     - Polynomial Model Predictions:")
                    for density, predicted_cd in zip(density_range_illustration, predicted_cds_poly):
                        print(f"       Density {density:.0f}%: Predicted CD = {predicted_cd:.2f} nm (Bias = {predicted_cd - CD_target_val:.2f} nm)")
                except Exception as e:
                    print(f"     - Error illustrating density bias with Polynomial model: {e}")

            # Illustrate using Double-Gaussian Model
            if alpha_fit is not None and beta_fit is not None and eta_fit is not None:
                try:
                    predicted_cds_gauss = [simulate_cd(reference_dose, d, alpha_fit, beta_fit, eta_fit) for d in density_range_illustration]
                    print("     - Double-Gaussian Model Predictions:")
                    for density, predicted_cd in zip(density_range_illustration, predicted_cds_gauss):
                         if np.isfinite(predicted_cd):
                             print(f"       Density {density:.0f}%: Predicted CD = {predicted_cd:.2f} nm (Bias = {predicted_cd - CD_target_val:.2f} nm)")
                         else:
                             print(f"       Density {density:.0f}%: Predicted CD = Non-finite")
                except Exception as e:
                    print(f"     - Error illustrating density bias with Double-Gaussian model: {e}")

        # Check if a reference dose can be determined for the illustration
        illustration_reference_dose = None
        illustration_reference_dose_source = 'None'

        if params_quadratic is not None:
            try:
                 illustration_reference_dose = quadratic_model(50.0, *params_quadratic) # Use 50% density for illustration reference
                 illustration_reference_dose_source = 'Quadratic Isofocal Fit at 50% Density'
            except Exception as e:
                 pass # Ignore error, try next option

        if illustration_reference_dose is None and params_linear is not None:
            try:
                illustration_reference_dose = linear_model(50.0, *params_linear) # Use 50% density for illustration reference
                illustration_reference_dose_source = 'Linear Isofocal Fit at 50% Density'
            except Exception as e:
                pass # Ignore error, try fallback

        if illustration_reference_dose is None and 'experimental_isofocal_doses' in globals() and experimental_isofocal_doses:
             # If no analytical fits, use the average of experimental isofocal doses as a fallback
             illustration_reference_dose = np.mean(experimental_isofocal_doses)
             illustration_reference_dose_source = 'Average Experimental Isofocal Dose'


        # Store the illustration function and parameters in global variables so the button can access them
        # This is necessary because the button callback is outside the run_analysis scope
        global density_bias_illustration_func, density_bias_illustration_params
        if illustration_reference_dose is not None:
            density_bias_illustration_func = illustrate_density_bias
            density_bias_illustration_params = (illustration_reference_dose, illustration_reference_dose_source)
        else:
            density_bias_illustration_func = None
            density_bias_illustration_params = None
            print("   - Could not determine a reference dose to set up the density pattern dependent lateral bias illustration button.")


        # === Plotting ===

        if show_poly_3d and poly_model_3d is not None:
            print("\nGenerating 3D Surface Plot (2nd Degree Polynomial Fit)...")
            dose_plot_poly = np.linspace(df['Dose'].min(), df['Dose'].max(), 100)
            density_plot_poly = np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 100)
            dose_grid_poly, density_grid_poly = np.meshgrid(dose_plot_poly, density_plot_poly)

            # Predict CD values using the fitted sklearn pipeline model
            X_grid_pred = np.column_stack([dose_grid_poly.ravel(), density_grid_poly.ravel()])
            cd_grid_flat_poly = poly_model_3d.predict(X_grid_pred)
            cd_grid_poly = cd_grid_flat_poly.reshape(dose_grid_poly.shape)


            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surface = ax.plot_surface(dose_grid_poly, density_grid_poly, cd_grid_poly, cmap=cm.viridis, antialiased=True, alpha=0.8)
            ax.scatter(df['Dose'], df['PatternDensity'], df['CD'], color='red', label='Measured Data')
            ax.set_xlabel('Dose (µC/cm²)')
            ax.set_ylabel('Pattern Density (%)')
            ax.set_zlabel('Predicted CD (nm)')
            ax.set_title('3D Surface Plot of Predicted CD (2nd Degree Polynomial Fit)')
            fig.colorbar(surface, label='Predicted CD (nm)')
            ax.legend()
            plt.tight_layout()
            plt.show()
        elif show_poly_3d:
            print("\nSkipping 3D Polynomial Surface Plot due to failed polynomial fitting.")


        if show_poly_2d and poly_model_3d is not None: # Use the sklearn model for 2D contour too
            print("\nGenerating 2D Contour Plot (2nd Degree Polynomial Fit)...")
            dose_plot_poly = np.linspace(df['Dose'].min(), df['Dose'].max(), 100)
            density_plot_poly = np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 100)
            dose_grid_poly, density_grid_poly = np.meshgrid(dose_plot_poly, density_plot_poly)

            # Predict CD values using the fitted sklearn pipeline model
            X_grid_pred = np.column_stack([dose_grid_poly.ravel(), density_grid_poly.ravel()])
            cd_grid_flat_poly = poly_model_3d.predict(X_grid_pred)
            cd_grid_poly = cd_grid_flat_poly.reshape(dose_grid_poly.shape)


            plt.figure(figsize=(10, 6))
            contour = plt.contourf(dose_grid_poly, density_grid_poly, cd_grid_poly, levels=20, cmap=cm.viridis)
            plt.colorbar(contour, label='Predicted CD (nm)')
            # Corrected keyword from linewidths to linewidth
            plt.contour(dose_grid_poly, density_grid_poly, cd_grid_poly, levels=[CD_target_val], colors='red', linewidth=2, linestyles='dashed')
            plt.scatter(df['Dose'], df['PatternDensity'], color='red', marker='x', label='Measured Data Points')
            plt.xlabel('Dose (µC/cm²)')
            plt.ylabel('Pattern Density (%)')
            plt.title('2D Contour Plot of Predicted CD (2nd Degree Polynomial Fit)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        elif show_poly_2d:
             print("\nSkipping 2D Polynomial Contour Plot due to failed polynomial fitting.")


    if show_isofocal:
        print("\nGenerating Isofocal Dose vs Pattern Density Plot...")
        plt.figure(figsize=(10,6))

        # Plot the experimental points
        plt.plot(experimental_densities, experimental_isofocal_doses, marker='x', color='red', linestyle='None', label=f'Experimental Isofocal Points (CD={CD_target_val}nm)')

        # Plot the fitted analytical models based on selection
        density_range_plot = np.linspace(0, 100, 100)
        if isofocal_fit_type == 'Linear' and params_linear is not None:
            plt.plot(density_range_plot, linear_model(density_range_plot, *params_linear), color='green', linestyle='--', label='Linear Fit')
        elif isofocal_fit_type == 'Quadratic' and params_quadratic is not None:
                 plt.plot(density_range_plot, quadratic_model(density_range_plot, *params_quadratic), color='purple', linestyle='-.', label='Quadratic Fit')
        elif isofocal_fit_type == 'Both':
                 if params_linear is not None:
                     plt.plot(density_range_plot, linear_model(density_range_plot, *params_linear), color='green', linestyle='--', label='Linear Fit')
                 if params_quadratic is not None:
                     plt.plot(density_range_plot, quadratic_model(density_range_plot, *params_quadratic), color='purple', linestyle='-.', label='Quadratic Fit')


        # Plot the fitted Double-Gaussian model's isofocal curve
        if alpha_fit is not None and beta_fit is not None and eta_fit is not None:
            densities_gauss = np.arange(0, 101, 5)
            isofocal_doses_gauss = []
            for d in densities_gauss:
                # Use root finding to find the dose that gives target CD
                # Define the function to find the root of (simulate_cd - target_cd)
                func_to_solve = lambda dose: simulate_cd(dose, d, alpha_fit, beta_fit, eta_fit) - CD_target_val
                # Use a reasonable initial guess for the dose, e.g., average experimental dose
                initial_dose_guess = np.mean(df['Dose']) if not df.empty else 1000
                try:
                    # Use fsolve from scipy.optimize to find the root
                    sol = root(func_to_solve, initial_dose_guess)
                    if sol.success:
                        isofocal_doses_gauss.append(sol.x[0])
                    else:
                         isofocal_doses_gauss.append(np.nan) # Append NaN if root finding fails
                except Exception as e:
                    print(f"Warning: Root finding failed for density {d}%: {e}")
                    isofocal_doses_gauss.append(np.nan) # Append NaN on error

            # Filter out NaN values before plotting
            valid_indices = ~np.isnan(isofocal_doses_gauss)
            plt.plot(np.array(densities_gauss)[valid_indices], np.array(isofocal_doses_gauss)[valid_indices], marker='o', color='blue', label=f'Fitted Double-Gaussian Model (CD={CD_target_val}nm)')


        plt.title(f'Isofocal Dose vs Pattern Density (Target CD = {CD_target_val} nm)')
        plt.xlabel('Pattern Density (%)')
        plt.ylabel(f'Dose (µC/cm²)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    if show_contour and alpha_fit is not None and beta_fit is not None and eta_fit is not None:
        print("\nGenerating Predicted CD Contour Map from Double-Gaussian Model...")
        dose_grid_gauss, density_grid_gauss = np.meshgrid(np.linspace(df['Dose'].min(), df['Dose'].max(), 50),
                                              np.linspace(df['PatternDensity'].min(), df['PatternDensity'].max(), 50))

        cd_grid_gauss = np.array([[simulate_cd(d, p, alpha_fit, beta_fit, eta_fit)
                            for d in dose_grid_gauss[0, :]] for p in density_grid_gauss[:, 0]])

        plt.figure(figsize=(8,6))
        cp = plt.contourf(dose_grid_gauss, density_grid_gauss, cd_grid_gauss, levels=20, cmap=cm.viridis)
        plt.colorbar(cp, label='Predicted CD (nm)')

        # Corrected keyword from linewidths to linewidth
        plt.contour(dose_grid_gauss, density_grid_gauss, cd_grid_gauss, levels=[CD_target_val], colors='red', linewidth=2, linestyles='dashed')

        if not df.empty:
            plt.scatter(df['Dose'], df['PatternDensity'], c=df['CD'], cmap=cm.viridis, edgecolors='k', s=50, label='Measured Data')

        # Corrected keyword from linewidths to linewidth
        plt.plot([], [], color='red', linewidth=2, linestyle='dashed', label=f'Target CD = {CD_target_val} nm')


        plt.title('Predicted CD Contour Map (Double-Gaussian Model)')
        plt.xlabel('Dose (µC/cm²)')
        plt.ylabel('Pattern Density (%)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif show_contour:
         print("\nSkipping Predicted CD Contour Map plot due to failed Double-Gaussian optimization.")


    if show_psf and alpha_fit is not None and beta_fit is not None:
        print("\nGenerating Normalized Double-Gaussian PSF Plot...")
        def double_gaussian_psf(r, alpha, beta, eta):
            """Double-Gaussian Point Spread Function."""
            term1 = (1 - eta) / (np.pi * alpha**2) + eta / (np.pi * beta**2) * np.exp(-r**2 / beta**2)
            term2 = eta / (np.pi * beta**2) * np.exp(-r**2 / beta**2)
            return term1 + term2

        r_values = np.linspace(0, max(alpha_fit * 5, beta_fit * 2), 200) # Plot out to a reasonable range
        psf_values = double_gaussian_psf(r_values, alpha_fit, beta_fit, eta_fit if eta_fit is not None else 0.5) # Use a default eta if not fitted

        # Normalize the PSF
        psf_values_normalized = psf_values / np.max(psf_values)


        plt.figure(figsize=(8, 5))
        plt.plot(r_values, psf_values_normalized)
        plt.xlabel('Distance from Exposure Point (nm)')
        plt.ylabel('Normalized Energy Density (Arbitrary Units)')
        plt.title('Fitted Double-Gaussian Point Spread Function (PSF)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif show_psf:
        print("\nSkipping Double-Gaussian PSF plot due to failed Double-Gaussian optimization.")


# === Reverse Calculation: Corrected Dose Function ===

# Define calculate_corrected_dose function
def calculate_corrected_dose(pattern_density, model_type='quadratic', linear_params=None, quadratic_params=None, alpha=None, beta=None, eta=None, target_cd=None):
    """
    Calculates the corrected dose required to achieve the target CD
    for a given pattern density using a fitted model.

    Args:
        pattern_density (float): The layout density (%).
        model_type (str): The type of fitted model to use ('linear', 'quadratic', or 'double_gaussian').
                          Defaults to 'quadratic'.
        linear_params (tuple): Fitted parameters for the linear model (m, c).
        quadratic_params (tuple): Fitted parameters for the quadratic model (a, b, c).
        alpha (float): Fitted alpha parameter for the double-Gaussian model.
        beta (float): Fitted beta parameter for the double-Gaussian model.
        eta (float): Fitted eta parameter for the double-Gaussian model.
        target_cd (float): The target CD value for the double-Gaussian model.

    Returns:
        float or None: The calculated corrected dose (µC/cm²) or None if the
                       model type is unrecognized or parameters are missing.
    """
    if model_type.lower() == 'linear':
        if linear_params is not None:
            m_fit, c_fit_linear = linear_params
            return linear_model(pattern_density, m_fit, c_fit_linear)
        else:
            print("Error: Linear model parameters not available for corrected dose calculation.")
            return None
    elif model_type.lower() == 'quadratic':
        if quadratic_params is not None:
            a_fit, b_fit, c_fit_quadratic = quadratic_params
            return quadratic_model(pattern_density, a_fit, b_fit, c_fit_quadratic)
        else:
            print("Error: Quadratic model parameters not available for corrected dose calculation.")
            return None
    elif model_type.lower() == 'double_gaussian':
        if alpha is not None and beta is not None and eta is not None and target_cd is not None:
            # This requires inverting the simulate_cd function, which is non-trivial
            # Use a numerical root-finding method here.
            func_to_solve = lambda dose: simulate_cd(dose, pattern_density, alpha, beta, eta) - target_cd
            # Use a reasonable initial guess for the dose, e.g., 1000
            initial_dose_guess = 1000.0
            try:
                sol = root(func_to_solve, initial_dose_guess)
                if sol.success:
                    return sol.x[0]
                else:
                    print(f"Error: Root finding failed for double-Gaussian model at density {pattern_density}%.")
                    return None
            except Exception as e:
                print(f"Error during root finding for double-Gaussian model: {e}")
                return None
        else:
             print(f"Error: Double-Gaussian model parameters or target CD not available.")
             return None
    else:
        print(f"Error: Unrecognized model_type: {model_type}. Use 'linear', 'quadratic', or 'double_gaussian'.")
        return None

# === Interactive Widgets Setup ===

uploader = FileUpload(accept='.xlsx,.csv,.txt', multiple=False, description='Upload Data File')
target_cd_widget = FloatText(value=200.0, description='Target CD (nm):') # Widget for target CD
plot_poly_3d_checkbox = Checkbox(value=False, description='Show 3D Polynomial Surface Plot') # Default to False initially
plot_poly_2d_checkbox = Checkbox(value=False, description='Show 2D Polynomial Contour Plot') # Default to False initially
plot_isofocal_checkbox = Checkbox(value=True, description='Show Isofocal Dose Plot')
plot_contour_checkbox = Checkbox(value=True, description='Show Double-Gaussian CD Contour Map')
plot_psf_checkbox = Checkbox(value=True, description='Show Double-Gaussian PSF Plot')

isofocal_fit_dropdown = Dropdown(options=['None', 'Linear', 'Quadratic', 'Both'],
                                 value='Quadratic',
                                 description='Isofocal Fit Model:')

# Create an output widget to capture print statements and plots
out = Output()

# Link the uploader value and target_cd_widget value to the analysis function
# We need a helper function to trigger analysis when a file is uploaded or target CD changes
def on_input_change(change):
    run_analysis(uploader.value,
                 target_cd_widget.value, # Pass the target CD value
                 plot_poly_3d_checkbox.value,
                 plot_poly_2d_checkbox.value,
                 plot_isofocal_checkbox.value,
                 plot_contour_checkbox.value,
                 plot_psf_checkbox.value,
                 isofocal_fit_dropdown.value)

uploader.observe(on_input_change, names='value')
target_cd_widget.observe(on_input_change, names='value') # Observe changes in target CD widget
plot_poly_3d_checkbox.observe(on_input_change, names='value')
plot_poly_2d_checkbox.observe(on_input_change, names='value')
plot_isofocal_checkbox.observe(on_input_change, names='value')
plot_contour_checkbox.observe(on_input_change, names='value')
plot_psf_checkbox.observe(on_input_change, names='value')
isofocal_fit_dropdown.observe(on_input_change, names='value')


# Create widgets for interactive corrected dose calculation
pattern_density_input = FloatText(value=50.0, description='Pattern Density (%):')
model_select_dropdown = Dropdown(options=['Quadratic', 'Linear', 'Double-Gaussian'], value='Quadratic', description='Model:') # Added Double-Gaussian option
calculate_dose_button = widgets.Button(description='Calculate Corrected Dose')
corrected_dose_output = Output() # Output for corrected dose calculation result


# Function to handle button click and calculate corrected dose
def on_calculate_dose_button_click(b):
    with corrected_dose_output:
        corrected_dose_output.clear_output()
        density = pattern_density_input.value
        model_type = model_select_dropdown.value.lower()

        # Ensure parameters are available from the last run of run_analysis
        # Access global variables set by run_analysis
        global params_quadratic, params_linear, alpha_fit, beta_fit, eta_fit, CD_target_val

        if 'params_quadratic' not in globals() and 'params_linear' not in globals() and 'alpha_fit' not in globals():
             print("Error: Model parameters not available. Please run the analysis first by uploading a data file or changing the target CD.")
             return
        if 'CD_target_val' not in globals() or CD_target_val is None:
             print("Error: Target CD value not available. Please set the Target CD and run the analysis.")
             return


        if model_type == 'quadratic':
            if 'params_quadratic' in globals() and params_quadratic is not None:
                dose = calculate_corrected_dose(density, model_type='quadratic', quadratic_params=params_quadratic)
                if dose is not None:
                    print(f"Corrected Dose for {density:.2f}% density ({model_select_dropdown.value} Model): {dose:.2f} µC/cm²")
                else:
                     print(f"Could not calculate dose: Quadratic model parameters not available or calculation failed.")
            else:
                print(f"Could not calculate dose: Quadratic model parameters not available.")

        elif model_type == 'linear':
            if 'params_linear' in globals() and params_linear is not None:
                dose = calculate_corrected_dose(density, model_type='linear', linear_params=params_linear)
                if dose is not None:
                    print(f"Corrected Dose for {density:.2f}% density ({model_select_dropdown.value} Model): {dose:.2f} µC/cm²")
                else:
                     print(f"Could not calculate dose: Linear model parameters not available or calculation failed.")
            else:
                 print(f"Could not calculate dose: Linear model parameters not available.")

        elif model_type == 'double-gaussian':
            if 'alpha_fit' in globals() and alpha_fit is not None and 'beta_fit' in globals() and beta_fit is not None and 'eta_fit' in globals() and eta_fit is not None:
                dose = calculate_corrected_dose(density, model_type='double_gaussian', alpha=alpha_fit, beta=beta_fit, eta=eta_fit, target_cd=CD_target_val)
                if dose is not None:
                     print(f"Corrected Dose for {density:.2f}% density ({model_select_dropdown.value} Model): {dose:.2f} µC/cm²")
                else:
                     print(f"Could not calculate dose: Double-Gaussian model parameters or target CD not available, or root finding failed.")
            else:
                 print(f"Could not calculate dose: Double-Gaussian model parameters not available.")

        else:
            # This case handles if the dropdown value is unexpected, though not likely with current options
            print(f"Could not calculate dose: Selected model type ({model_select_dropdown.value}) is not supported or parameters are missing.")


calculate_dose_button.on_click(on_calculate_dose_button_click)


# Create a button and output for the density bias illustration
calculate_bias_illustration_button = widgets.Button(description='Calculate Lateral Bias Illustration')
bias_illustration_output = Output()

def on_calculate_bias_illustration_button_click(b):
    with bias_illustration_output:
        bias_illustration_output.clear_output()
        # Access the illustration function and parameters stored globally by run_analysis
        global density_bias_illustration_func, density_bias_illustration_params, out, CD_target_val

        if density_bias_illustration_func is not None and density_bias_illustration_params is not None and CD_target_val is not None:
             reference_dose, reference_dose_source = density_bias_illustration_params
             # Directly call the illustration function
             density_bias_illustration_func(reference_dose, reference_dose_source)
        else:
             print("Density bias illustration could not be performed. Please ensure data is loaded, models are fitted, and a reference dose could be determined.")


calculate_bias_illustration_button.on_click(on_calculate_bias_illustration_button_click)


# Create a VBox to arrange widgets vertically
widget_layout = VBox([
    uploader,
    target_cd_widget, # Add target CD widget to layout
    plot_poly_3d_checkbox,
    plot_poly_2d_checkbox,
    plot_isofocal_checkbox,
    plot_contour_checkbox,
    plot_psf_checkbox,
    isofocal_fit_dropdown,
    out, # Display the output widget
    # Add corrected dose calculation widgets
    VBox([
        widgets.Label("--- Interactive Corrected Dose Calculation ---"),
        pattern_density_input,
        model_select_dropdown,
        calculate_dose_button,
        corrected_dose_output
    ]),
    # Add density bias illustration widgets
    VBox([
         widgets.Label("--- Density Pattern Dependent Lateral Bias Illustration ---"),
         calculate_bias_illustration_button,
         bias_illustration_output
    ])

])

print("--- Upload Data, Set Target CD, and Select Plots Below ---")
display(widget_layout)

# === Example usage of the corrected dose function ===
# This part can be run separately or remain here.
# It will use the parameters fitted based on the selected target CD.
# print("\n--- Corrected Dose Calculation Example (after running analysis) ---")

# Use densities from the experimental data for the example
# Check if experimental_densities is populated before attempting to use it
# if 'experimental_densities' in globals() and experimental_densities:
#     sample_densities_example = experimental_densities
#     print(f"Target CD: {target_cd_widget.value} nm") # Use the current widget value


    # Note: params_linear and params_quadratic are set within run_analysis.
    # To access them here reliably after interaction, you might need to store them
    # in global variables or pass them around differently. For simplicity in this example,
    # we assume run_analysis has been executed and these variables exist in the global scope
    # from the *last* successful run.

    # for density in sample_densities_example:
    #     corrected_dose_quad = calculate_corrected_dose(density, model_type='quadratic', quadratic_params=params_quadratic if 'params_quadratic' in globals() else None)
    #     if corrected_dose_quad is not None:
    #         print(f"Corrected Dose for {density}% density (Quadratic Model): {corrected_dose_quad:.2f} µC/cm²")
    #     else:
    #          print(f"Could not calculate corrected dose for {density}% density (Quadratic Model): Quadratic parameters not available.")

    #     corrected_dose_linear = calculate_corrected_dose(density, model_type='linear', linear_params=params_linear if 'linear_params' in globals() else None)
    #     if corrected_dose_linear is not None:
    #          print(f"Corrected Dose for {density}% density (Linear Model): {corrected_dose_linear:.2f} µC/cm²")
    #     else:
    #          print(f"Could not calculate corrected dose for {density}% density (Linear Model): Linear parameters not available.")
# else:
#     print("Corrected Dose Calculation Example skipped: Experimental densities not available (data not loaded successfully).")
