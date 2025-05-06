import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import uuid
import shutil

# Page configuration
st.set_page_config(
    page_title="Soil Bearing Capacity Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        .main {
            background-color: #111111;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6, label, .stTextInput, .stNumberInput {
            color: #EAEAEA !important;
        }
        .stButton>button {
            background-color: #5cdb95;
            color: black;
            font-weight: bold;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stNumberInput>div>div>input {
            background-color: #262730;
            color: #EAEAEA;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("logo.jpg", use_container_width=True)
    st.markdown("<h2 style='text-align: center; color:#ff6347;'>Made by Vaibhav Singh</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🌍 Geotechnical ML App**")
    st.markdown("Upload your dataset and predict soil bearing capacity with advanced ensemble models.")
    

# Main Title
st.title(":earth_asia: Soil Bearing Capacity Predictor")
st.markdown("""
    <div style='font-size: 18px; color: #ff0000;'>
    Harness the power of <strong>Machine Learning</strong> to predict soil bearing capacity based on geotechnical parameters.
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"], help="Upload a CSV file with geotechnical parameters.")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    expected_features = ['Width', 'Depth', 'Friction Angle', 'Unit Weight', 'L/B', 'Bitumen Emulsion Content (%)']
    if list(data.columns[:-1]) != expected_features:
        st.warning(f"Expected features: {expected_features}. Got: {list(data.columns[:-1])}. Please check your dataset.")
    st.success("✅ Dataset uploaded successfully!", icon="🎉")
    st.write(f"Dataset features: {list(data.columns[:-1])}")
    st.write(f"Number of features: {data.shape[1] - 1}")

    # Dataset preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True, height=250)

    # Visual insights
    st.subheader("📈 Data Insights")
    with st.expander("Visualize Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(data, x=data.columns[-1], title="Bearing Capacity Distribution", color_discrete_sequence=['#4CAF50'])
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            corr = data.corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale='Greens')
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # Prepare features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Data splitting
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    st.subheader("🧠 Dataset Split")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", X_train.shape[0])
    col2.metric("Validation Samples", X_val.shape[0])
    col3.metric("Test Samples", X_test.shape[0])

    # Hyperparameter tuning section
    st.subheader("⚙️ Hyperparameter Settings")
    with st.expander("Customize Parameters", expanded=False):
        n_estimators_rf = st.multiselect("Random Forest - n_estimators", [100, 200, 300], default=[200])
        max_depth_rf = st.multiselect("Random Forest - max_depth", [5, 10, 20], default=[10])
        min_samples_split_rf = st.multiselect("Random Forest - min_samples_split", [2, 5, 10], default=[5])

        n_estimators_gb = st.multiselect("Gradient Boosting - n_estimators", [100, 200, 300], default=[200])
        learning_rate_gb = st.multiselect("Gradient Boosting - learning_rate", [0.01, 0.1, 0.2], default=[0.1])
        max_depth_gb = st.multiselect("Gradient Boosting - max_depth", [3, 5, 10], default=[5])

    # Model training
    @st.cache_resource
    def load_or_train_model(_X_train, _y_train, n_rf, d_rf, mss_rf, n_gb, lr_gb, d_gb, cache_key=str(uuid.uuid4())):
        model_path = "models/improved_soil_bearing_model.pkl"
        # Clear existing model and scalers to ensure fresh training
        if os.path.exists("models"):
            shutil.rmtree("models")

        rf = GridSearchCV(
            RandomForestRegressor(random_state=42),
            {'n_estimators': n_rf, 'max_depth': d_rf, 'min_samples_split': mss_rf},
            cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        rf.fit(_X_train, _y_train)
        rf_best = rf.best_estimator_
        st.write(f"Random Forest Best Params: {rf.best_params_}")

        gb = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            {'n_estimators': n_gb, 'learning_rate': lr_gb, 'max_depth': d_gb},
            cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        gb.fit(_X_train, _y_train)
        gb_best = gb.best_estimator_
        st.write(f"Gradient Boosting Best Params: {gb.best_params_}")

        ensemble = VotingRegressor([('rf', rf_best), ('gb', gb_best)], weights=[0.3, 0.7])
        ensemble.fit(_X_train, _y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(ensemble, model_path)
        joblib.dump(scaler_X, "models/scaler_X.pkl")
        joblib.dump(scaler_y, "models/scaler_y.pkl")

        return ensemble, scaler_X, scaler_y, rf_best, gb_best

    st.subheader("⚙️ Model Training")
    if st.button("🚀 Train Model"):
        with st.spinner("Training ensemble model..."):
            model, scaler_X, scaler_y, rf_best, gb_best = load_or_train_model(
                X_train, y_train,
                n_estimators_rf, max_depth_rf, min_samples_split_rf,
                n_estimators_gb, learning_rate_gb, max_depth_gb
            )
            st.success("🎉 Model trained and saved!")

            def evaluate(model, X, y, label, model_name=None):
                pred_scaled = model.predict(X)
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                actual = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
                mse = mean_squared_error(actual, pred)
                mae = mean_absolute_error(actual, pred)
                r2 = r2_score(actual, pred)
                # Debugging output
                st.write(f"{label} - Sample Predictions: {pred[:5]}")
                st.write(f"{label} - Sample Actual: {actual[:5]}")
                st.markdown(f"### 📌 {label} Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.2f}")
                col2.metric("MAE", f"{mae:.2f}")
                col3.metric("R²", f"{r2:.4f}")
                return pred, actual

            # Evaluate models and collect predictions
            pred_rf, actual_test = evaluate(rf_best, X_test, y_test, "Random Forest on Test", "Random Forest")
            pred_gb, _ = evaluate(gb_best, X_test, y_test, "Gradient Boosting on Test", "Gradient Boosting")
            pred_ensemble, _ = evaluate(model, X_test, y_test, "Ensemble on Test", "Ensemble")

            # Plot comparison of predictions vs. actual values
            st.subheader("📉 Model Predictions vs. Actual Values (Test Set)")
            fig = go.Figure()
            # Scatter plots for each model
            fig.add_trace(go.Scatter(
                x=actual_test, y=pred_rf, mode='markers', name='Random Forest',
                marker=dict(color='#4CAF50', size=8), showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=actual_test, y=pred_gb, mode='markers', name='Gradient Boosting',
                marker=dict(color='#FF9800', size=8), showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=actual_test, y=pred_ensemble, mode='markers', name='Ensemble',
                marker=dict(color='#2196F3', size=8), showlegend=True
            ))
            # Diagonal line (y=x)
            min_val = min(actual_test.min(), pred_rf.min(), pred_gb.min(), pred_ensemble.min())
            max_val = max(actual_test.max(), pred_rf.max(), pred_gb.max(), pred_ensemble.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                name='Perfect Prediction', line=dict(color='gray', dash='dash')
            ))
            fig.update_layout(
                title="Predicted vs. Actual Bearing Capacity (kN/m²)",
                xaxis_title="Actual Bearing Capacity (kN/m²)",
                yaxis_title="Predicted Bearing Capacity (kN/m²)",
                showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
                width=800, height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # Evaluate Ensemble on Validation
            evaluate(model, X_val, y_val, "Ensemble on Validation", "Ensemble")

    # Prediction Section
    st.divider()
    st.header("📍 Predict Soil Bearing Capacity")
    st.markdown("**Enter Geotechnical Parameters below:**")

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input("Width (m)", min_value=0.0, value=1.0, step=0.1)
            depth = st.number_input("Depth (m)", min_value=0.0, value=1.0, step=0.1)
            friction_angle = st.number_input("Friction Angle (°)", min_value=0.0, value=30.0, step=0.1)
        with col2:
            unit_weight = st.number_input("Unit Weight (kN/m³)", min_value=0.0, value=18.0, step=0.1)
            L_B_ratio = st.number_input("L/B Ratio", min_value=0.0, value=1.0, step=0.1)
            bitumen_emulsion_content = st.number_input("Bitumen Emulsion Content (%)", min_value=0.0, value=0.0, step=0.1)
                
        submit = st.form_submit_button("🔍 Predict")

    if submit:
        try:
            if not os.path.exists("models/improved_soil_bearing_model.pkl"):
                st.error("⚠️ No trained model found. Please train the model first.")
            else:
                model = joblib.load("models/improved_soil_bearing_model.pkl")
                scaler_X = joblib.load("models/scaler_X.pkl")
                scaler_y = joblib.load("models/scaler_y.pkl")
                input_data = np.array([[width, depth, friction_angle, unit_weight,  L_B_ratio, bitumen_emulsion_content]])
                input_scaled = scaler_X.transform(input_data)
                prediction = model.predict(input_scaled)
                result = scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]
                st.success(f"🧮 Predicted Soil Bearing Capacity: **{result:.2f} kN/m²**")
        except ValueError as ve:
            st.error(f"Input Error: {ve}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.warning("📌 Please upload a dataset to continue.")