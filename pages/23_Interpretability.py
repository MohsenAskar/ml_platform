import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from lime import lime_tabular
from pdpbox import pdp
import plotly.express as px
import plotly.graph_objects as go
import shap.plots as shap_plots
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Model Interpretability", page_icon="🔍", layout="wide")
st.title("Global and Local Interpretation of ML models 🔍")

st.markdown("""
### Model Interpretability Techniques

#### Global Interpretation
Understand how the model works overall:
1. **Feature Importance**: Which features matter most?
2. **Partial Dependence Plots (PDP)**: How features affect predictions on average
3. **SHAP Summary Plots**: Overall feature impact

#### Local Interpretation
Understand individual predictions:
1. **LIME**: Local surrogate models
2. **SHAP Values**: Individual feature contributions
3. **ICE Plots**: How features affect specific instances
4. **Counterfactuals**: "What-if" scenarios
""")

# Sidebar for dataset generation options
st.sidebar.header("Dataset Generation")
dataset_type = st.sidebar.selectbox("Select Dataset Type", ["Linear", "Nonlinear", "Clusters"])
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)

@st.cache_data
def generate_data(dataset_type, n_samples, noise):
    # Define 10 realistic feature names
    feature_names = ['Age', 'Annual_Income', 'Debt_Ratio', 'Credit_Score', 'Loan_Amount',
                     'Years_of_Experience', 'Education_Level', 'Marital_Status', 'Employment_Status', 'Property_Value']
    
    if dataset_type == "Linear":
        X = np.random.randn(n_samples, 10)
        # Use predetermined weights for a linear combination
        weights = np.array([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
        y = X.dot(weights)
        y += noise * np.random.randn(n_samples)
        y = (y > y.mean()).astype(int)
    elif dataset_type == "Nonlinear":
        X = np.random.randn(n_samples, 10)
        y = np.sin(X[:, 0]) + np.square(X[:, 1]) + np.exp(X[:, 2] / 3)
        y += noise * np.random.randn(n_samples)
        y = (y > y.mean()).astype(int)
    else:  # Clusters
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=10,
                          cluster_std=noise, random_state=42)
    return X, y, feature_names

# Generate data based on sidebar selections
X, y, feature_names = generate_data(dataset_type, n_samples, noise_level)

# Display dataset overview (only once)
st.write("### Generated Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Shape: {X.shape}")
with col2:
    st.write(f"Features: {len(feature_names)}")
with col3:
    st.write(f"Class Balance: {np.mean(y):.2f}")

# Create a scatter plot of the first two features
fig = px.scatter(
    x=X[:, 0], y=X[:, 1], color=y.astype(str),
    title="Data Distribution (First Two Features)",
    labels={'x': feature_names[0], 'y': feature_names[1]}
)
st.plotly_chart(fig)

# Model training
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Prepare data: split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = train_model(X_train_scaled, y_train)

# Initialize a SHAP TreeExplainer for later use
explainer = shap.TreeExplainer(model)


col1, col2 = st.columns(2)

with col1:
    global_clicked = st.button("🌍 Global Interpretation")
with col2:
    local_clicked = st.button("🔬 Local Interpretation")

# Update mode based on user selection
if global_clicked:
    interpretation_mode = "Global Interpretation"
elif local_clicked:
    interpretation_mode = "Local Interpretation"
else:
    interpretation_mode = "Global Interpretation"  # Default



# ------------------------- Global Interpretation Section -------------------------
if interpretation_mode == "Global Interpretation":
    st.header("Global Model Interpretation")
    
    # 1. Feature Importance
    st.subheader("1. Feature Importance")
    importance_fig, ax = plt.subplots(figsize=(12, 6))
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    ax.barh(feature_imp['Feature'], feature_imp['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Global Feature Importance')
    st.pyplot(importance_fig)
    
    # 2. SHAP Summary Plot
    st.subheader("2. SHAP Summary Plot")
    shap_values = explainer.shap_values(X_test_scaled)
    with st.spinner("Generating SHAP summary plot..."):
        shap_fig = plt.figure(figsize=(12, 8))
        # For classification, we choose the shap values for the positive class (index 1)
        shap.summary_plot(shap_values[1], X_test_scaled, feature_names=feature_names, show=False)
        st.pyplot(shap_fig)
    
    # 3. Partial Dependence Plots (PDP)
    st.subheader("3. Partial Dependence Plots")
    selected_feature = st.selectbox("Select feature for PDP", feature_names, key="pdp_feature")
    with st.spinner("Generating PDP..."):
        from sklearn.inspection import partial_dependence
        feature_idx = feature_names.index(selected_feature)
        pdp_result = partial_dependence(model, X_train_scaled, [feature_idx], kind='average')
        
        pdp_fig, ax = plt.subplots(figsize=(10, 6))
        # Use the dictionary keys to access the results
        ax.plot(pdp_result['values'][0], pdp_result['average'][0])
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Partial dependence')
        ax.set_title(f'Partial Dependence Plot for {selected_feature}')
        # Add a rug plot for the feature values
        ax.plot(X_train_scaled[:, feature_idx], 
                np.zeros_like(X_train_scaled[:, feature_idx]) - 0.1,
                '|', color='red', alpha=0.2)
        st.pyplot(pdp_fig)

# ------------------------- Local Interpretation Section -------------------------
elif interpretation_mode == "Local Interpretation":
    st.header("Local Model Interpretation")
    
    # Instance selection for local interpretation
    st.subheader("Select an instance to interpret")
    instance_idx = st.slider("Select instance index", 0, len(X_test)-1, 0, key="instance_idx")
    
    # 1. LIME Explanation
    st.subheader("1. LIME Explanation")
    explainer_lime = lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )
    with st.spinner("Generating LIME explanation..."):
        exp = explainer_lime.explain_instance(
            X_test_scaled[instance_idx],
            model.predict_proba,
            num_features=10
        )
        # Call as_pyplot_figure without passing an 'ax' parameter
        lime_fig = exp.as_pyplot_figure()
        st.pyplot(lime_fig)
    


    # 2. SHAP Local Explanation
    st.subheader("2. SHAP Local Explanation")
    with st.spinner("Generating local SHAP values..."):
        # Get SHAP values for the instance
        instance_for_explanation = X_test_scaled[instance_idx:instance_idx+1,:]
        shap_values_local = explainer.shap_values(instance_for_explanation)
        
        # Determine which values to use based on whether it's binary classification
        if isinstance(shap_values_local, list):
            shap_values_to_plot = shap_values_local[1][0]  # Get values for positive class
            expected_value = explainer.expected_value[1]
            shap_values_force = shap_values_local[1]  # Keep matrix format for force plot
        else:
            shap_values_to_plot = shap_values_local[0]
            expected_value = explainer.expected_value
            shap_values_force = shap_values_local
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Force Plot", "Waterfall Plot"])
        
        with tab1:
            st.write("### SHAP Force Plot")
            
            # Create and display force plot using JavaScript plot
            force_plot = shap.force_plot(
                expected_value, 
                shap_values_force,
                instance_for_explanation,
                feature_names=feature_names,
                matplotlib=False,
                show=False
            )
            
            # Get the HTML string and enable notebook mode for proper rendering
            shap.initjs()
            # Save the plot to HTML string
            html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            # Display using streamlit components
            components.html(html, height=150)
            
            # Add a note about the visualization
            st.caption("The force plot shows how each feature pushes the prediction from the base value (average model output) to the final prediction.")
        
        with tab2:
            st.write("### SHAP Waterfall Plot")
            # Waterfall plot
            waterfall_fig = plt.figure(figsize=(10, 6))
            plt.clf()
            shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_values_to_plot,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            st.pyplot(waterfall_fig)
        
        # Display feature contributions as a table
        contributions = pd.DataFrame({
            'Feature': feature_names,
            'Value': instance_for_explanation[0],
            'SHAP Impact': shap_values_to_plot
        })
        
        # Sort by absolute SHAP impact
        contributions['Abs_Impact'] = abs(contributions['SHAP Impact'])
        contributions = contributions.sort_values('Abs_Impact', ascending=False)
        contributions = contributions.drop('Abs_Impact', axis=1)
        
        st.write("### Feature Contributions")
        st.dataframe(contributions.style.format({
            'Value': '{:.3f}',
            'SHAP Impact': '{:.3f}'
        }))
        
        # Add a bar chart of SHAP values
        st.write("### SHAP Values Visualization")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=contributions['Feature'],
            y=contributions['SHAP Impact'],
            marker_color=['red' if x < 0 else 'blue' for x in contributions['SHAP Impact']]
        ))
        fig.update_layout(
            title='Feature Impact on Prediction',
            xaxis_title='Features',
            yaxis_title='SHAP Impact',
            xaxis_tickangle=45,
            height=400
        )
        st.plotly_chart(fig)   
        
    # 3. ICE Plot
    st.subheader("3. ICE Plot")
    selected_feature_ice = st.selectbox("Select feature for ICE plot", feature_names, key='ice_feature')
    with st.spinner("Generating ICE plot..."):
        ice_fig, ax = plt.subplots(figsize=(10, 6))
        feature_idx = feature_names.index(selected_feature_ice)
        x_range = np.linspace(X_train_scaled[:, feature_idx].min(),
                              X_train_scaled[:, feature_idx].max(),
                              num=50)
        ice_predictions = []
        instance = X_test_scaled[instance_idx].copy()
        for x in x_range:
            temp_instance = instance.copy()
            temp_instance[feature_idx] = x
            pred = model.predict_proba([temp_instance])[0][1]
            ice_predictions.append(pred)
        ax.plot(x_range, ice_predictions)
        ax.set_xlabel(selected_feature_ice)
        ax.set_ylabel('Predicted Probability')
        ax.set_title(f'ICE Plot for {selected_feature_ice}')
        ax.axvline(instance[feature_idx], color='red', linestyle='--', label='Current Value')
        ax.legend()
        st.pyplot(ice_fig)
    
    # 4. Counterfactual Explanation
    st.subheader("4. Counterfactual Exploration")
    st.write("""
    Explore how changing feature values would affect the prediction.
    Adjust the sliders below to modify feature values.
    """)
    cf_col1, cf_col2 = st.columns(2)
    with cf_col1:
        st.write("### Modify Features")
        modified_instance = X_test_scaled[instance_idx].copy()
        modified_values = {}
        for feature in feature_names:
            feature_idx = feature_names.index(feature)
            current_value = modified_instance[feature_idx]
            modified_value = st.slider(
                f"{feature}",
                float(X_train_scaled[:, feature_idx].min()),
                float(X_train_scaled[:, feature_idx].max()),
                float(current_value),
                key=f"cf_{feature}"
            )
            modified_instance[feature_idx] = modified_value
            modified_values[feature] = modified_value
    with cf_col2:
        st.write("### Prediction Impact")
        original_pred = model.predict_proba([X_test_scaled[instance_idx]])[0][1]
        modified_pred = model.predict_proba([modified_instance])[0][1]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Original', 'Modified'],
            y=[original_pred, modified_pred],
            text=[f'{original_pred:.3f}', f'{modified_pred:.3f}'],
            textposition='auto',
        ))
        fig.update_layout(
            title='Prediction Probability Comparison',
            yaxis_title='Probability of Positive Class',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig)
        pred_change = modified_pred - original_pred
        st.write(f"Prediction Change: {pred_change:.3f}")

# ------------------------- Educational Notes -------------------------
st.markdown("""
### Understanding the Interpretations

#### Global Interpretation Methods:
1. **Feature Importance**
   - Shows overall impact of each feature.
2. **SHAP Summary Plot**
   - Displays feature impacts across all predictions.
3. **Partial Dependence Plots**
   - Illustrates the average effect of a feature on predictions.

#### Local Interpretation Methods:
1. **LIME**
   - Provides a local surrogate model for interpretation.
2. **SHAP Local**
   - Breaks down how each feature contributes to the prediction.
3. **ICE Plots**
   - Shows prediction changes with feature modifications.
4. **Counterfactuals**
   - "What-if" scenarios to explore changes in predictions.
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "Which of the following best describes the purpose of global model interpretation techniques?",
    (
        "To understand the overall influence of features on model predictions across the entire dataset.",
        "To explain the prediction for a single instance in detail.",
        "To optimize the model's hyperparameters automatically."
    ),
    key="quiz_q1"
)

if quiz_answer_1 == "To understand the overall influence of features on model predictions across the entire dataset.":
    st.success("Correct! Global interpretation techniques provide insights into how each feature contributes to the overall predictions of the model across the dataset.")
else:
    st.error("Not quite. Global interpretation methods help us see the big picture of feature impact on predictions, not just individual cases or hyperparameter optimization.")

# Question 2
quiz_answer_2 = st.radio(
    "Which method is primarily used for local model interpretation by creating a surrogate model around an individual prediction?",
    (
        "LIME Explanation",
        "Partial Dependence Plot (PDP)",
        "Feature Importance Plot"
    ),
    key="quiz_q2"
)

if quiz_answer_2 == "LIME Explanation":
    st.success("Correct! LIME creates a local surrogate model to explain individual predictions, highlighting how each feature contributes to the result.")
else:
    st.error("Not quite. LIME is specifically designed for local interpretability, while PDP and feature importance are generally used for global interpretation.")
