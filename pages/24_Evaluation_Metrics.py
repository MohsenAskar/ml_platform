import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, mean_squared_error,
                             r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Model Evaluation Metrics ", page_icon="⚖️", layout="wide")
st.title("Model Evaluation Metrics ⚖️")

# Custom CSS for enhanced styling and layout
st.markdown("""
    <style>
        /* General layout */
        .header {text-align: center; margin-bottom: 20px;}
        .plot-container {padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px;}
        
        /* Sidebar adjustments */
        .sidebar .sidebar-content {font-size: 16px;}

        /* Tabs styling */
        .stTabs [role="tablist"] {font-size: 18px; font-weight: bold;}

        /* Button styling for the refresh button */
        .stButton>button {
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Refresh Data button to regenerate datasets
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()

# Create tabs for Classification and Regression metrics
tab_class, tab_reg = st.tabs(["Classification Metrics", "Regression Metrics"])

# ========================== Classification Tab ==========================
with tab_class:
    st.header("Classification Metrics Explorer")
    
    # Sidebar for classification parameters
    st.sidebar.header("Classification Data Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500, key="class_n_samples")
    class_imbalance = st.sidebar.slider("Class Imbalance Ratio", 0.1, 0.9, 0.5, key="class_imbalance")
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2, key="class_noise")
    
    @st.cache_data
    def generate_classification_data(n_samples, class_imbalance, noise):
        # Determine number of samples per class
        n_class1 = int(n_samples * class_imbalance)
        n_class2 = n_samples - n_class1
        
        # Generate data for each class
        X1 = np.random.normal(0, 1, (n_class1, 2))
        y1 = np.ones(n_class1)
        X2 = np.random.normal(2, 1, (n_class2, 2))
        y2 = np.zeros(n_class2)
        
        # Combine and add noise
        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2])
        X += np.random.normal(0, noise, X.shape)
        return X, y

    X, y = generate_classification_data(n_samples, class_imbalance, noise_level)
    
    # Split data and train models with fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    rf_clf.fit(X_train, y_train)
    lr_clf.fit(X_train, y_train)
    
    # Predictions and probabilities
    rf_pred = rf_clf.predict(X_test)
    lr_pred = lr_clf.predict(X_test)
    rf_prob = rf_clf.predict_proba(X_test)[:, 1]
    lr_prob = lr_clf.predict_proba(X_test)[:, 1]
    
    # Layout: Data Distribution and Metrics side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Distribution")
        fig_data = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str),
                              title="Dataset Distribution",
                              labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'})
        st.plotly_chart(fig_data, use_container_width=True)
    
    with col2:
        st.subheader("Model Comparison Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Random Forest': [
                accuracy_score(y_test, rf_pred),
                precision_score(y_test, rf_pred),
                recall_score(y_test, rf_pred),
                f1_score(y_test, rf_pred)
            ],
            'Logistic Regression': [
                accuracy_score(y_test, lr_pred),
                precision_score(y_test, lr_pred),
                recall_score(y_test, lr_pred),
                f1_score(y_test, lr_pred)
            ]
        })
        st.dataframe(metrics_df)
    
    # ROC and Precision-Recall curves in side-by-side columns
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
        auc_rf = auc(fpr_rf, tpr_rf)
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines',
                                     name=f'Random Forest (AUC={auc_rf:.3f})'))
        
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
        auc_lr = auc(fpr_lr, tpr_lr)
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines',
                                     name=f'Logistic Regression (AUC={auc_lr:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                     name='Random', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate',
                              title='ROC Curves Comparison')
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with col4:
        st.subheader("Precision-Recall Curves")
        fig_pr = go.Figure()
        precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_prob)
        fig_pr.add_trace(go.Scatter(x=recall_rf, y=precision_rf, mode='lines', name='Random Forest'))
        
        precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_prob)
        fig_pr.add_trace(go.Scatter(x=recall_lr, y=precision_lr, mode='lines', name='Logistic Regression'))
        fig_pr.update_layout(xaxis_title='Recall',
                             yaxis_title='Precision',
                             title='Precision-Recall Curves')
        st.plotly_chart(fig_pr, use_container_width=True)
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Random Forest Confusion Matrix**")
        cm_rf = confusion_matrix(y_test, rf_pred)
        fig_cm_rf = px.imshow(cm_rf, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              title="Random Forest Confusion Matrix")
        st.plotly_chart(fig_cm_rf, use_container_width=True)
    with col6:
        st.markdown("**Logistic Regression Confusion Matrix**")
        cm_lr = confusion_matrix(y_test, lr_pred)
        fig_cm_lr = px.imshow(cm_lr, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              title="Logistic Regression Confusion Matrix")
        st.plotly_chart(fig_cm_lr, use_container_width=True)

# ========================== Regression Tab ==========================
with tab_reg:
    st.header("Regression Metrics Explorer")
    
    st.sidebar.header("Regression Data Parameters")
    n_samples_reg = st.sidebar.slider("Number of Samples (Regression)", 100, 1000, 500, key="reg_n_samples")
    noise_level_reg = st.sidebar.slider("Noise Level (Regression)", 0.0, 1.0, 0.2, key="reg_noise")
    nonlinearity = st.sidebar.slider("Nonlinearity", 0.0, 2.0, 1.0, key="nonlinearity")
    
    @st.cache_data
    def generate_regression_data(n_samples, noise, nonlinearity):
        X = np.random.uniform(-5, 5, (n_samples, 2))
        y = (X[:, 0] + nonlinearity * X[:, 0]**2 + 
             X[:, 1] + nonlinearity * np.sin(X[:, 1]))
        y += noise * np.random.normal(0, 1, n_samples)
        return X, y
    
    X_reg, y_reg = generate_regression_data(n_samples_reg, noise_level_reg, nonlinearity)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    lr_reg = LinearRegression()
    rf_reg.fit(X_train_reg, y_train_reg)
    lr_reg.fit(X_train_reg, y_train_reg)
    
    rf_pred_reg = rf_reg.predict(X_test_reg)
    lr_pred_reg = lr_reg.predict(X_test_reg)
    
    # Scatter plots: Actual vs. Predicted
    col1_reg, col2_reg = st.columns(2)
    with col1_reg:
        st.subheader("Actual vs Predicted (Random Forest)")
        fig_rf = px.scatter(x=y_test_reg, y=rf_pred_reg,
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Random Forest: Actual vs Predicted")
        fig_rf.add_trace(go.Scatter(x=[y_test_reg.min(), y_test_reg.max()],
                                    y=[y_test_reg.min(), y_test_reg.max()],
                                    mode='lines', name='Perfect Prediction'))
        st.plotly_chart(fig_rf, use_container_width=True)
    
    with col2_reg:
        st.subheader("Actual vs Predicted (Linear Regression)")
        fig_lr = px.scatter(x=y_test_reg, y=lr_pred_reg,
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Linear Regression: Actual vs Predicted")
        fig_lr.add_trace(go.Scatter(x=[y_test_reg.min(), y_test_reg.max()],
                                    y=[y_test_reg.min(), y_test_reg.max()],
                                    mode='lines', name='Perfect Prediction'))
        st.plotly_chart(fig_lr, use_container_width=True)
    
    # Regression metrics table
    st.subheader("Regression Metrics Comparison")
    metrics_reg_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE'],
        'Random Forest': [
            mean_squared_error(y_test_reg, rf_pred_reg),
            np.sqrt(mean_squared_error(y_test_reg, rf_pred_reg)),
            mean_absolute_error(y_test_reg, rf_pred_reg),
            r2_score(y_test_reg, rf_pred_reg),
            mean_absolute_percentage_error(y_test_reg, rf_pred_reg)
        ],
        'Linear Regression': [
            mean_squared_error(y_test_reg, lr_pred_reg),
            np.sqrt(mean_squared_error(y_test_reg, lr_pred_reg)),
            mean_absolute_error(y_test_reg, lr_pred_reg),
            r2_score(y_test_reg, lr_pred_reg),
            mean_absolute_percentage_error(y_test_reg, lr_pred_reg)
        ]
    })
    st.dataframe(metrics_reg_df)
    
    # Residual plots in two columns
    col3_reg, col4_reg = st.columns(2)
    with col3_reg:
        st.subheader("Residual Plot (Random Forest)")
        residuals_rf = y_test_reg - rf_pred_reg
        fig_res_rf = px.scatter(x=rf_pred_reg, y=residuals_rf,
                                labels={'x': 'Predicted', 'y': 'Residuals'},
                                title="Random Forest Residual Plot")
        fig_res_rf.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_res_rf, use_container_width=True)
    
    with col4_reg:
        st.subheader("Residual Plot (Linear Regression)")
        residuals_lr = y_test_reg - lr_pred_reg
        fig_res_lr = px.scatter(x=lr_pred_reg, y=residuals_lr,
                                labels={'x': 'Predicted', 'y': 'Residuals'},
                                title="Linear Regression Residual Plot")
        fig_res_lr.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_res_lr, use_container_width=True)

# ==================== Educational Content ====================
st.markdown("""
### Understanding Model Evaluation Metrics

#### Classification Metrics:
1. **Accuracy**: Overall correctness of predictions  
   - *Pros*: Simple to understand.  
   - *Cons*: Can be misleading with imbalanced classes.

2. **Precision**: Accuracy of positive predictions  
   - *High precision* = Low false positives.  
   - Important when false positives are costly.

3. **Recall**: Ability to identify all positive cases  
   - *High recall* = Low false negatives.  
   - Important when false negatives are costly.

4. **F1-Score**: Harmonic mean of precision and recall  
   - Balances precision and recall.  
   - Useful for imbalanced datasets.

5. **ROC-AUC**: Ability of the model to distinguish between classes  
   - 1.0 = Perfect prediction; 0.5 = Random guessing.

#### Regression Metrics:
1. **MSE (Mean Squared Error)**  
   - Penalizes larger errors more heavily.  
   - Units are squared.

2. **RMSE (Root Mean Squared Error)**  
   - Same units as the target variable; more interpretable.

3. **MAE (Mean Absolute Error)**  
   - Average absolute error; less sensitive to outliers.

4. **R² (R-squared)**  
   - Proportion of variance explained by the model.  
   - 1.0 = Perfect fit; can be negative if the model is poor.

5. **MAPE (Mean Absolute Percentage Error)**  
   - Error expressed in percentage terms; useful for comparing across scales.

### Best Practices:
- Use multiple metrics to get a comprehensive view of model performance.
- Consider both aggregate metrics and detailed plots.
- Pay attention to data distribution and the cost of different types of errors.
""")



# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1: Classification Metrics
quiz_answer_1 = st.radio(
    "Which metric is most suitable for evaluating a classification model when dealing with an imbalanced dataset?",
    (
        "Accuracy",
        "Precision-Recall AUC",
        "Mean Absolute Error (MAE)"
    ),
    key="quiz_q1"
)

if quiz_answer_1 == "Precision-Recall AUC":
    st.success("✅ Correct! Precision-Recall AUC is better for imbalanced datasets because it focuses on positive class performance.")
else:
    st.error("❌ Not quite. Accuracy can be misleading with imbalanced data, and MAE is a regression metric.")

# Question 2: Regression Metrics
quiz_answer_2 = st.radio(
    "Which regression metric penalizes larger errors more heavily?",
    (
        "Mean Absolute Error (MAE)",
        "Mean Squared Error (MSE)",
        "R-squared (R²)"
    ),
    key="quiz_q2"
)

if quiz_answer_2 == "Mean Squared Error (MSE)":
    st.success("✅ Correct! MSE gives more weight to large errors, making it sensitive to outliers.")
else:
    st.error("❌ Not quite. MAE treats all errors equally, and R² measures variance explained by the model.")
