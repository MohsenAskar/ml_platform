import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
st.set_page_config(page_title="Elastic Net Visualization", page_icon="🕸️", layout="wide")
st.title("Elastic Net Regression 🕸️")

# Educational content
st.markdown("""
### What is Elastic Net?
Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization:

1. **Key Concepts**
   - Combines Lasso and Ridge penalties
   - Controls feature selection (L1)
   - Handles correlated features (L2)
   - Balances sparsity and stability

2. **Parameters**
   - Alpha (α): Overall regularization strength
   - L1 Ratio (r): Balance between L1 and L2
     * r = 1: Pure Lasso
     * r = 0: Pure Ridge
     * 0 < r < 1: Elastic Net

3. **Advantages**
   - Better than Lasso for correlated features
   - Performs feature selection
   - More stable than pure Lasso
   - Helps prevent overfitting
""")

# Parameters explanation in the sidebar
st.sidebar.header("Elastic Net Parameters")

st.sidebar.markdown("**Regularization Strength (α)**")
alpha = st.sidebar.slider("Alpha", 0.0, 2.0, 1.0)

st.sidebar.markdown("**L1 Ratio (r)**")
l1_ratio = st.sidebar.slider("L1 Ratio", 0.0, 1.0, 0.5)

# Data parameters
st.sidebar.markdown("**Data Generation**")
n_samples = st.sidebar.slider("Number of Samples", 50, 200, 100)
n_features = st.sidebar.slider("Number of Features", 2, 10, 5)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)

# Generate synthetic data
@st.cache_data
def generate_data(n_samples, n_features, noise):
    # Generate true coefficients with some sparsity
    true_coef = np.zeros(n_features)
    true_coef[:3] = [1.0, 0.5, 0.2]  # Only first 3 features are important
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.2  # Correlate feature 1 with 0
    
    # Generate target
    y = np.dot(X, true_coef)
    y += noise * np.random.randn(n_samples)
    
    return X, y, true_coef

X, y, true_coef = generate_data(n_samples, n_features, noise_level)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def plot_coefficients_comparison(elastic_net, lasso, ridge, true_coef):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.2
    x = np.arange(len(true_coef))
    
    ax.bar(x - 1.5*width, true_coef, width, label='True', color='gray', alpha=0.5)
    ax.bar(x - 0.5*width, elastic_net.coef_, width, label='Elastic Net', color='blue')
    ax.bar(x + 0.5*width, lasso.coef_, width, label='Lasso', color='red')
    ax.bar(x + 1.5*width, ridge.coef_, width, label='Ridge', color='green')
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'X{i+1}' for i in range(len(true_coef))])
    ax.legend()
    
    return fig

def plot_regularization_path(X, y, l1_ratio):
    alphas = np.logspace(-2, 2, 50)
    coefs = []
    
    for alpha in alphas:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X, y)
        coefs.append(model.coef_)
    
    coefs = np.array(coefs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature_idx in range(X.shape[1]):
        ax.plot(np.log10(alphas), coefs[:, feature_idx],
                label=f'Feature {feature_idx+1}')
    
    ax.axvline(np.log10(alpha), color='k', linestyle='--', alpha=0.5,
               label='Current Alpha')
    ax.set_xlabel('log(alpha)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Regularization Path')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

def plot_prediction_error(elastic_net, X, y):
    y_pred = elastic_net.predict(X)
    residuals = y - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual')
    ax.set_title('Residual Plot')
    
    return fig

# Fit models
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
lasso = Lasso(alpha=alpha)
ridge = Ridge(alpha=alpha)

elastic_net.fit(X_scaled, y)
lasso.fit(X_scaled, y)
ridge.fit(X_scaled, y)

# Create containers for visualization
row1 = st.container()
row2 = st.container()

with row1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Coefficient Comparison")
        fig = plot_coefficients_comparison(elastic_net, lasso, ridge, true_coef)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Regularization Path")
        fig = plot_regularization_path(X_scaled, y, l1_ratio)
        st.pyplot(fig)
        plt.close(fig)

with row2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Prediction Error")
        fig = plot_prediction_error(elastic_net, X_scaled, y)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Model Performance")
        performance_df = pd.DataFrame({
            'Metric': ['R² Score', 'Active Features', 'Mean Absolute Error'],
            'Elastic Net': [
                elastic_net.score(X_scaled, y),
                np.sum(elastic_net.coef_ != 0),
                np.mean(np.abs(y - elastic_net.predict(X_scaled)))
            ],
            'Lasso': [
                lasso.score(X_scaled, y),
                np.sum(lasso.coef_ != 0),
                np.mean(np.abs(y - lasso.predict(X_scaled)))
            ],
            'Ridge': [
                ridge.score(X_scaled, y),
                np.sum(ridge.coef_ != 0),
                np.mean(np.abs(y - ridge.predict(X_scaled)))
            ]
        })
        st.write(performance_df)

# Feature importance analysis
st.write("### Feature Importance Analysis")
importance_df = pd.DataFrame({
    'Feature': [f'X{i+1}' for i in range(n_features)],
    'True Coefficient': true_coef,
    'Elastic Net': elastic_net.coef_,
    'Lasso': lasso.coef_,
    'Ridge': ridge.coef_
})
st.write(importance_df)

# Best use cases and references
st.markdown("""
### When to Use Elastic Net?

#### Best Use Cases:
1. **High-Dimensional Data**
   - Genomics data
   - Text analysis
   - Feature selection

2. **Correlated Features**
   - Financial indicators
   - Sensor data
   - Multiple related predictors

3. **Sparse Solutions Needed**
   - Feature selection
   - Model interpretation
   - Resource constraints

#### Advantages:
- Handles correlated features
- Performs feature selection
- More stable than Lasso
- Combines best of L1 and L2
- Good for high-dimensional data

#### Limitations:
- Two parameters to tune
- May be slower than pure Lasso/Ridge
- Less interpretable than simple models
- Requires feature scaling
- May not capture non-linear relationships

### Real-World Applications:

1. **Genomics**
   - Gene selection
   - Disease prediction
   - Biomarker identification

2. **Finance**
   - Portfolio optimization
   - Risk modeling
   - Feature selection

3. **Text Analysis**
   - Document classification
   - Feature extraction
   - Sparse representation

### Learn More:
1. "Elements of Statistical Learning" - Regularization Chapter
2. Original Elastic Net Paper by Zou and Hastie
3. Scikit-learn Elastic Net Documentation
""")



# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary advantage of Elastic Net over Lasso and Ridge regression?",
    (
        "It combines both L1 (Lasso) and L2 (Ridge) regularization penalties.",
        "It completely eliminates multicollinearity in all cases.",
        "It always performs better than Lasso and Ridge regression."
    )
)

if quiz_answer_1 == "It combines both L1 (Lasso) and L2 (Ridge) regularization penalties.":
    st.success("Correct! Elastic Net is a hybrid of Lasso and Ridge, using both L1 and L2 penalties to balance feature selection and shrinkage.")
else:
    st.error("Not quite. While Elastic Net helps reduce multicollinearity and can outperform Lasso or Ridge in some cases, it does not always guarantee better performance.")

# Question 2
quiz_answer_2 = st.radio(
    "When is Elastic Net particularly useful?",
    (
        "When there are many correlated features in the dataset.",
        "When there are only a few features in the dataset.",
        "When the dataset has no noise or redundant features."
    )
)

if quiz_answer_2 == "When there are many correlated features in the dataset.":
    st.success("Correct! Elastic Net is particularly useful when dealing with highly correlated features, as it can handle multicollinearity better than Lasso.")
else:
    st.error("Not quite. Elastic Net is designed for cases with many features, especially when they are correlated, rather than for small or perfectly clean datasets.")
