import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Bagging vs Boosting Comparison", page_icon="🌳", layout="wide")
st.title("Ensemble Learning: Bagging🌳 vs Boosting 🚀 Comparison")

# Educational content
st.markdown("""
### Understanding Ensemble Methods
Think of it like getting opinions from a group of experts:

#### Bagging (Bootstrap Aggregating)
- Like asking multiple experts independently
- Each expert sees a random subset of data
- All experts are equally important
- Final decision is an average of all opinions

#### Boosting
- Like experts learning from previous mistakes
- Each expert focuses on hard cases
- Experts have different importance weights
- Final decision is a weighted combination

### Key Differences
1. **Training Process**
   - Bagging: Parallel (all models trained independently)
   - Boosting: Sequential (each model learns from previous errors)

2. **Error Handling**
   - Bagging: Reduces variance (overfitting)
   - Boosting: Reduces bias (underfitting)

3. **Model Weights**
   - Bagging: Equal weights
   - Boosting: Different weights based on performance
""")

# Parameters explanation in the sidebar
st.sidebar.header("Ensemble Parameters")

st.sidebar.markdown("**Number of Estimators**")
st.sidebar.markdown("""
How many models in ensemble:
- More models = More stable but slower
- Fewer models = Faster but less stable
""")
n_estimators = st.sidebar.slider("Number of Estimators", 1, 10, 5)

st.sidebar.markdown("**Max Tree Depth**")
st.sidebar.markdown("""
How complex each tree can be:
- Deeper = More complex patterns
- Shallower = More generalization
""")
max_depth = st.sidebar.slider("Max Depth", 1, 5, 2)

st.sidebar.markdown("**Learning Rate (Boosting)**")
learning_rate = st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.3)

# Data parameters
n_points = st.sidebar.slider("Number of Points", 50, 200, 100)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)

# Generate synthetic data
@st.cache_data
def generate_data(n_points, noise_level):
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y = np.sin(X).ravel() + np.cos(X/2).ravel()
    y += np.random.normal(0, noise_level, n_points)
    return X, y

X, y = generate_data(n_points, noise_level)

def plot_prediction_process(X, y, model_type="bagging", n_trees=5, max_depth=2, 
                          learning_rate=0.1):
    # Calculate number of rows and columns needed
    n_cols = min(3, n_trees)
    n_rows = int(np.ceil(n_trees / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle(f'{"Bagging" if model_type=="bagging" else "Boosting"} Process', 
                y=1.02)
    
    # Make axes 2D if it's 1D
    if n_trees <= n_cols:
        axes = axes.reshape(1, -1)
    
    # Hide unused subplots
    for i in range(n_trees, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    predictions = []
    trees = []
    errors = []
    
    for i in range(n_trees):
        # Create and train individual tree
        if model_type == "bagging":
            # Random subset for bagging
            indices = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegressor(max_depth=max_depth)
            tree.fit(X[indices], y[indices])
            pred = tree.predict(X)
            
        else:  # boosting
            if i == 0:
                tree = DecisionTreeRegressor(max_depth=max_depth)
                tree.fit(X, y)
                pred = tree.predict(X)
                current_pred = pred.copy()
            else:
                # Fit on residuals
                residuals = y - current_pred
                tree = DecisionTreeRegressor(max_depth=max_depth)
                tree.fit(X, residuals)
                pred = tree.predict(X) * learning_rate
                current_pred += pred
                pred = current_pred
        
        trees.append(tree)
        predictions.append(pred)
        
        # Calculate error
        if model_type == "bagging":
            ensemble_pred = np.mean(predictions, axis=0)
        else:  # boosting
            ensemble_pred = pred
        
        error = np.mean((y - ensemble_pred) ** 2)
        errors.append(error)
        
        # Plot individual tree prediction
        # Plot for all trees
        row = i // min(3, n_trees)
        col = i % min(3, n_trees)
        ax = axes[row, col]
        ax.scatter(X, y, alpha=0.5, label='Data')
        ax.plot(X, pred, 'r-', label=f'Tree {i+1}')
        if model_type == "bagging":
            ax.plot(X, ensemble_pred, 'g-', label='Ensemble')
        ax.set_title(f'Tree {i+1}')
        ax.legend()
    
    return fig, trees, errors

def plot_error_comparison(bagging_errors, boosting_errors):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(bagging_errors) + 1), bagging_errors, 
            label='Bagging', marker='o')
    ax.plot(range(1, len(boosting_errors) + 1), boosting_errors, 
            label='Boosting', marker='o')
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Error Reduction: Bagging vs Boosting')
    ax.legend()
    return fig

# Create containers for visualization
row1 = st.container()
row2 = st.container()
row3 = st.container()

# Run both methods
with st.spinner("Training models..."):
    # Bagging process
    bagging_fig, bagging_trees, bagging_errors = plot_prediction_process(
        X, y, "bagging", n_estimators, max_depth)
    
    # Boosting process
    boosting_fig, boosting_trees, boosting_errors = plot_prediction_process(
        X, y, "boosting", n_estimators, max_depth, learning_rate)
    
    # Error comparison
    error_fig = plot_error_comparison(bagging_errors, boosting_errors)

# Display visualizations
with row1:
    st.write("### Bagging Process")
    st.pyplot(bagging_fig)
    plt.close(bagging_fig)

with row2:
    st.write("### Boosting Process")
    st.pyplot(boosting_fig)
    plt.close(boosting_fig)

with row3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Error Comparison")
        st.pyplot(error_fig)
        plt.close(error_fig)
    
    with col2:
        st.write("### Performance Metrics")
        metrics_df = pd.DataFrame({
            'Method': ['Bagging', 'Boosting'],
            'Final Error': [bagging_errors[-1], boosting_errors[-1]],
            'Error Reduction (%)': [
                (bagging_errors[0] - bagging_errors[-1]) / bagging_errors[0] * 100,
                (boosting_errors[0] - boosting_errors[-1]) / boosting_errors[0] * 100
            ]
        })
        st.write(metrics_df)

# Best use cases and references
st.markdown("""
### When to Use Each Method?

#### Bagging Best Use Cases:
1. **High Variance Problems**
   - Complex models prone to overfitting
   - Noisy data
   - Limited data

2. **Parallel Processing Needed**
   - Large datasets
   - Time constraints
   - Available computing resources

3. **Feature Importance**
   - Random Forest feature selection
   - Variable ranking
   - Feature elimination

#### Boosting Best Use Cases:
1. **High Bias Problems**
   - Simple models
   - Underfitting issues
   - Complex patterns

2. **Accuracy is Critical**
   - Competition winning
   - Critical predictions
   - Fine-tuned performance

3. **Gradual Learning**
   - Incremental improvement
   - Error focusing
   - Pattern refinement

### Key Considerations:

#### Bagging
- Pros:
  * Reduces overfitting
  * Parallel processing
  * More stable
- Cons:
  * May not be as accurate as boosting
  * Requires more memory
  * May be computationally intensive

#### Boosting
- Pros:
  * Often better accuracy
  * Smaller memory footprint
  * Good for feature selection
- Cons:
  * Can overfit
  * Sequential (slower)
  * Sensitive to noisy data

### Real-World Applications:

1. **Bagging**
   - Random Forest for robust predictions
   - Bootstrap for uncertainty estimation
   - Ensemble feature selection

2. **Boosting**
   - XGBoost for competitions
   - AdaBoost for face detection
   - LightGBM for large datasets

### Learn More:
1. "Introduction to Statistical Learning" - Ensemble Methods
2. Scikit-learn Ensemble Documentation
3. "Elements of Statistical Learning" - Boosting Chapter
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the key difference between Bagging and Boosting?",
    (
        "Bagging trains models independently, while Boosting trains models sequentially.",
        "Bagging always results in higher accuracy than Boosting.",
        "Boosting randomly selects features, whereas Bagging does not."
    )
)

if quiz_answer_1 == "Bagging trains models independently, while Boosting trains models sequentially.":
    st.success("Correct! Bagging builds multiple models in parallel, while Boosting builds models sequentially, with each new model correcting the errors of the previous one.")
else:
    st.error("Not quite. Bagging and Boosting differ in how they train models: Bagging trains them independently in parallel, while Boosting trains them sequentially.")

# Question 2
quiz_answer_2 = st.radio(
    "Which of the following is a common drawback of Boosting compared to Bagging?",
    (
        "Boosting is more prone to overfitting.",
        "Boosting cannot be used for classification problems.",
        "Boosting always requires more training data than Bagging."
    )
)

if quiz_answer_2 == "Boosting is more prone to overfitting.":
    st.success("Correct! Since Boosting focuses on correcting mistakes from previous models, it can overfit the training data if not properly tuned.")
else:
    st.error("Not quite. Boosting can be used for classification and does not always require more data, but it is prone to overfitting if not well-tuned.")


