import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.data_generators import generate_regression_data
from sklearn.tree import DecisionTreeRegressor, plot_tree
import seaborn as sns

st.set_page_config(page_title="Gradient Boosting Machine Visualization", page_icon="🚀", layout="wide")
st.title("Gradient Boosting Machine (GBM) 🚀")

# Educational content
st.markdown("""
### What is Gradient Boosting Machine (GBM)?
GBM is like a team of experts learning from each other's mistakes:

1. **Sequential Learning**
   - Each tree learns from the errors of previous trees
   - Trees are built one after another, not independently
   - Later trees focus on hard examples

2. **Error Correction**
   - First tree makes initial predictions
   - Next trees predict the errors (residuals)
   - Predictions are combined with weights

3. **Controlled Learning Rate**
   - Each tree's contribution is scaled (learning rate)
   - Helps prevent overfitting
   - Slower but more robust learning
""")

# Parameters explanation in the sidebar
st.sidebar.header("GBM Parameters")

st.sidebar.markdown("**Number of Trees**")
st.sidebar.markdown("""
How many trees to use:
- More trees = More complex patterns
- But risk of overfitting
""")
n_trees = st.sidebar.slider("Number of Trees", 1, 10, 3)

st.sidebar.markdown("**Learning Rate**")
st.sidebar.markdown("""
How much to trust each tree:
- Higher = Faster learning but less stable
- Lower = Slower but more robust
""")
learning_rate = st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.3)

st.sidebar.markdown("**Max Tree Depth**")
st.sidebar.markdown("""
How complex each tree should be:
- Deeper = More detailed correction
- Shallower = More generalization
""")
max_depth = st.sidebar.slider("Max Depth", 1, 5, 2)

n_points = st.sidebar.slider("Number of Data Points", 50, 200, 100)

# Generate synthetic data
@st.cache_data
def get_data(n_points):
    X, y = generate_regression_data(n_points)
    # Sort X for better visualization
    sort_idx = np.argsort(X.ravel())
    return X[sort_idx], y[sort_idx]

X, y = get_data(n_points)

def plot_predictions_and_residuals(X, y, y_pred, residuals, tree_idx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Current predictions
    ax1.scatter(X, y, c='blue', label='True values', alpha=0.5)
    ax1.plot(X, y_pred, 'r-', label='Current prediction', linewidth=2)
    ax1.set_title(f'Predictions after Tree {tree_idx}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    
    # Plot 2: Residuals
    ax2.scatter(X, residuals, c='green', label='Residuals', alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title(f'Residuals after Tree {tree_idx}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Residual')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_tree_contribution(X, tree_pred, tree_idx):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, tree_pred * learning_rate, 'g-', 
            label=f'Tree {tree_idx} contribution')
    ax.set_title(f'Tree {tree_idx} Contribution (scaled by learning rate)')
    ax.set_xlabel('X')
    ax.set_ylabel('Contribution')
    ax.legend()
    return fig

# Create containers for the two main rows
row1 = st.container()
row2 = st.container()

# Initialize or get state
if 'trees' not in st.session_state:
    st.session_state.trees = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Training controls
train_button = st.button("Train GBM")

if train_button:
    st.session_state.trees = []
    current_prediction = np.zeros_like(y)
    
    for i in range(n_trees):
        with row1:
            col1, col2 = st.columns(2)
            
            # Calculate residuals
            residuals = y - current_prediction
            
            # Train tree on residuals
            tree = DecisionTreeRegressor(max_depth=max_depth)
            tree.fit(X, residuals)
            st.session_state.trees.append(tree)
            
            # Get tree predictions and update
            tree_pred = tree.predict(X)
            current_prediction += learning_rate * tree_pred
            
            # Update visualizations
            with col1:
                st.write(f"### Tree {i+1} Learning Progress")
                fig = plot_predictions_and_residuals(
                    X.ravel(), y, current_prediction, residuals, i+1)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.write(f"### Tree {i+1} Contribution")
                fig = plot_tree_contribution(X.ravel(), tree_pred, i+1)
                st.pyplot(fig)
                plt.close(fig)
        
        # Second row: tree structure
        with row2:
            st.write(f"### Tree {i+1} Structure (Learning Residuals)")
            fig, ax = plt.subplots(figsize=(20, 8))
            plot_tree(tree, filled=True, feature_names=['X'], 
                     ax=ax, precision=3)
            st.pyplot(fig)
            plt.close(fig)
    
    # After training, show final model analysis
    st.write("### Final Model Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot actual vs predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y, current_prediction, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', 
                label='Perfect prediction')
        ax.set_xlabel('Actual values')
        ax.set_ylabel('Predicted values')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Plot error distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        final_errors = y - current_prediction
        sns.histplot(final_errors, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title('Final Error Distribution')
        ax.set_xlabel('Error')
        st.pyplot(fig)
        plt.close(fig)
    
    # Show error metrics
    mse = np.mean((y - current_prediction) ** 2)
    mae = np.mean(np.abs(y - current_prediction))
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

# Best use cases and references
st.markdown("""
### When to Use Gradient Boosting Machines?

#### Best Use Cases:
1. **Structured/Tabular Data**
   - Customer analytics
   - Financial forecasting
   - Risk modeling

2. **High-Precision Required**
   - Competition winning models
   - Critical predictions
   - Industrial processes

3. **Feature Importance Analysis**
   - Marketing attribution
   - Risk factor analysis
   - Variable selection

#### Advantages:
- High prediction accuracy
- Handles mixed type features
- Good for imbalanced data
- Built-in feature importance
- Less data preprocessing needed

#### Limitations:
- Sequential nature (can't parallelize)
- Risk of overfitting
- Requires careful parameter tuning
- More computationally intensive

### Real-World Applications:

1. **Finance**
   - Stock price prediction
   - Credit risk assessment
   - Fraud detection

2. **Marketing**
   - Click-through rate prediction
   - Customer lifetime value
   - Churn prediction

3. **Web Search**
   - Learning to rank
   - Relevance scoring
   - Ad click prediction

### Learn More:
1. XGBoost documentation
2. "The Elements of Statistical Learning" - Chapter on Boosting
3. "Gradient Boosting Machines" course by fast.ai
""")

import streamlit as st

# Interactive Quiz
st.subheader("Test Your Understanding of Gradient Boosting Machines (GBM)")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary concept behind Gradient Boosting Machines (GBM)?",
    (
        "It builds multiple independent trees and averages their predictions.",
        "It trains models sequentially, where each model corrects the errors of the previous one.",
        "It uses deep learning techniques to improve tree-based models."
    )
)

if quiz_answer_1 == "It trains models sequentially, where each model corrects the errors of the previous one.":
    st.success("Correct! GBM builds trees iteratively, with each tree learning from the residual errors of the previous ones to improve performance.")
else:
    st.error("Not quite. GBM follows a sequential boosting approach where each model corrects the mistakes of its predecessor.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a potential drawback of Gradient Boosting Machines (GBM)?",
    (
        "It is highly prone to overfitting if not properly tuned.",
        "It cannot handle missing values.",
        "It does not perform well with large datasets."
    )
)

if quiz_answer_2 == "It is highly prone to overfitting if not properly tuned.":
    st.success("Correct! GBM can overfit the training data, especially if too many trees are used or if learning rates are too high.")
else:
    st.error("Not quite. GBM can handle missing values and large datasets, but it requires careful hyperparameter tuning to prevent overfitting.")
