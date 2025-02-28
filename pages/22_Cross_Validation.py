import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.datasets import make_classification, make_blobs
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Cross Validation Visualization", page_icon="✂️", layout="wide")
st.title("Cross Validation ✂️")

# Educational content
st.markdown("""
### What is Cross Validation?
Cross Validation helps assess model performance more reliably:

1. **Key Concepts**
   - Split data into training and testing sets
   - Repeat process multiple times
   - Average performance across splits
   - Helps detect overfitting

2. **Common Methods**
   - K-Fold: Split into K equal parts
   - Stratified: Preserve class proportions
   - Time Series: Respect temporal order
   - Leave-One-Out: Special case of K-Fold

3. **Why Cross Validate?**
   - More reliable performance estimates
   - Detect overfitting/underfitting
   - Better model selection
   - Assess model stability
""")

# Parameters explanation in the sidebar
st.sidebar.header("Cross Validation Parameters")

st.sidebar.markdown("**CV Method**")
cv_method = st.sidebar.selectbox("Validation Method", 
    ["K-Fold", "Stratified K-Fold", "Time Series Split"])

st.sidebar.markdown("**Number of Splits**")
n_splits = st.sidebar.slider("Number of Splits", 2, 10, 5)

st.sidebar.markdown("**Model Type**")
model_type = st.sidebar.selectbox("Model", 
    ["SVM", "Decision Tree"])

# Data parameters
n_samples = st.sidebar.slider("Number of Samples", 50, 200, 100)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)

# Generate synthetic data
@st.cache_data
def generate_data(n_samples, noise):
    if cv_method == "Time Series Split":
        # Generate time series data with trend and seasonality
        t = np.linspace(0, 10, n_samples)
        y = np.sin(t) + 0.1 * t + noise * np.random.randn(n_samples)
        X = np.column_stack([t, np.cos(t)])
        y = (y > np.median(y)).astype(int)  # Convert to binary classification
    else:
        # For regular classification, use class_sep instead of noise
        X, y = make_classification(n_samples=n_samples, 
                                 n_features=2, 
                                 n_redundant=0, 
                                 n_informative=2,
                                 class_sep=1.0 - noise,  # Convert noise to class separation
                                 random_state=42)
    return X, y

X, y = generate_data(n_samples, noise_level)

def get_cv_splitter():
    if cv_method == "K-Fold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_method == "Stratified K-Fold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        return TimeSeriesSplit(n_splits=n_splits)

def get_model():
    if model_type == "SVM":
        return SVC(kernel='linear')
    else:
        return DecisionTreeClassifier()

def plot_cv_splits(X, y, cv):
    n_splits = cv.n_splits
    n_rows = int(np.ceil(n_splits / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        ax = axes[i]
        
        # Plot all points in light gray
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.5)
        
        # Plot train and test points
        ax.scatter(X[train_idx, 0], X[train_idx, 1], c=y[train_idx],
                  label='Train', alpha=0.6)
        ax.scatter(X[test_idx, 0], X[test_idx, 1], c=y[test_idx],
                  label='Test', alpha=0.6, marker='s')
        
        ax.set_title(f'Split {i+1}')
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_splits, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_fold_metrics(cv_scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual fold scores
    ax.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', label='Fold Score')
    ax.axhline(y=np.mean(cv_scores), color='r', linestyle='--',
               label=f'Mean Score: {np.mean(cv_scores):.3f}')
    
    ax.fill_between(range(1, len(cv_scores) + 1),
                    np.mean(cv_scores) - np.std(cv_scores),
                    np.mean(cv_scores) + np.std(cv_scores),
                    alpha=0.2, color='r')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross Validation Scores')
    ax.legend()
    
    return fig

def plot_learning_curves(X, y, cv):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    test_scores = []
    
    model = get_model()
    
    for size in train_sizes:
        size_train_scores = []
        size_test_scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            # Take subset of training data
            n_train = int(len(train_idx) * size)
            subset_idx = train_idx[:n_train]
            
            # Train model
            model.fit(X[subset_idx], y[subset_idx])
            
            # Score
            train_score = model.score(X[subset_idx], y[subset_idx])
            test_score = model.score(X[test_idx], y[test_idx])
            
            size_train_scores.append(train_score)
            size_test_scores.append(test_score)
        
        train_scores.append(np.mean(size_train_scores))
        test_scores.append(np.mean(size_test_scores))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes * 100, train_scores, 'o-', label='Training Score')
    ax.plot(train_sizes * 100, test_scores, 'o-', label='Cross-validation Score')
    ax.set_xlabel('Training Data Size (%)')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curves')
    ax.legend()
    
    return fig

# Create containers for visualization
row1 = st.container()
row2 = st.container()
row3 = st.container()

# Get CV splitter and perform cross validation
cv = get_cv_splitter()
model = get_model()

# Compute CV scores
cv_scores = []
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
    cv_scores.append(score)

with row1:
    st.write("### Cross Validation Splits")
    fig = plot_cv_splits(X, y, cv)
    st.pyplot(fig)
    plt.close(fig)

with row2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Cross Validation Scores")
        fig = plot_fold_metrics(cv_scores)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Learning Curves")
        fig = plot_learning_curves(X, y, cv)
        st.pyplot(fig)
        plt.close(fig)

# Summary statistics
st.write("### Cross Validation Results")
results_df = pd.DataFrame({
    'Metric': ['Mean Accuracy', 'Std Accuracy', 'Min Accuracy', 'Max Accuracy'],
    'Value': [np.mean(cv_scores), np.std(cv_scores),
              np.min(cv_scores), np.max(cv_scores)]
})
st.write(results_df)

# Best use cases and references
st.markdown("""
### When to Use Cross Validation?

#### Best Use Cases:
1. **Model Selection**
   - Comparing different algorithms
   - Tuning hyperparameters
   - Feature selection

2. **Performance Estimation**
   - Reliable error estimates
   - Model stability assessment
   - Overfitting detection

3. **Limited Data**
   - Small datasets
   - Imbalanced classes
   - Expensive data collection

#### Different CV Methods:

1. **K-Fold**
   - General purpose
   - Independent data
   - Sufficient data size

2. **Stratified K-Fold**
   - Imbalanced classes
   - Small datasets
   - Binary classification

3. **Time Series Split**
   - Temporal data
   - Sequential patterns
   - Future prediction

#### Advantages:
- More reliable performance estimates
- Better model selection
- Overfitting detection
- Model stability assessment
- Better use of limited data

#### Limitations:
- Computationally expensive
- May be slow for large datasets
- Some methods not suitable for time series
- May need modification for special cases
- Requires careful selection of K

### Real-World Applications:

1. **Machine Learning**
   - Model selection
   - Hyperparameter tuning
   - Performance estimation

2. **Scientific Research**
   - Method validation
   - Result reliability
   - Performance comparison

3. **Business Applications**
   - Risk assessment
   - Predictive modeling
   - Resource allocation

### Learn More:
1. Scikit-learn Cross Validation Guide
2. "Introduction to Statistical Learning" - Model Assessment
3. "Elements of Statistical Learning" - Cross Validation
""")


# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary purpose of cross-validation in machine learning?",
    (
        "To evaluate a model’s performance on unseen data.",
        "To increase the training accuracy of a model.",
        "To eliminate the need for a test dataset."
    )
)

if quiz_answer_1 == "To evaluate a model’s performance on unseen data.":
    st.success("Correct! Cross-validation helps assess how well a model generalizes to new, unseen data by splitting the dataset into multiple training and validation sets.")
else:
    st.error("Not quite. Cross-validation does not increase training accuracy or remove the need for a test set; it helps evaluate a model’s generalization ability.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a common advantage of k-fold cross-validation?",
    (
        "It provides a more reliable estimate of model performance by using multiple training-validation splits.",
        "It guarantees the best possible accuracy for the model.",
        "It prevents the model from overfitting completely."
    )
)

if quiz_answer_2 == "It provides a more reliable estimate of model performance by using multiple training-validation splits.":
    st.success("Correct! K-fold cross-validation improves the reliability of performance estimates by averaging results across multiple splits.")
else:
    st.error("Not quite. While k-fold cross-validation helps reduce variance in performance estimation, it does not guarantee the best accuracy or completely prevent overfitting.")
