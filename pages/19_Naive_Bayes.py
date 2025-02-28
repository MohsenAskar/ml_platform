import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
import seaborn as sns
from scipy.stats import norm
import pandas as pd

st.set_page_config(page_title="Naive Bayes Visualization", page_icon="📊", layout="wide")
st.title("Naïve Bayes calssification 🎲")

# Educational content
st.markdown("""
### What is Naive Bayes?
Naive Bayes is like a probability calculator that assumes each feature is independent:

1. **Key Concepts**
   - Based on Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
   - "Naive" because it assumes feature independence
   - Learns probability distributions for each feature
   - Fast and simple but powerful

2. **How it Works**
   - Calculate probability distributions for each feature per class
   - Multiply probabilities together (due to independence assumption)
   - Choose class with highest probability
   - Works well for text classification and simple datasets

3. **Types of Naive Bayes**
   - Gaussian: For continuous data
   - Multinomial: For count data (like text)
   - Bernoulli: For binary data
""")

# Parameters explanation in the sidebar
st.sidebar.header("Naive Bayes Parameters")

st.sidebar.markdown("**Dataset Parameters**")
n_classes = st.sidebar.slider("Number of Classes", 2, 4, 2)
n_features = st.sidebar.slider("Number of Features", 2, 3, 2)
n_points = st.sidebar.slider("Number of Points per Class", 50, 200, 100)

st.sidebar.markdown("**Visualization**")
show_probability = st.sidebar.checkbox("Show Probability Distributions", True)
show_decision = st.sidebar.checkbox("Show Decision Boundaries", True)

# Generate synthetic data
@st.cache_data
def generate_data(n_classes, n_features, n_points):
    X, y = make_blobs(n_samples=n_points*n_classes, 
                      n_features=n_features,
                      centers=n_classes,
                      cluster_std=1.5,
                      random_state=42)
    return X, y

X, y = generate_data(n_classes, n_features, n_points)

# Train Naive Bayes
model = GaussianNB()
model.fit(X, y)

def plot_feature_distributions(X, y, feature_idx, classes):
    plt.figure(figsize=(10, 6))
    
    for class_idx in classes:
        class_data = X[y == class_idx, feature_idx]
        
        # Plot histogram
        plt.hist(class_data, bins=20, density=True, alpha=0.3,
                label=f'Class {class_idx} (hist)')
        
        # Plot fitted Gaussian
        mu = np.mean(class_data)
        std = np.std(class_data)
        x = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 100)
        plt.plot(x, norm.pdf(x, mu, std),
                label=f'Class {class_idx} (Gaussian)')
    
    plt.title(f'Feature {feature_idx + 1} Distribution by Class')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    return plt.gcf()

def plot_decision_boundary_2d(X, y, model):
    plt.figure(figsize=(10, 6))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='black')
    plt.colorbar(scatter)
    
    plt.title('Decision Boundaries')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return plt.gcf()

def plot_probability_surface_2d(X, y, model):
    fig = plt.figure(figsize=(10, 6))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get probabilities
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot probability surface
    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), alpha=0.5, cmap='RdYlBu')
    plt.colorbar(label='Probability of Class 1')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='black')
    
    plt.title('Probability Surface')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return fig

def plot_feature_importance():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate feature importance based on variance
    var_per_class = []
    for class_idx in range(n_classes):
        class_data = X[y == class_idx]
        var_per_class.append(np.var(class_data, axis=0))
    
    var_per_class = np.array(var_per_class)
    importance = np.mean(var_per_class, axis=0)
    
    # Plot feature importance
    ax.bar(range(n_features), importance)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([f'Feature {i+1}' for i in range(n_features)])
    ax.set_title('Feature Importance (Based on Variance)')
    ax.set_ylabel('Average Variance')
    
    return fig

# Create main display areas
row1 = st.container()
row2 = st.container()

with row1:
    # Show feature distributions
    st.write("### Feature Distributions")
    cols = st.columns(n_features)
    for i, col in enumerate(cols):
        with col:
            fig = plot_feature_distributions(X, y, i, range(n_classes))
            st.pyplot(fig)
            plt.close(fig)

with row2:
    col1, col2 = st.columns(2)
    
    with col1:
        if n_features == 2:
            if show_decision:
                st.write("### Decision Boundaries")
                fig = plot_decision_boundary_2d(X, y, model)
                st.pyplot(fig)
                plt.close(fig)
            
            if show_probability and n_classes == 2:
                st.write("### Probability Surface")
                fig = plot_probability_surface_2d(X, y, model)
                st.pyplot(fig)
                plt.close(fig)
    
    with col2:
        st.write("### Feature Importance")
        fig = plot_feature_importance()
        st.pyplot(fig)
        plt.close(fig)

# Show prediction demo
st.write("### Try a Prediction")
cols = st.columns(n_features)
test_point = []
for i, col in enumerate(cols):
    with col:
        value = st.slider(f"Feature {i+1}", 
                         float(X[:, i].min()), 
                         float(X[:, i].max()),
                         float(X[:, i].mean()))
        test_point.append(value)

test_point = np.array(test_point).reshape(1, -1)
prediction = model.predict(test_point)
probabilities = model.predict_proba(test_point)

st.write(f"Predicted Class: {prediction[0]}")
st.write("### Class Probabilities:")
prob_df = pd.DataFrame({
    'Class': range(n_classes),
    'Probability': probabilities[0]
})
st.write(prob_df)

# Best use cases and references
st.markdown("""
### When to Use Naive Bayes?

#### Best Use Cases:
1. **Text Classification**
   - Spam detection
   - Document categorization
   - Sentiment analysis

2. **Small Datasets**
   - Medical diagnosis
   - Risk assessment
   - Quick prototyping

3. **Real-time Prediction**
   - Streaming data
   - Online learning
   - Real-time classification

#### Advantages:
- Fast training and prediction
- Works well with high dimensions
- Needs less training data
- Handles missing values well
- Easy to understand

#### Limitations:
- Assumes feature independence
- Can be outperformed by modern methods
- Sensitive to feature correlations
- "Zero frequency" problem
- Assumes normal distribution (Gaussian NB)

### Real-World Applications:

1. **Text Processing**
   - Email filtering
   - News categorization
   - Language detection

2. **Medical Diagnosis**
   - Disease prediction
   - Risk assessment
   - Patient classification

3. **Real-time Systems**
   - Face detection
   - Speech recognition
   - Recommendation systems

### Learn More:
1. "Introduction to Statistical Learning" - Naive Bayes Chapter
2. Scikit-learn Naive Bayes Documentation
3. "Machine Learning" by Tom Mitchell
""")

import streamlit as st

# Interactive Quiz
st.subheader("Test Your Understanding of Naïve Bayes")

# Question 1
quiz_answer_1 = st.radio(
    "What is the key assumption made by the Naïve Bayes classifier?",
    (
        "All features are independent given the class label.",
        "All features have equal importance in classification.",
        "The dataset must be normally distributed."
    )
)

if quiz_answer_1 == "All features are independent given the class label.":
    st.success("Correct! Naïve Bayes assumes conditional independence, meaning that each feature contributes independently to the probability of a class.")
else:
    st.error("Not quite. The key assumption in Naïve Bayes is that features are independent given the class label, though this may not always hold in real-world data.")

# Question 2
quiz_answer_2 = st.radio(
    "Which of the following is a major advantage of Naïve Bayes?",
    (
        "It performs well with small datasets and high-dimensional data.",
        "It always provides 100% accuracy in classification problems.",
        "It does not require labeled training data."
    )
)

if quiz_answer_2 == "It performs well with small datasets and high-dimensional data.":
    st.success("Correct! Naïve Bayes is computationally efficient and works well even with small datasets and many features.")
else:
    st.error("Not quite. While Naïve Bayes is fast and effective, it does not guarantee perfect accuracy and requires labeled training data for supervised learning.")
