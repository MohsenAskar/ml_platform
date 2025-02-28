import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="K-Nearest Neighbors Visualization", page_icon="🎯", layout="wide")
st.title("K-Nearest Neighbors (KNN) 🎯")

# Educational content
st.markdown("""
### What is K-Nearest Neighbors (KNN)?
KNN is like asking your closest friends for advice:

1. **The Concept**
   - Look at K nearest training points
   - Take a vote among these neighbors
   - Majority class wins!

2. **Key Parameters**
   - K: Number of neighbors to consider
   - Distance Metric: How to measure closeness
   - Weights: Equal vote or closer neighbors matter more?

3. **Why KNN is Special**
   - No training phase (lazy learning)
   - Very intuitive
   - Learns complex patterns
""")

# Parameters explanation in the sidebar
st.sidebar.header("KNN Parameters")

st.sidebar.markdown("**Number of Neighbors (K)**")
st.sidebar.markdown("""
How many neighbors to consider:
- Larger K: Smoother boundaries, more robust
- Smaller K: More complex boundaries, might overfit
""")
n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)

st.sidebar.markdown("**Distance Metric**")
distance_metric = st.sidebar.selectbox("Distance Metric", 
    ["euclidean", "manhattan", "chebyshev"])

st.sidebar.markdown("**Weights**")
weights = st.sidebar.selectbox("Neighbor Weights", 
    ["uniform", "distance"])

# Data parameters
n_samples = st.sidebar.slider("Number of Training Points", 50, 200, 100)

# Generate synthetic data
@st.cache_data
def generate_data(n_samples):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                             n_informative=2, random_state=1,
                             n_clusters_per_class=1)
    return X, y

# Create the KNN classifier
def create_knn(n_neighbors, metric, weights):
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, 
                               weights=weights)

def plot_decision_boundary(X, y, model, test_point=None, neighbors=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Predict for all mesh points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4)
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
    
    # If test point provided, plot it and its neighbors
    if test_point is not None:
        ax.scatter(test_point[0], test_point[1], c='yellow', s=200, marker='*',
                  label='Test Point', edgecolor='black')
        
        if neighbors is not None:
            # Draw lines to neighbors
            for neighbor in neighbors:
                ax.plot([test_point[0], X[neighbor, 0]], 
                       [test_point[1], X[neighbor, 1]], 
                       'k--', alpha=0.3)
            # Highlight neighbors
            ax.scatter(X[neighbors, 0], X[neighbors, 1], s=100, 
                      facecolor='none', edgecolor='black', linewidth=2,
                      label='Neighbors')
    
    ax.set_title('Decision Boundary and Nearest Neighbors')
    ax.legend()
    return fig

def plot_distance_influence(X, y, test_point, neighbors, distances, weights):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distance vs influence
    if weights == 'uniform':
        influences = np.ones(len(neighbors)) / len(neighbors)
    else:  # 'distance'
        influences = 1 / (distances + 1e-6)
        influences /= influences.sum()
    
    bars = ax.bar(range(len(neighbors)), influences)
    
    # Color bars by class
    for i, neighbor in enumerate(neighbors):
        bars[i].set_color(plt.cm.RdYlBu(y[neighbor]))
    
    ax.set_xlabel('Neighbor Index')
    ax.set_ylabel('Influence Weight')
    ax.set_title('Neighbor Influence on Classification')
    return fig

# Create containers for the two main rows
row1 = st.container()
row2 = st.container()

# Generate data
X, y = generate_data(n_samples)

# Create classifier
knn = create_knn(n_neighbors, distance_metric, weights)
knn.fit(X, y)

# Allow user to select test point
st.write("### Test Point Selection")
col1, col2 = st.columns(2)
with col1:
    test_x = st.slider("X coordinate", float(X[:, 0].min()), 
                      float(X[:, 0].max()), float(X[:, 0].mean()))
with col2:
    test_y = st.slider("Y coordinate", float(X[:, 1].min()), 
                      float(X[:, 1].max()), float(X[:, 1].mean()))

test_point = np.array([test_x, test_y])

# Get neighbors and distances
distances, neighbors = knn.kneighbors([test_point])
neighbors = neighbors[0]
distances = distances[0]

# Make prediction
prediction = knn.predict([test_point])[0]
proba = knn.predict_proba([test_point])[0]

with row1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Decision Boundary and Neighbors")
        fig = plot_decision_boundary(X, y, knn, test_point, neighbors)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Neighbor Influence")
        fig = plot_distance_influence(X, y, test_point, neighbors, distances, weights)
        st.pyplot(fig)
        plt.close(fig)

# Show prediction details
st.write("### Prediction Details")
st.write(f"Predicted Class: {prediction}")
st.write(f"Probability Distribution: Class 0: {proba[0]:.3f}, Class 1: {proba[1]:.3f}")

# Show neighbor details
st.write("### Neighbor Analysis")
neighbor_df = pd.DataFrame({
    'Neighbor': range(len(neighbors)),
    'Class': y[neighbors],
    'Distance': distances,
    'Weight': 1/distances if weights == 'distance' else 1/len(neighbors)
})
st.write(neighbor_df)

# Best use cases and references
st.markdown("""
### When to Use KNN?

#### Best Use Cases:
1. **Small to Medium Datasets**
   - Medical diagnosis
   - Recommendation systems
   - Pattern recognition

2. **Feature-Rich Data**
   - Image classification
   - Document categorization
   - Sensor data analysis

3. **Non-linear Problems**
   - When decision boundaries are complex
   - When patterns are irregular
   - When local patterns matter

#### Advantages:
- Simple to understand
- No training phase
- Works with any number of classes
- Naturally handles multi-class
- Non-parametric (no assumptions)

#### Limitations:
- Slow for large datasets
- Sensitive to irrelevant features
- Needs feature scaling
- Memory intensive
- Curse of dimensionality

### Real-World Applications:

1. **Recommender Systems**
   - Product recommendations
   - Movie suggestions
   - Music playlists

2. **Pattern Recognition**
   - Handwriting recognition
   - Face recognition
   - Speech recognition

3. **Medical Diagnosis**
   - Disease classification
   - Patient similarity
   - Treatment recommendation

### Learn More:
1. Scikit-learn KNN Documentation
2. "Introduction to Statistical Learning" - KNN Chapter
3. Stanford CS231n KNN Tutorial
""")


# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary principle behind the K-Nearest Neighbors (KNN) algorithm?",
    (
        "It predicts a data point’s label based on the majority class of its nearest neighbors.",
        "It creates a decision boundary using hyperplanes.",
        "It reduces the dimensionality of data before classification."
    )
)

if quiz_answer_1 == "It predicts a data point’s label based on the majority class of its nearest neighbors.":
    st.success("Correct! KNN classifies a data point based on the most common label among its K closest neighbors.")
else:
    st.error("Not quite. KNN does not use hyperplanes like SVM nor perform dimensionality reduction like PCA. It relies on majority voting from nearby points.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a major drawback of the KNN algorithm?",
    (
        "It requires extensive training before making predictions.",
        "It is computationally expensive at prediction time, especially for large datasets.",
        "It cannot be used for regression tasks."
    )
)

if quiz_answer_2 == "It is computationally expensive at prediction time, especially for large datasets.":
    st.success("Correct! Since KNN must compute distances for every new query point, it can be slow for large datasets.")
else:
    st.error("Not quite. KNN has minimal training time but can be slow during prediction. Also, it can be used for both classification and regression.")
