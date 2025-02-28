import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
import seaborn as sns

st.set_page_config(page_title="K-means Clustering Visualization", page_icon="🎯", layout="wide")
st.title("K-means Clustering 🎯")

# Educational content
st.markdown("""
### What is K-means Clustering?
K-means is like organizing colored balls into buckets based on their positions:

1. **The Process**
   - Start with K random centroids (bucket positions)
   - Assign each point to nearest centroid
   - Move centroids to center of their points
   - Repeat until stable

2. **Key Concepts**
   - K: Number of clusters you want
   - Centroids: Centers of clusters
   - Assignment: Points belong to nearest centroid
   - Update: Centroids move to average position

3. **Convergence**
   - Algorithm stops when centroids stop moving
   - Or when maximum iterations reached
""")

# Parameters explanation in the sidebar
st.sidebar.header("K-means Parameters")

st.sidebar.markdown("**Number of Clusters (K)**")
st.sidebar.markdown("""
How many groups to create:
- More clusters = More detailed grouping
- Fewer clusters = More general grouping
""")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)

st.sidebar.markdown("**Data Generation**")
dataset_type = st.sidebar.selectbox("Dataset Type", 
    ["Blobs", "Moons", "Circles"])

st.sidebar.markdown("**Number of Points**")
n_points = st.sidebar.slider("Number of Points", 50, 300, 200)

st.sidebar.markdown("**Animation Speed**")
speed = st.sidebar.slider("Delay between iterations (ms)", 500, 2000, 1000)

# Generate synthetic data
@st.cache_data
def generate_data(dataset_type, n_points, n_clusters):
    if dataset_type == "Blobs":
        X, _ = make_blobs(n_samples=n_points, n_features=2, 
                         centers=n_clusters, random_state=42)
    elif dataset_type == "Moons":
        X, _ = make_moons(n_samples=n_points, noise=0.1, random_state=42)
    else:  # Circles
        X, _ = make_circles(n_samples=n_points, noise=0.1, random_state=42)
    return X

# K-means implementation
def kmeans_step(X, centroids):
    # Assign points to nearest centroid
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    
    # Update centroids
    new_centroids = np.array([X[labels == k].mean(axis=0) 
                             for k in range(len(centroids))])
    
    # Check for empty clusters and reinitialize them
    empty_clusters = np.isnan(new_centroids).any(axis=1)
    new_centroids[empty_clusters] = X[np.random.choice(len(X), 
                                    empty_clusters.sum(), replace=False)]
    
    return new_centroids, labels

def plot_kmeans_step(X, centroids, labels=None, iteration=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if labels is not None:
        # Plot points with cluster colors
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                           alpha=0.6)
    else:
        # Plot initial points
        scatter = ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', 
               s=200, label='Centroids')
    
    if iteration is not None:
        ax.set_title(f'Iteration {iteration}')
    
    ax.legend()
    return fig

def plot_inertia(X, centroids, labels):
    # Calculate distances to centroids
    distances = np.sqrt(((X - centroids[labels])**2).sum(axis=1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of distances
    sns.histplot(distances, bins=30, ax=ax)
    ax.axvline(distances.mean(), color='r', linestyle='--', 
               label='Mean Distance')
    ax.set_title('Distribution of Distances to Centroids')
    ax.set_xlabel('Distance to Centroid')
    ax.set_ylabel('Count')
    ax.legend()
    
    return fig

# Create containers for the two main rows
row1 = st.container()
row2 = st.container()

# Generate data
X = generate_data(dataset_type, n_points, n_clusters)

# Training controls
train_button = st.button("Run K-means")

if train_button:
    # Initialize centroids randomly
    initial_centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    centroids = initial_centroids.copy()
    
    with row1:
        col1, col2 = st.columns(2)
        
        # Show initial state
        with col1:
            st.write("### Clustering Progress")
            progress_plot = st.empty()
            fig = plot_kmeans_step(X, centroids)
            progress_plot.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("### Centroid Movement")
            movement_plot = st.empty()
            
            # Initialize centroid movement tracking
            centroid_history = [centroids.copy()]
    
    # Main loop
    iteration = 0
    max_iter = 10
    
    while iteration < max_iter:
        # Perform one step
        new_centroids, labels = kmeans_step(X, centroids)
        
        # Check convergence
        if np.all(np.abs(new_centroids - centroids) < 1e-4):
            break
        
        centroids = new_centroids
        centroid_history.append(centroids.copy())
        iteration += 1
        
        # Update visualization
        with col1:
            fig = plot_kmeans_step(X, centroids, labels, iteration)
            progress_plot.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Plot centroid movement
            fig, ax = plt.subplots(figsize=(10, 6))
            for k in range(n_clusters):
                path = np.array([ch[k] for ch in centroid_history])
                ax.plot(path[:, 0], path[:, 1], 'r.-', alpha=0.5)
                ax.scatter(path[-1, 0], path[-1, 1], c='red', marker='*', s=200)
            ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.1)
            ax.set_title('Centroid Movement History')
            movement_plot.pyplot(fig)
            plt.close(fig)
        
        # Add delay
        st.empty().info(f"Iteration {iteration}")
        import time
        time.sleep(speed/1000)
    
    # Show final analysis
    with row2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Cluster Analysis")
            # Show cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(n_clusters), counts)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Points')
            ax.set_title('Cluster Sizes')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("### Distance Distribution")
            fig = plot_inertia(X, centroids, labels)
            st.pyplot(fig)
            plt.close(fig)
    
    # Final metrics
    inertia = np.sum((X - centroids[labels])**2)
    st.write(f"### Final Metrics")
    st.write(f"Total Inertia (Sum of squared distances): {inertia:.2f}")
    st.write(f"Number of iterations: {iteration + 1}")

# Best use cases and references
st.markdown("""
### When to Use K-means?

#### Best Use Cases:
1. **Customer Segmentation**
   - Group customers by behavior
   - Market basket analysis
   - Target marketing

2. **Image Processing**
   - Color quantization
   - Image segmentation
   - Feature clustering

3. **Data Preprocessing**
   - Feature aggregation
   - Data binning
   - Dimensionality reduction

#### Advantages:
- Simple to understand
- Fast for small datasets
- Guaranteed to converge
- Works well with globular clusters
- Easy to implement

#### Limitations:
- Needs predefined K
- Sensitive to outliers
- Assumes spherical clusters
- May converge to local optimum
- Struggles with varying cluster sizes

### Real-World Applications:

1. **Marketing**
   - Customer segmentation
   - Product categorization
   - Campaign targeting

2. **Computer Vision**
   - Image compression
   - Object detection
   - Color palette generation

3. **Document Analysis**
   - Topic clustering
   - Document categorization
   - Feature extraction

### Learn More:
1. Scikit-learn K-means Documentation
2. "Introduction to Statistical Learning" - Clustering Chapter
3. Stanford CS231n K-means Tutorial
""")

import streamlit as st

# Interactive Quiz
st.subheader("Test Your Understanding of K-Means Clustering")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary objective of the K-Means clustering algorithm?",
    (
        "To divide data into K clusters by minimizing intra-cluster variance.",
        "To find hierarchical relationships between data points.",
        "To predict a target variable based on labeled data."
    )
)

if quiz_answer_1 == "To divide data into K clusters by minimizing intra-cluster variance.":
    st.success("Correct! K-Means partitions data into K clusters by minimizing the sum of squared distances between points and their cluster centroids.")
else:
    st.error("Not quite. K-Means is an unsupervised clustering algorithm that minimizes intra-cluster variance, rather than predicting labels or finding hierarchical relationships.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a common limitation of K-Means clustering?",
    (
        "It requires specifying the number of clusters (K) in advance.",
        "It does not work with numerical data.",
        "It guarantees finding the globally optimal clustering solution."
    )
)

if quiz_answer_2 == "It requires specifying the number of clusters (K) in advance.":
    st.success("Correct! One of K-Means' limitations is that the number of clusters (K) must be chosen beforehand, which can be challenging.")
else:
    st.error("Not quite. K-Means works with numerical data but does not guarantee a globally optimal solution due to its reliance on random initialization.")
