import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Hierarchical Clustering Visualization", page_icon="🌳", layout="wide")
st.title("Hierarchical Clustering 📂")

# Educational content
st.markdown("""
### What is Hierarchical Clustering?
Hierarchical clustering builds a tree of clusters:

1. **Two Main Approaches**
   - Agglomerative (bottom-up): Start with points as clusters, then merge
   - Divisive (top-down): Start with one cluster, then split

2. **Key Concepts**
   - Dendrogram: Tree showing cluster hierarchy
   - Linkage: How to measure cluster distances
   - Distance threshold: Where to cut the tree

3. **Linkage Methods**
   - Single: Minimum distance between clusters
   - Complete: Maximum distance between clusters
   - Average: Mean distance between clusters
   - Ward: Minimize variance within clusters
""")

# Parameters explanation in the sidebar
st.sidebar.header("Hierarchical Clustering Parameters")

st.sidebar.markdown("**Dataset Type**")
dataset_type = st.sidebar.selectbox("Select Dataset", 
    ["Blobs", "Moons", "Circles"])

st.sidebar.markdown("**Linkage Method**")
linkage_method = st.sidebar.selectbox("Select Linkage", 
    ["single", "complete", "average", "ward"])

st.sidebar.markdown("**Number of Samples**")
n_samples = st.sidebar.slider("Number of Samples", 20, 200, 50)

st.sidebar.markdown("**Number of Clusters**")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)

# Generate synthetic data
@st.cache_data
def generate_data(dataset_type, n_samples):
    if dataset_type == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                         random_state=42)
    elif dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1)
    else:  # Circles
        X, y = make_moons(n_samples=n_samples, noise=0.1)
        # Transform to circles
        angles = np.linspace(0, 2*np.pi, n_samples)
        radius = 1 + (y * 0.5)
        X = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    return X, y

X, y_true = generate_data(dataset_type, n_samples)

def plot_dendrogram(X, method='single'):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Compute linkage matrix
    Z = linkage(X, method=method)
    # Plot dendrogram
    dendrogram(Z, ax=ax)
    ax.set_title(f'Dendrogram ({method} linkage)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    return fig

def plot_clustering_process(X, n_clusters_range, method='single'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, k in enumerate(n_clusters_range):
        clustering = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = clustering.fit_predict(X)
        
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[i].set_title(f'{k} Clusters')
    
    fig.suptitle(f'Clustering Process ({method} linkage)', y=1.02)
    plt.tight_layout()
    return fig

def plot_cluster_distances(X, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate distances between points in each cluster
    distances = []
    cluster_labels = []
    
    for i in range(max(labels) + 1):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            # Calculate pairwise distances within cluster
            for j in range(len(cluster_points)):
                for k in range(j + 1, len(cluster_points)):
                    dist = np.linalg.norm(cluster_points[j] - cluster_points[k])
                    distances.append(dist)
                    cluster_labels.append(f'Cluster {i}')
    
    # Create violin plot
    if distances:
        sns.violinplot(x=cluster_labels, y=distances, ax=ax)
        ax.set_title('Distribution of Intra-cluster Distances')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Distance')
    
    return fig

# Create main display areas
row1 = st.container()
row2 = st.container()

# Perform clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
labels = clustering.fit_predict(X)

with row1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Dendrogram")
        fig = plot_dendrogram(X, method=linkage_method)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Final Clustering")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        ax.set_title(f'Hierarchical Clustering Results\n({linkage_method} linkage)')
        st.pyplot(fig)
        plt.close(fig)

with row2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Clustering Process")
        n_clusters_range = [2, 3, 4, 5, 6, 7]
        fig = plot_clustering_process(X, n_clusters_range, linkage_method)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Cluster Analysis")
        fig = plot_cluster_distances(X, labels)
        st.pyplot(fig)
        plt.close(fig)

# Show cluster statistics
st.write("### Cluster Statistics")
cluster_stats = pd.DataFrame({
    'Cluster': range(n_clusters),
    'Size': [sum(labels == i) for i in range(n_clusters)],
    'Average Distance to Center': [
        np.mean([np.linalg.norm(x - np.mean(X[labels == i], axis=0)) 
                for x in X[labels == i]]) 
        for i in range(n_clusters)
    ]
})
st.write(cluster_stats)

# Best use cases and references
st.markdown("""
### When to Use Hierarchical Clustering?

#### Best Use Cases:
1. **Small to Medium Datasets**
   - Taxonomies
   - Customer segmentation
   - Document organization

2. **Need Hierarchy Understanding**
   - Biological classification
   - Organization structures
   - Topic hierarchies

3. **Unknown Number of Clusters**
   - Exploratory data analysis
   - Pattern discovery
   - Natural grouping detection

#### Advantages:
- No need to specify clusters upfront
- Provides hierarchical view
- Dendrogram visualization
- Multiple clustering levels
- More deterministic than k-means

#### Limitations:
- Computationally intensive
- Memory intensive
- Not suitable for large datasets
- Can't undo steps
- Sensitive to noise

### Real-World Applications:

1. **Biology**
   - Gene clustering
   - Species taxonomy
   - Protein structure analysis

2. **Business**
   - Market segmentation
   - Document classification
   - Customer hierarchy

3. **Social Network Analysis**
   - Community detection
   - Interest grouping
   - Relationship mapping

### Learn More:
1. "Introduction to Statistical Learning" - Clustering Chapter
2. Scikit-learn Hierarchical Clustering Documentation
3. "Data Mining" by Tan, Steinbach, Kumar
""")


# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is a key characteristic of Hierarchical Clustering?",
    (
        "It does not require specifying the number of clusters in advance.",
        "It always forms clusters of the same size.",
        "It assigns data points to clusters randomly."
    )
)

if quiz_answer_1 == "It does not require specifying the number of clusters in advance.":
    st.success("Correct! Unlike K-Means, Hierarchical Clustering builds a tree-like structure (dendrogram) that allows for flexible cluster selection.")
else:
    st.error("Not quite. Hierarchical Clustering does not create equally sized clusters or assign data randomly; it builds a hierarchy based on distance measures.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a major drawback of Hierarchical Clustering?",
    (
        "It is computationally expensive for large datasets.",
        "It cannot be used for categorical data.",
        "It always requires a predefined number of clusters."
    )
)

if quiz_answer_2 == "It is computationally expensive for large datasets.":
    st.success("Correct! Hierarchical Clustering has a high time complexity (O(n²) or worse), making it inefficient for large datasets.")
else:
    st.error("Not quite. Hierarchical Clustering can handle categorical data and does not require a predefined number of clusters, but it is computationally intensive.")
