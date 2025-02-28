import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs, load_digits
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import pandas as pd

st.set_page_config(page_title="UMAP Visualization", page_icon="🗺️", layout="wide")
st.title("Uniform Manifold Approximation and Projection (UMAP) 🗺️")

# Educational content
st.markdown("""
### What is UMAP?
UMAP is a modern dimensionality reduction technique that helps visualize high-dimensional data:

1. **Key Concepts**
   - Preserves both local and global structure
   - Based on manifold learning and topology
   - Faster than t-SNE
   - Better global structure preservation

2. **Parameters**
   - n_neighbors: Local structure size
   - min_dist: Minimum distance between points
   - n_components: Output dimensions
   - metric: Distance measure

3. **Advantages**
   - Fast computation
   - Preserves global structure
   - Handles large datasets
   - Works with various data types
""")

# Parameters explanation in the sidebar
st.sidebar.header("UMAP Parameters")

st.sidebar.markdown("**Number of Neighbors**")
st.sidebar.markdown("""
Size of local neighborhood:
- Larger: More global structure
- Smaller: More local structure
""")
n_neighbors = st.sidebar.slider("n_neighbors", 2, 100, 15)

st.sidebar.markdown("**Minimum Distance**")
st.sidebar.markdown("""
Minimum distance between points:
- Larger: More spread out
- Smaller: Tighter clusters
""")
min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1)

# Data parameters
dataset_type = st.sidebar.selectbox("Dataset Type", 
    ["Swiss Roll", "Blobs", "Digits"])
n_samples = st.sidebar.slider("Number of Samples", 500, 2000, 1000)

# Generate/load data
@st.cache_data
def get_data(dataset_type, n_samples):
    if dataset_type == "Swiss Roll":
        X, color = make_swiss_roll(n_samples=n_samples, random_state=42)
        feature_names = [f"Dimension {i+1}" for i in range(X.shape[1])]
    elif dataset_type == "Blobs":
        X, color = make_blobs(n_samples=n_samples, n_features=10, 
                            centers=5, random_state=42)
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    else:  # Digits
        digits = load_digits()
        X = digits.data
        color = digits.target
        feature_names = [f"Pixel {i+1}" for i in range(X.shape[1])]
        if n_samples < len(X):
            idx = np.random.choice(len(X), n_samples, replace=False)
            X = X[idx]
            color = color[idx]
    
    return X, color, feature_names

# Function to perform dimensionality reduction
def reduce_dimensions(X, method, **params):
    if method == "UMAP":
        reducer = umap.UMAP(**params)
    elif method == "PCA":
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2)
    
    start_time = time.time()
    X_reduced = reducer.fit_transform(X)
    computation_time = time.time() - start_time
    
    return X_reduced, computation_time

def plot_projection(X_reduced, color, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                        c=color, cmap='Spectral')
    plt.colorbar(scatter)
    ax.set_title(title)
    return fig

def plot_feature_correlations(X, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = np.corrcoef(X.T)
    sns.heatmap(corr, xticklabels=False, yticklabels=False, 
                cmap='RdBu', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    return fig

# Create containers for visualization
row1 = st.container()
row2 = st.container()

# Load and process data
X, color, feature_names = get_data(dataset_type, n_samples)
X_scaled = StandardScaler().fit_transform(X)

# Compute projections
with st.spinner("Computing projections..."):
    umap_params = {
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'n_components': 2,
        'random_state': 42
    }
    
    X_umap, umap_time = reduce_dimensions(X_scaled, "UMAP", **umap_params)
    X_pca, pca_time = reduce_dimensions(X_scaled, "PCA")
    X_tsne, tsne_time = reduce_dimensions(X_scaled, "t-SNE")

# Display visualizations
with row1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### UMAP Projection")
        fig = plot_projection(X_umap, color, f'UMAP ({umap_time:.2f}s)')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### PCA Projection")
        fig = plot_projection(X_pca, color, f'PCA ({pca_time:.2f}s)')
        st.pyplot(fig)
        plt.close(fig)
    
    with col3:
        st.write("### t-SNE Projection")
        fig = plot_projection(X_tsne, color, f't-SNE ({tsne_time:.2f}s)')
        st.pyplot(fig)
        plt.close(fig)

with row2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Feature Correlations")
        fig = plot_feature_correlations(X_scaled, feature_names)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Dimension Reduction Quality")
        
        # Calculate and display quality metrics
        from sklearn.metrics import pairwise_distances
        
        # Sample points for efficiency
        sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X_sample = X_scaled[sample_idx]
        
        # Calculate distance matrices
        orig_dist = pairwise_distances(X_sample)
        umap_dist = pairwise_distances(X_umap[sample_idx])
        
        # Calculate correlation between distance matrices
        corr = np.corrcoef(orig_dist.flatten(), umap_dist.flatten())[0, 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(orig_dist.flatten(), umap_dist.flatten(), alpha=0.1)
        ax.set_xlabel('Original Distances')
        ax.set_ylabel('UMAP Distances')
        ax.set_title(f'Distance Preservation (correlation: {corr:.3f})')
        st.pyplot(fig)
        plt.close(fig)

# Display timing comparison
st.write("### Performance Comparison")
times = {
    'UMAP': umap_time,
    'PCA': pca_time,
    't-SNE': tsne_time
}
st.write(pd.DataFrame({'Method': list(times.keys()), 
                      'Time (seconds)': list(times.values())}))

# Best use cases and references
st.markdown("""
### When to Use UMAP?

#### Best Use Cases:
1. **High-Dimensional Data Visualization**
   - Single-cell RNA sequencing
   - Image datasets
   - Text embeddings

2. **Dimensionality Reduction**
   - Feature extraction
   - Data preprocessing
   - Visualization

3. **Pattern Discovery**
   - Cluster visualization
   - Anomaly detection
   - Structure exploration

#### Advantages:
- Faster than t-SNE
- Better global structure preservation
- Scalable to large datasets
- Theoretical foundations in topology
- Preserves both local and global structure

#### Limitations:
- Non-deterministic results
- Parameter sensitivity
- Less interpretable than PCA
- No inverse transform
- Not for general dimensionality reduction

### Real-World Applications:

1. **Bioinformatics**
   - Gene expression analysis
   - Protein structure visualization
   - Cell type identification

2. **Computer Vision**
   - Image embedding visualization
   - Feature space analysis
   - Dataset exploration

3. **Natural Language Processing**
   - Document clustering
   - Word embeddings
   - Topic modeling

### Learn More:
1. UMAP Paper by McInnes et al.
2. UMAP Documentation and Tutorial
3. "How UMAP Works" by the author
""")



# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary goal of UMAP?",
    (
        "To reduce the dimensionality of data while preserving local and global structure.",
        "To cluster data points into different groups.",
        "To improve the accuracy of classification models."
    )
)

if quiz_answer_1 == "To reduce the dimensionality of data while preserving local and global structure.":
    st.success("Correct! UMAP is a dimensionality reduction technique that maintains both local and global structures of high-dimensional data.")
else:
    st.error("Not quite. UMAP is mainly used for dimensionality reduction, focusing on preserving meaningful structures in the data.")

# Question 2
quiz_answer_2 = st.radio(
    "How does UMAP compare to t-SNE in terms of performance?",
    (
        "UMAP is generally faster and scales better to large datasets than t-SNE.",
        "UMAP always produces better clustering than t-SNE.",
        "UMAP requires more computational resources than t-SNE."
    )
)

if quiz_answer_2 == "UMAP is generally faster and scales better to large datasets than t-SNE.":
    st.success("Correct! UMAP is computationally efficient and better suited for large datasets compared to t-SNE.")
else:
    st.error("Not quite. While both methods have strengths, UMAP is preferred for its speed and scalability.")
