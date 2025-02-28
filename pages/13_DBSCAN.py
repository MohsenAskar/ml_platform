import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

st.set_page_config(page_title="DBSCAN Clustering Visualization", page_icon="🌟", layout="wide")
st.title("Density-Based Spatial Clustering (DBSCAN) 🌟")

# Educational content
st.markdown("""
### What is DBSCAN?
DBSCAN finds clusters by looking at the density of points:

1. **Key Concepts**
   - Core Points: Points with many neighbors
   - Border Points: Points near core points
   - Noise Points: Isolated points
   - Epsilon (ε): Neighborhood radius
   - MinPts: Minimum points for core status

2. **How it Works**
   - Find core points by counting neighbors
   - Connect core points within ε distance
   - Add border points to their nearest clusters
   - Label remaining points as noise

3. **Advantages**
   - Finds clusters of any shape
   - Handles noise automatically
   - No need to specify number of clusters
""")

# Parameters explanation in the sidebar
st.sidebar.header("DBSCAN Parameters")

st.sidebar.markdown("**Epsilon (ε)**")
st.sidebar.markdown("""
Radius of neighborhood:
- Larger: Bigger neighborhoods, fewer clusters
- Smaller: Smaller neighborhoods, more clusters
""")
epsilon = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5)

st.sidebar.markdown("**Minimum Points**")
st.sidebar.markdown("""
Points needed for core status:
- Higher: Denser clusters required
- Lower: More points become core points
""")
min_pts = st.sidebar.slider("Minimum Points", 2, 15, 5)

# Data generation parameters
dataset_type = st.sidebar.selectbox("Dataset Type", 
    ["Moons", "Circles", "Blobs", "Random"])
n_points = st.sidebar.slider("Number of Points", 50, 300, 200)

# Generate synthetic data
@st.cache_data
def generate_data(dataset_type, n_points):
    if dataset_type == "Moons":
        X, _ = make_moons(n_samples=n_points, noise=0.1)
    elif dataset_type == "Circles":
        X, _ = make_circles(n_samples=n_points, noise=0.1, factor=0.5)
    elif dataset_type == "Blobs":
        X, _ = make_blobs(n_samples=n_points, centers=3, cluster_std=0.6)
    else:  # Random
        X = np.random.randn(n_points, 2)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

# DBSCAN implementation
def find_neighbors(X, point_idx, epsilon):
    distances = np.sqrt(((X - X[point_idx]) ** 2).sum(axis=1))
    return np.where(distances <= epsilon)[0]

def dbscan_step(X, epsilon, min_pts):
    n_points = len(X)
    labels = np.full(n_points, -1)  # -1 for unvisited
    
    # Find core points
    core_points = []
    border_points = []
    noise_points = []
    
    for i in range(n_points):
        neighbors = find_neighbors(X, i, epsilon)
        if len(neighbors) >= min_pts:
            core_points.append(i)
        elif len(neighbors) > 1:
            border_points.append(i)
        else:
            noise_points.append(i)
    
    return np.array(core_points), np.array(border_points), np.array(noise_points)

def plot_dbscan_step(X, core_points, border_points, noise_points, epsilon=None, 
                     highlight_point=None, highlight_neighbors=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all points initially as noise (gray)
    ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, label='Unassigned')
    
    # Plot core points
    if len(core_points) > 0:
        ax.scatter(X[core_points, 0], X[core_points, 1], 
                  c='blue', label='Core Points')
    
    # Plot border points
    if len(border_points) > 0:
        ax.scatter(X[border_points, 0], X[border_points, 1], 
                  c='green', label='Border Points')
    
    # Plot noise points
    if len(noise_points) > 0:
        ax.scatter(X[noise_points, 0], X[noise_points, 1], 
                  c='red', label='Noise Points')
    
    # Highlight specific point and its neighborhood if provided
    if highlight_point is not None:
        ax.scatter(X[highlight_point, 0], X[highlight_point, 1], 
                  c='yellow', s=200, alpha=0.5, label='Selected Point')
        
        if epsilon is not None:
            circle = plt.Circle((X[highlight_point, 0], X[highlight_point, 1]), 
                              epsilon, fill=False, linestyle='--', color='black')
            ax.add_artist(circle)
        
        if highlight_neighbors is not None:
            ax.scatter(X[highlight_neighbors, 0], X[highlight_neighbors, 1], 
                      c='orange', s=100, alpha=0.5, label='Neighbors')
    
    ax.set_title('DBSCAN Clustering Process')
    ax.legend()
    return fig

def plot_distance_distribution(X):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(X)):
        dist = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        distances.extend(dist)
    
    # Plot distance distribution
    sns.histplot(distances, bins=50, ax=ax)
    ax.set_title('Distribution of Pairwise Distances')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    
    return fig

# Create containers for the two main rows
row1 = st.container()
row2 = st.container()

# Generate data
X = generate_data(dataset_type, n_points)

# Allow point selection for neighborhood visualization
st.write("### Point Selection")
col1, col2 = st.columns(2)
with col1:
    point_x = st.slider("X coordinate", float(X[:, 0].min()), 
                       float(X[:, 0].max()), float(X[:, 0].mean()))
with col2:
    point_y = st.slider("Y coordinate", float(X[:, 1].min()), 
                       float(X[:, 1].max()), float(X[:, 1].mean()))

# Find closest point in dataset
selected_point = np.array([point_x, point_y])
distances = np.sqrt(((X - selected_point) ** 2).sum(axis=1))
highlight_point = np.argmin(distances)
highlight_neighbors = find_neighbors(X, highlight_point, epsilon)

# Run DBSCAN
core_points, border_points, noise_points = dbscan_step(X, epsilon, min_pts)

with row1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Clustering Visualization")
        fig = plot_dbscan_step(X, core_points, border_points, noise_points, 
                              epsilon, highlight_point, highlight_neighbors)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Distance Distribution")
        fig = plot_distance_distribution(X)
        st.pyplot(fig)
        plt.close(fig)

# Show cluster statistics
st.write("### Cluster Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Core Points: {len(core_points)}")
with col2:
    st.write(f"Border Points: {len(border_points)}")
with col3:
    st.write(f"Noise Points: {len(noise_points)}")

# Show point analysis
st.write("### Selected Point Analysis")
neighbors_count = len(highlight_neighbors) - 1  # subtract self
is_core = neighbors_count >= min_pts
is_border = not is_core and any(p in core_points for p in highlight_neighbors)
is_noise = not (is_core or is_border)

st.write(f"Number of neighbors within ε: {neighbors_count}")
st.write(f"Point type: {'Core' if is_core else 'Border' if is_border else 'Noise'}")

# Best use cases and references
st.markdown("""
### When to Use DBSCAN?

#### Best Use Cases:
1. **Irregular Shaped Clusters**
   - Geographic clustering
   - Image segmentation
   - Anomaly detection

2. **Noise Handling**
   - Sensor data analysis
   - Signal processing
   - Outlier detection

3. **Unknown Number of Clusters**
   - Market segmentation
   - Social network analysis
   - Pattern recognition

#### Advantages:
- No predefined number of clusters
- Finds arbitrarily shaped clusters
- Robust to noise
- No assumption about cluster shapes
- Handles outliers well

#### Limitations:
- Sensitive to parameters
- Struggles with varying densities
- Memory intensive
- Not suitable for high dimensions
- Cannot handle clusters of varying density

### Real-World Applications:

1. **Geographic Information Systems**
   - Hotspot detection
   - Region clustering
   - Traffic analysis

2. **Anomaly Detection**
   - Fraud detection
   - Network intrusion
   - Quality control

3. **Scientific Applications**
   - Astronomy (galaxy clustering)
   - Biology (gene expression)
   - Climate zone identification

### Learn More:
1. Scikit-learn DBSCAN Documentation
2. "Introduction to Data Mining" - Clustering Chapter
3. DBSCAN Original Paper by Ester et al.
""")


# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is a key advantage of DBSCAN compared to K-Means?",
    (
        "DBSCAN does not require specifying the number of clusters in advance.",
        "DBSCAN always produces spherical clusters.",
        "DBSCAN is faster than K-Means for large datasets."
    )
)

if quiz_answer_1 == "DBSCAN does not require specifying the number of clusters in advance.":
    st.success("Correct! Unlike K-Means, DBSCAN determines the number of clusters automatically based on data density.")
else:
    st.error("Not quite. DBSCAN does not assume spherical clusters and is often slower than K-Means, but its key advantage is not requiring a predefined number of clusters.")

# Question 2
quiz_answer_2 = st.radio(
    "How does DBSCAN classify points in a dataset?",
    (
        "By assigning points to predefined clusters based on centroid distances.",
        "By labeling points as core, border, or noise based on density.",
        "By using hierarchical merging of data points."
    )
)

if quiz_answer_2 == "By labeling points as core, border, or noise based on density.":
    st.success("Correct! DBSCAN groups points based on density and categorizes them into core, border, or noise points.")
else:
    st.error("Not quite. DBSCAN does not use centroids or hierarchical merging but relies on density-based clustering.")
