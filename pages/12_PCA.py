import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Principal Component Analysis Visualization", page_icon="📊", layout="wide")
st.title("Principal Component Analysis (PCA) 📊")

# Educational content
st.markdown("""
### What is Principal Component Analysis (PCA)?
PCA is like finding the best camera angles to photograph a 3D object in 2D:

1. **Dimension Reduction**
   - Finds the most informative viewpoints (principal components)
   - Keeps the most important patterns in data
   - Reduces noise and redundancy

2. **Principal Components**
   - Directions where data varies the most
   - Ordered by importance (explained variance)
   - Orthogonal (perpendicular) to each other

3. **Data Transformation**
   - Centers the data (subtract mean)
   - Optionally scales features
   - Projects onto new directions
""")

# Parameters explanation in the sidebar
st.sidebar.header("PCA Parameters")

st.sidebar.markdown("**Number of Components**")
st.sidebar.markdown("""
How many dimensions to keep:
- More = Retain more information
- Fewer = More reduction
""")
n_components = st.sidebar.slider("Number of Components", 1, 3, 2)

st.sidebar.markdown("**Data Generation**")
data_type = st.sidebar.selectbox("Data Type", 
    ["Linear", "Circular", "S-Curve"])
n_points = st.sidebar.slider("Number of Points", 50, 500, 200)

# Generate synthetic data
@st.cache_data
def generate_data(data_type, n_points):
    if data_type == "Linear":
        # Generate correlated data
        x = np.random.normal(0, 1, n_points)
        noise = np.random.normal(0, 0.1, n_points)
        y = 2*x + noise
        z = -0.5*x + noise
        return np.column_stack([x, y, z])
    
    elif data_type == "Circular":
        # Generate data in a spiral
        t = np.linspace(0, 10, n_points)
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = t + np.random.normal(0, 0.1, n_points)
        return np.column_stack([x, y, z])
    
    else:  # S-Curve
        t = np.linspace(-8, 8, n_points)
        x = t
        y = np.sin(t)
        z = np.sign(t) * (np.cos(t) - 1)
        return np.column_stack([x, y, z])

# Generate and preprocess data
X = generate_data(data_type, n_points)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create containers for the two main rows
row1 = st.container()
row2 = st.container()

def plot_3d_data(X, pca=None, components=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', alpha=0.6)
    
    if pca is not None and components is not None:
        # Plot principal components
        mean = np.zeros(3)
        for i in range(components):
            direction = pca.components_[i]
            ax.quiver(mean[0], mean[1], mean[2],
                     direction[0], direction[1], direction[2],
                     color=['r', 'g', 'purple'][i],
                     length=3, label=f'PC{i+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    return fig

def plot_explained_variance(pca):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scree plot
    variance_ratio = pca.explained_variance_ratio_
    ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    
    # Cumulative variance
    cumulative_variance_ratio = np.cumsum(variance_ratio)
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'bo-')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_transformed_data(X_original, X_pca, n_components):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original first two dimensions
    axes[0].scatter(X_original[:, 0], X_original[:, 1], alpha=0.6)
    axes[0].set_title('Original Data (First 2 Dimensions)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # PCA transformed
    axes[1].scatter(X_pca[:, 0], 
                   X_pca[:, 1] if n_components > 1 else np.zeros(len(X_pca)), 
                   alpha=0.6)
    axes[1].set_title('PCA Transformed Data')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2' if n_components > 1 else '')
    
    plt.tight_layout()
    return fig

# Display visualizations
with row1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Original Data with Principal Components")
        fig = plot_3d_data(X_scaled, pca, n_components)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Explained Variance Analysis")
        fig = plot_explained_variance(pca)
        st.pyplot(fig)
        plt.close(fig)

with row2:
    st.write("### Data Transformation")
    fig = plot_transformed_data(X_scaled, X_pca, n_components)
    st.pyplot(fig)
    plt.close(fig)

# Show variance explanation
total_var = pca.explained_variance_ratio_[:n_components].sum()
st.write(f"### Variance Explained by {n_components} Components: {total_var:.2%}")

# Component interpretation
st.write("### Principal Component Interpretation")
for i in range(n_components):
    st.write(f"**PC{i+1} Direction:**")
    components_df = pd.DataFrame({
        'Feature': ['X', 'Y', 'Z'],
        'Weight': pca.components_[i]
    })
    st.write(components_df)

# Best use cases and references
st.markdown("""
### When to Use PCA?

#### Best Use Cases:
1. **Dimensionality Reduction**
   - High-dimensional data visualization
   - Feature extraction
   - Data compression

2. **Preprocessing**
   - Before applying other ML algorithms
   - Dealing with multicollinearity
   - Noise reduction

3. **Data Analysis**
   - Finding patterns in data
   - Understanding feature relationships
   - Data exploration

#### Advantages:
- Reduces dimensionality
- Removes multicollinearity
- Helps with visualization
- Reduces noise in data
- No hyperparameters to tune

#### Limitations:
- Only captures linear relationships
- May lose interpretability
- Sensitive to scaling
- May not preserve important patterns

### Real-World Applications:

1. **Image Processing**
   - Face recognition (Eigenfaces)
   - Image compression
   - Feature extraction

2. **Finance**
   - Portfolio optimization
   - Risk analysis
   - Market analysis

3. **Biology**
   - Gene expression analysis
   - Population genetics
   - Biological signal processing

### Learn More:
1. "Elements of Statistical Learning" - Chapter on PCA
2. Scikit-learn PCA Documentation
3. StatQuest PCA Videos on YouTube
""")



# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary purpose of Principal Component Analysis (PCA)?",
    (
        "To reduce the dimensionality of data while preserving as much variance as possible.",
        "To improve the accuracy of a classification model.",
        "To cluster similar data points together."
    )
)

if quiz_answer_1 == "To reduce the dimensionality of data while preserving as much variance as possible.":
    st.success("Correct! PCA transforms the data into a lower-dimensional space by selecting principal components that capture the most variance.")
else:
    st.error("Not quite. PCA is mainly used for dimensionality reduction, not necessarily for classification accuracy or clustering.")

# Question 2
quiz_answer_2 = st.radio(
    "What do principal components represent in PCA?",
    (
        "New uncorrelated variables that capture the most variance in the data.",
        "The original features of the dataset, but rearranged in a different order.",
        "Clusters of similar data points."
    )
)

if quiz_answer_2 == "New uncorrelated variables that capture the most variance in the data.":
    st.success("Correct! Principal components are linear combinations of the original features that maximize variance and reduce redundancy.")
else:
    st.error("Not quite. Principal components are new variables created by transforming the original features to capture the most variance in the data.")
