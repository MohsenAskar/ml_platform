import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_moons, make_circles, make_blobs
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Decision Tree Visualization", page_icon="🌲", layout="wide")
st.title("Decision Trees (DT) 🌲")

# Educational content
st.markdown("""
### What is a Decision Tree?
A Decision Tree is like a flowchart for making decisions:

1. **Key Concepts**
   - Root Node: Starting point
   - Internal Nodes: Decision points
   - Leaf Nodes: Final predictions
   - Splitting Criteria: How to make decisions

2. **How it Works**
   - Start at root node
   - Answer yes/no questions
   - Follow path based on answers
   - Reach a prediction at leaf

3. **Splitting Criteria**
   - Gini Impurity
   - Entropy
   - Information Gain
""")

# Parameters explanation in the sidebar
st.sidebar.header("Decision Tree Parameters")

st.sidebar.markdown("**Tree Depth**")
st.sidebar.markdown("""
How many levels of decisions:
- Deeper = More complex decisions
- Shallower = More generalization
""")
max_depth = st.sidebar.slider("Maximum Depth", 1, 10, 3)

st.sidebar.markdown("**Split Criterion**")
criterion = st.sidebar.selectbox("Splitting Criterion", 
    ["gini", "entropy"])

st.sidebar.markdown("**Minimum Samples per Split**")
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

# Data parameters
dataset_type = st.sidebar.selectbox("Dataset Type", 
    ["Moons", "Circles", "Blobs"])
n_samples = st.sidebar.slider("Number of Samples", 50, 200, 100)

# Generate synthetic data
@st.cache_data
def generate_data(dataset_type, n_samples):
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2)
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=0.2)
    else:  # Blobs
        X, y = make_blobs(n_samples=n_samples, centers=2)
    return X, y

X, y = generate_data(dataset_type, n_samples)

def plot_tree_growth(X, y, max_depth):
    depths = range(1, max_depth + 1)
    n_depths = len(depths)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, depth in enumerate(depths):
        if i < len(axes):
            clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion,
                                       min_samples_split=min_samples_split)
            clf.fit(X, y)
            
            # Plot decision boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[i].contourf(xx, yy, Z, alpha=0.4)
            axes[i].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            axes[i].set_title(f'Depth {depth}')
    
    # Hide unused subplots
    for i in range(n_depths, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(tree):
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = pd.DataFrame({
        'feature': ['X1', 'X2'],
        'importance': tree.feature_importances_
    })
    sns.barplot(data=importance, x='feature', y='importance', ax=ax)
    ax.set_title('Feature Importance')
    return fig

def plot_node_samples(X, y, tree):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def compute_node_depths(tree):
        """Compute the depth of each node in the tree."""
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [(0, 0)]  # Start with root node and depth 0
        
        while len(stack) > 0:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            
            # If internal node, add children to stack
            if children_left[node_id] != -1:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
        
        return node_depth
    
    def get_node_samples(tree):
        n_nodes = tree.tree_.node_count
        depths = compute_node_depths(tree)
        samples = []
        depths_list = []
        
        for i in range(n_nodes):
            if tree.tree_.children_left[i] == -1:  # leaf node
                samples.append(tree.tree_.n_node_samples[i])
                depths_list.append(depths[i])
        return samples, depths_list
    
    samples, depths = get_node_samples(tree)
    
    # Create scatter plot with size proportional to samples
    for depth in range(max(depths) + 1):
        depth_samples = [s for s, d in zip(samples, depths) if d == depth]
        if depth_samples:
            ax.scatter([depth] * len(depth_samples), depth_samples,
                      s=100, alpha=0.5, label=f'Depth {depth}')
    
    ax.set_xlabel('Tree Depth')
    ax.set_ylabel('Samples in Node')
    ax.set_title('Sample Distribution across Tree Depth')
    ax.legend()
    
    return fig

# Create main tree
tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                             min_samples_split=min_samples_split)
tree.fit(X, y)

# Create containers for visualization
row1 = st.container()
row2 = st.container()
row3 = st.container()

with row1:
    st.write("### Tree Structure")
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(tree, feature_names=['X1', 'X2'], 
              class_names=['0', '1'], filled=True, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

with row2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Feature Importance")
        fig = plot_feature_importance(tree)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("### Node Sample Distribution")
        fig = plot_node_samples(X, y, tree)
        st.pyplot(fig)
        plt.close(fig)

with row3:
    st.write("### Tree Growth Process")
    fig = plot_tree_growth(X, y, max_depth)
    st.pyplot(fig)
    plt.close(fig)

# Interactive prediction
st.write("### Try a Prediction")
col1, col2 = st.columns(2)
with col1:
    x1 = st.slider("X1", float(X[:, 0].min()), float(X[:, 0].max()),
                   float(X[:, 0].mean()))
with col2:
    x2 = st.slider("X2", float(X[:, 1].min()), float(X[:, 1].max()),
                   float(X[:, 1].mean()))

test_point = np.array([[x1, x2]])
prediction = tree.predict(test_point)
probabilities = tree.predict_proba(test_point)

st.write(f"Predicted Class: {prediction[0]}")
st.write("Class Probabilities:")
prob_df = pd.DataFrame({
    'Class': [0, 1],
    'Probability': probabilities[0]
})
st.write(prob_df)

# Show decision path
path = tree.decision_path(test_point)
st.write("### Decision Path")
path_indices = path.indices
node_features = []
for idx in path_indices:
    if idx == 0:
        node_features.append("Root")
    elif tree.tree_.feature[idx] != -2:  # not a leaf
        feature = f"X{tree.tree_.feature[idx] + 1}"
        threshold = tree.tree_.threshold[idx]
        value = test_point[0][tree.tree_.feature[idx]]
        if value <= threshold:
            decision = "<="
        else:
            decision = ">"
        node_features.append(f"{feature} {decision} {threshold:.2f}")
    else:
        node_features.append("Leaf")

st.write(" → ".join(node_features))

# Best use cases and references
st.markdown("""
### When to Use Decision Trees?

#### Best Use Cases:
1. **Interpretable Models Needed**
   - Medical diagnosis
   - Risk assessment
   - Rule-based decisions

2. **Mixed Data Types**
   - Categorical and numerical features
   - Missing values
   - Non-linear relationships

3. **Feature Importance Analysis**
   - Variable selection
   - Feature ranking
   - Understanding data structure

#### Advantages:
- Easy to understand and interpret
- Requires little data preparation
- Handles numerical and categorical data
- Can be visualized
- Implicitly performs feature selection

#### Limitations:
- Can create overly complex trees
- Can overfit easily
- Unstable (small changes = different tree)
- Biased toward dominant classes
- May create biased trees if classes unbalanced

### Real-World Applications:

1. **Healthcare**
   - Disease diagnosis
   - Treatment decisions
   - Risk assessment

2. **Finance**
   - Credit approval
   - Fraud detection
   - Investment decisions

3. **Customer Service**
   - Troubleshooting guides
   - Customer segmentation
   - Response automation

### Learn More:
1. "Introduction to Statistical Learning" - Decision Trees Chapter
2. Scikit-learn Decision Trees Documentation
3. "The Elements of Statistical Learning" - Tree Methods
""")

import streamlit as st

# Interactive Quiz
st.subheader("Test Your Understanding of Decision Trees")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary criterion used by Decision Trees to split nodes?",
    (
        "Random selection of features.",
        "Maximizing the depth of the tree.",
        "Minimizing impurity in the split."
    )
)

if quiz_answer_1 == "Minimizing impurity in the split.":
    st.success("Correct! Decision Trees split nodes based on criteria like Gini impurity or entropy to create the most informative partitions.")
else:
    st.error("Not quite. Decision Trees aim to minimize impurity using measures like Gini or entropy to improve classification accuracy.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a common drawback of Decision Trees?",
    (
        "They cannot handle categorical variables.",
        "They are prone to overfitting, especially on small datasets.",
        "They require a deep understanding of neural networks to implement."
    )
)

if quiz_answer_2 == "They are prone to overfitting, especially on small datasets.":
    st.success("Correct! Decision Trees tend to overfit when they are too deep, capturing noise instead of patterns.")
else:
    st.error("Not quite. While Decision Trees can handle categorical variables and do not require neural networks, they often overfit without pruning or proper tuning.")
