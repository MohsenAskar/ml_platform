import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils.data_generators import generate_classification_data

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Random Forest Visualization",
    page_icon="🌲",
    layout="wide"
)

# -----------------------------
# Title & Educational Content
# -----------------------------
st.title("Random Forest (RF) 🌲🌲🌲")

st.markdown("""
### What is a Random Forest?
Think of a Random Forest as a group of decision-making trees working together, like a committee of experts:

1. **Multiple Trees**
   - Each tree is trained on a random subset of data.
   - Each tree looks at different features.
   - More trees = More stable predictions.

2. **Voting System**
   - Each tree makes its own prediction.
   - The final decision is made by majority vote.
   - Like asking multiple experts and taking the most common answer.

3. **Random Sampling**
   - Each tree sees a different random sample of data (bagging).
   - This makes the forest robust and prevents overfitting.
   - Like each expert having different experiences.

### How Does It Work?
1. Create multiple decision trees.
2. Each tree:
   - Gets random subset of data.
   - Considers random subset of features at each split.
   - Makes its own prediction.
3. Combine predictions by voting.
""")

# -----------------------------
# Sidebar Parameters
# -----------------------------
st.sidebar.header("Random Forest Parameters")

st.sidebar.markdown("**Number of Trees**")
st.sidebar.markdown("""
How many trees in the forest:
- More trees = More stable but slower
- Fewer trees = Faster but less reliable
""")
n_trees = st.sidebar.slider("Number of Trees", 1, 10, 3)

st.sidebar.markdown("**Max Tree Depth**")
st.sidebar.markdown("""
How deep each tree can grow:
- Deeper = More complex patterns but might overfit
- Shallower = Simpler patterns, more generalization
""")
max_depth = st.sidebar.slider("Max Depth", 1, 5, 3)

st.sidebar.markdown("**Sample Ratio**")
st.sidebar.markdown("""
Percentage of data each tree sees:
- Higher = More stable predictions
- Lower = More diverse trees
""")
sample_ratio = st.sidebar.slider("Sample Ratio", 0.1, 1.0, 0.8)

st.sidebar.markdown("**Number of Data Points**")
n_points = st.sidebar.slider("Number of Data Points", 50, 200, 100)

st.sidebar.markdown("**Random Seed**")
seed = st.sidebar.number_input("Random Seed (for reproducibility)", min_value=0, value=42, step=1)

# -----------------------------
# Session State Initialization
# -----------------------------
if "trees" not in st.session_state:
    st.session_state.trees = []
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = np.zeros(2, dtype=float)

# -----------------------------
# Data Generation (Cached)
# -----------------------------
@st.cache_data
def get_data(n_points, seed=42):
    """
    Generate synthetic 2D classification data using the provided
    random seed for reproducibility.
    """
    np.random.seed(seed)
    return generate_classification_data(n_points)

X, y = get_data(n_points, seed=seed)

# -----------------------------
# Plotting Utilities
# -----------------------------
def plot_decision_boundary_with_samples(
    tree,
    X,
    y,
    samples_used=None,
    title="Decision Boundary"
):
    """
    Plots the decision boundary of a single decision tree,
    highlighting the sampled points used to train that tree.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Get predicted probabilities across the grid
    Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")

    # Plot all points in light color
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", alpha=0.1)

    # Highlight sampled points (if provided)
    if samples_used is not None:
        ax.scatter(
            X[samples_used][:, 0],
            X[samples_used][:, 1],
            c=y[samples_used],
            cmap="RdYlBu",
            edgecolor="black"
        )

    ax.set_title(title)
    return fig


def plot_feature_importance(importances):
    """
    Bar plot of feature importance values for two features (X1, X2).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    features = ["X1", "X2"]
    sns.barplot(x=features, y=importances, ax=ax)
    ax.set_title("Feature Importance")
    ax.set_ylabel("Importance Score")
    ax.set_ylim(0, max(importances.max() * 1.2, 0.1))  # scale for nicer visualization
    return fig


def plot_combined_prediction(trees, X, y, test_point):
    """
    Show:
    1) Each individual tree's predicted probability for 'test_point'
    2) The ensemble decision boundary + test point.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------
    # (1) Individual Predictions Bar Plot
    # ------------------
    predictions = []
    for tree in trees:
        pred_prob = tree.predict_proba([test_point])[0][1]
        predictions.append(pred_prob)

    ax1.bar(range(len(trees)), predictions, alpha=0.7)
    ax1.axhline(y=np.mean(predictions), color="r", linestyle="-", label="Ensemble Average")
    ax1.axhline(y=0.5, color="g", linestyle="--", label="Decision Threshold")
    ax1.set_xlabel("Tree Index")
    ax1.set_ylabel("Probability of Class 1")
    ax1.set_title("Individual Tree Predictions")
    ax1.legend()

    # ------------------
    # (2) Ensemble Boundary w/ Test Point
    # ------------------
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Average predicted probability from all trees
    Z = np.zeros(xx.shape)
    for tree in trees:
        Z += tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    Z /= len(trees)

    ax2.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolor="black", alpha=0.6)
    ax2.scatter(
        test_point[0], test_point[1],
        c="yellow", s=200, marker="*",
        label="Test Point", edgecolor="black"
    )
    ax2.set_title("Ensemble Prediction")
    ax2.legend()

    plt.tight_layout()
    return fig


# -----------------------------
# Main App Logic
# -----------------------------
train_button = st.button("Train Forest")

if train_button:
    # Reset stored state
    st.session_state.trees = []
    st.session_state.feature_importance = np.zeros(2, dtype=float)

    # Container or placeholders to show intermediate results
    training_container = st.container()
    tree_structure_container = st.container()

    with training_container:
        st.write("### Training Each Tree in the Forest")

    # Train each tree one at a time
    for i in range(n_trees):
        # Randomly sample data (bagging approach)
        n_samples = int(sample_ratio * len(X))
        np.random.seed(seed + i)  # Different seed for each tree, but reproducible
        indices = np.random.choice(len(X), n_samples, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        # Train a single Decision Tree
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed + i)
        tree.fit(X_sample, y_sample)

        # Save it in session state
        st.session_state.trees.append(tree)
        st.session_state.feature_importance += tree.feature_importances_

        # Show training results
        with training_container:
            colA, colB = st.columns([1, 1.2])
            with colA:
                st.subheader(f"Tree {i+1}")
                fig_dt = plot_decision_boundary_with_samples(
                    tree=tree,
                    X=X,
                    y=y,
                    samples_used=indices,
                    title=f"Tree {i+1} Decision Boundary"
                )
                st.pyplot(fig_dt)
                plt.close(fig_dt)

            with colB:
                st.subheader("Current Feature Importance")
                avg_importance = st.session_state.feature_importance / (i + 1)
                fig_imp = plot_feature_importance(avg_importance)
                st.pyplot(fig_imp)
                plt.close(fig_imp)

        # Display tree structure
        with tree_structure_container:
            st.write(f"### Tree {i+1} Structure")
            fig_tree, ax_tree = plt.subplots(figsize=(15, 6))
            plot_tree(
                tree,
                filled=True,
                feature_names=["X1", "X2"],
                class_names=["0", "1"],
                ax=ax_tree
            )
            st.pyplot(fig_tree)
            plt.close(fig_tree)

    # After finishing training all trees:
    st.write("---")
    st.write("### Test the Ensemble")

    # Let user pick a test point
    colA, colB = st.columns(2)
    with colA:
        x_test = st.slider(
            "Test Point X", 
            float(X[:, 0].min()), 
            float(X[:, 0].max()), 
            float(X[:, 0].mean())
        )
    with colB:
        y_test = st.slider(
            "Test Point Y", 
            float(X[:, 1].min()), 
            float(X[:, 1].max()), 
            float(X[:, 1].mean())
        )

    test_point = np.array([x_test, y_test])

    st.write("### Ensemble Prediction Process")
    fig_ensemble = plot_combined_prediction(
        trees=st.session_state.trees, 
        X=X, 
        y=y, 
        test_point=test_point
    )
    st.pyplot(fig_ensemble)
    plt.close(fig_ensemble)

    # Final prediction
    predictions = [
        tree.predict_proba([test_point])[0][1]
        for tree in st.session_state.trees
    ]
    final_pred = np.mean(predictions)
    pred_class = "1" if final_pred > 0.5 else "0"

    st.write(f"**Final Ensemble Prediction:**")
    st.write(f"- Average probability for class 1: `{final_pred:.3f}`")
    st.write(f"- Final class prediction: **{pred_class}**")

# -----------------------------
# Info & References
# -----------------------------
st.markdown("""
---
### When to Use Random Forests?

#### Best Use Cases:
1. **Complex Classification Problems**
   - Customer segmentation
   - Image classification
   - Medical diagnosis

2. **Feature Selection**
   - Understanding important variables
   - Reducing data dimensionality
   - Feature ranking

3. **Balanced Performance**
   - When you need good accuracy
   - When you want to avoid overfitting
   - When you have noisy data

#### Advantages:
- Handles both numerical and categorical data
- Provides feature importance
- Less prone to overfitting
- Works well with unbalanced data
- No need for feature scaling

#### Limitations:
- More of a "black box" than a single decision tree
- Can be computationally expensive
- May require more memory for large datasets
- Not the best for strictly linear relationships

### Real-World Applications:

1. **Finance**
   - Credit risk assessment
   - Stock price prediction
   - Fraud detection

2. **Healthcare**
   - Disease prediction
   - Patient risk stratification
   - Drug response prediction

3. **Marketing**
   - Customer behavior prediction
   - Campaign response prediction
   - Market segmentation

### Learn More:
1. "Introduction to Statistical Learning" - Chapter on Tree-Based Methods
2. Scikit-learn Random Forest [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary advantage of using a Random Forest over a single Decision Tree?",
    (
        "It reduces overfitting by averaging multiple trees.",
        "It requires fewer computational resources.",
        "It always achieves 100% accuracy."
    )
)

if quiz_answer_1 == "It reduces overfitting by averaging multiple trees.":
    st.success("Correct! Random Forests combine multiple Decision Trees to improve generalization and reduce overfitting.")
else:
    st.error("Not quite. While Random Forests may improve accuracy, their main advantage is reducing overfitting by aggregating multiple trees.")

# Question 2
quiz_answer_2 = st.radio(
    "How does Random Forest introduce randomness in the model?",
    (
        "By using different machine learning algorithms for each tree.",
        "By training each tree on a random subset of the data and selecting a random subset of features at each split.",
        "By using a single Decision Tree with random weights."
    )
)

if quiz_answer_2 == "By training each tree on a random subset of the data and selecting a random subset of features at each split.":
    st.success("Correct! Random Forest employs bagging (bootstrap sampling) and feature randomness to build diverse trees and improve robustness.")
else:
    st.error("Not quite. Random Forest introduces randomness through bootstrapping and feature selection, not by using different algorithms or a single tree.")
