import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.data_generators import generate_svm_data
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Support Vector Machine Visualization", page_icon="⚔️", layout="wide")
st.title("Support Vector Machine (SVM) ⚔️")

# Educational content specific to SVM
st.markdown("""
### What is a Support Vector Machine (SVM)?
An SVM is like a smart boundary drawer that tries to separate different classes with the widest possible street 
between them. Imagine you're organizing apples and oranges on a table:

1. **The Margin Street**: SVM creates the widest possible street between the fruits
2. **Support Vectors**: The fruits closest to the street are the most important ones (support vectors)
3. **The Kernel Trick**: When fruits can't be separated by a straight street, SVM can bend and transform the street!

### Key Concepts:
1. **Maximum Margin**
   - SVM finds the boundary with largest possible gap between classes
   - Larger margin = More confident predictions = Better generalization

2. **Support Vectors**
   - Points closest to the decision boundary
   - Only these points matter for making predictions
   - Fewer support vectors = Simpler model

3. **Kernel Trick**
   - Transform data into higher dimensions where it becomes separable
   - Common kernels: Linear, RBF (Radial Basis Function), Polynomial
   - Like viewing the data from a different angle
""")

# Parameters explanation in the sidebar
st.sidebar.header("SVM Parameters")

st.sidebar.markdown("**Kernel Type**")
st.sidebar.markdown("""
How to transform the data:
- Linear: Straight line boundary
- RBF: Flexible curved boundary
""")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"])

st.sidebar.markdown("**Margin Strength (C)**")
st.sidebar.markdown("""
How strict should the boundary be:
- Higher: Focus on separating training points
- Lower: Focus on wider margin
""")
C = st.sidebar.slider("C", 0.1, 10.0, 1.0)

if kernel == "rbf":
    st.sidebar.markdown("**RBF Kernel Width (gamma)**")
    st.sidebar.markdown("""
    How flexible should the boundary be:
    - Higher: More curved boundary
    - Lower: More straight boundary
    """)
    gamma = st.sidebar.slider("Gamma", 0.1, 10.0, 1.0)
else:
    gamma = 1.0

n_points = st.sidebar.slider("Number of Data Points", 50, 200, 100)

# Initialize session state
if 'svm_losses' not in st.session_state:
    st.session_state.svm_losses = []
if 'support_vectors' not in st.session_state:
    st.session_state.support_vectors = None

# Generate synthetic data
@st.cache_data
def get_svm_data(n_points):
    return generate_svm_data(n_points)

X, y = get_svm_data(n_points)

# SVM Model (Using PyTorch for real-time visualization)
class SVM(nn.Module):
    def __init__(self, kernel='linear', gamma=1.0):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.linear = nn.Linear(2 if kernel == 'linear' else n_points, 1)
        
    def rbf_kernel(self, x1, x2):
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff**2, dim=2))
    
    def forward(self, x):
        if self.kernel == 'linear':
            return self.linear(x)
        else:  # rbf
            if not hasattr(self, 'support_vectors'):
                self.support_vectors = x
            K = self.rbf_kernel(x, self.support_vectors)
            return torch.mm(K, self.linear.weight[0].unsqueeze(1)) + self.linear.bias

# Create model and optimizer
model = SVM(kernel=kernel, gamma=gamma)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Custom hinge loss
def hinge_loss(outputs, labels, C=1.0):
    margin = 1 - outputs * (2 * labels - 1)
    loss = torch.mean(torch.clamp(margin, min=0))
    # Add L2 regularization
    l2_reg = torch.sum(model.linear.weight ** 2)
    return loss + (1 / (2 * C)) * l2_reg

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Create three columns for visualizations
col1, col2, col3 = st.columns(3)

def plot_decision_boundary(model, X, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions
    X_grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(X_grid).numpy()
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.4, 
                colors=['#FF9999', '#FFFFFF', '#99FF99'])
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'green'],
               linestyles=['--', '-', '--'])
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
    
    # Highlight support vectors if available
    if st.session_state.support_vectors is not None:
        ax.scatter(st.session_state.support_vectors[:, 0], 
                  st.session_state.support_vectors[:, 1],
                  s=100, linewidth=1, facecolors='none', 
                  edgecolor='black')
    
    ax.set_title(f'Decision Boundary ({kernel.upper()} Kernel)')
    return fig

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(losses)
    ax.set_title('Hinge Loss Over Time')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    return fig

def plot_margin_distribution(model, X, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    with torch.no_grad():
        margins = 1 - model(X_tensor).numpy() * (2 * y - 1)
    
    sns.histplot(data=margins, bins=30, ax=ax, legend=False)
    ax.axvline(x=1, color='r', linestyle='--')
    ax.axvline(x=-1, color='r', linestyle='--')
    ax.set_title('Margin Distribution')
    ax.set_xlabel('Distance to Margin')
    ax.set_ylabel('Count')
    return fig

# Create placeholder containers
with col1:
    decision_plot = st.empty()
with col2:
    loss_plot = st.empty()
with col3:
    margin_plot = st.empty()

# Training controls
epochs = st.slider("Number of epochs to train", 1, 100, 20)
train_button = st.button("Train Model")

# Status containers
progress_bar = st.empty()
status_text = st.empty()

if train_button:
    st.session_state.svm_losses = []
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = hinge_loss(outputs, y_tensor, C)
        loss.backward()
        optimizer.step()
        
        # Store loss
        st.session_state.svm_losses.append(loss.item())
        
        # Find support vectors
        with torch.no_grad():
            margins = 1 - outputs * (2 * y_tensor - 1)
            sv_indices = margins.squeeze() > -0.1
            st.session_state.support_vectors = X[sv_indices]
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        
        # Calculate accuracy
        predictions = (outputs >= 0).float()
        accuracy = (predictions == y_tensor).float().mean()
        status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy.item():.4f}')
        
        # Update plots
        decision_plot.pyplot(plot_decision_boundary(model, X, y))
        loss_plot.pyplot(plot_loss(st.session_state.svm_losses))
        margin_plot.pyplot(plot_margin_distribution(model, X, y))
        plt.close('all')

# Initial plots
if not train_button:
    decision_plot.pyplot(plot_decision_boundary(model, X, y))
    loss_plot.pyplot(plot_loss([1] if not st.session_state.svm_losses else st.session_state.svm_losses))
    margin_plot.pyplot(plot_margin_distribution(model, X, y))
    plt.close('all')

# Best use cases and references
st.markdown("""
### When to Use SVM?

#### Best Use Cases:
1. **High-Dimensional Data**
   - Text classification
   - Gene expression analysis
   - Image classification

2. **Clear Margin of Separation Needed**
   - Medical diagnosis
   - Face detection
   - Handwriting recognition

3. **Complex Non-linear Problems**
   - Protein structure prediction
   - Weather forecasting
   - Financial time series

#### Advantages:
- Effective in high dimensional spaces
- Memory efficient (only support vectors matter)
- Versatile (different kernel functions)
- Good generalization

#### Limitations:
- Can be slow on large datasets
- Sensitive to feature scaling
- Kernel selection can be tricky
- Not directly probabilistic

### Real-World Applications:

1. **Bioinformatics**
   - Protein fold classification
   - Gene classification
   - Cancer detection

2. **Computer Vision**
   - Face detection
   - Object recognition
   - Character recognition

3. **Text Classification**
   - Spam detection
   - Sentiment analysis
   - Document categorization

### Learn More:
1. "Introduction to Statistical Learning" - Chapter on SVMs
2. Coursera Machine Learning Course by Andrew Ng
3. "Understanding Support Vector Machines" by MIT OpenCourseWare
""")

import streamlit as st

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary objective of a Support Vector Machine (SVM)?",
    (
        "To minimize the number of support vectors.",
        "To maximize the margin between different classes.",
        "To cluster data points into different groups."
    )
)

if quiz_answer_1 == "To maximize the margin between different classes.":
    st.success("Correct! SVM aims to find the optimal hyperplane that maximizes the margin between different classes.")
else:
    st.error("Not quite. SVM focuses on maximizing the margin between classes to improve generalization.")

# Question 2
quiz_answer_2 = st.radio(
    "What role does the kernel trick play in SVM?",
    (
        "It allows SVM to work with non-linearly separable data by transforming it into a higher-dimensional space.",
        "It speeds up the training process by reducing the number of support vectors.",
        "It eliminates the need for feature scaling in SVM."
    )
)

if quiz_answer_2 == "It allows SVM to work with non-linearly separable data by transforming it into a higher-dimensional space.":
    st.success("Correct! The kernel trick helps map data into a higher-dimensional space where it becomes linearly separable.")
else:
    st.error("Not quite. The kernel trick is used to transform non-linearly separable data into a higher-dimensional space for better classification.")
