import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.data_generators import generate_classification_data
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

st.set_page_config(page_title="Logistic Regression Visualization", page_icon="🎯", layout="wide")
st.title("Logistic Regression 🎯")

# Educational content specific to Logistic Regression
st.markdown("""
### What is Logistic Regression?
Despite its name, Logistic Regression is actually used for classification problems, not regression! 
It's perfect for questions with Yes/No answers, like:
- Will this customer buy the product? (Yes/No)
- Is this email spam? (Yes/No)
- Will this student pass the exam? (Yes/No)

### How Does It Work?
1. **The S-Curve (Sigmoid Function)**
   - Takes any number and squeezes it between 0 and 1
   - This gives us a probability estimate
   - Example: 0.8 means 80% chance of being "Yes"

2. **Decision Making**
   - Usually use 0.5 as the threshold
   - Above 0.5 → Class 1 ("Yes")
   - Below 0.5 → Class 0 ("No")

### Key Concepts
- **Probability Output**: Unlike Linear Regression, outputs are probabilities between 0 and 1
- **Decision Boundary**: The line that separates the two classes
- **Classification Threshold**: The probability cutoff for making decisions (default 0.5)
""")

# Parameters explanation in the sidebar
st.sidebar.header("Model Parameters")

st.sidebar.markdown("**Learning Rate**")
st.sidebar.markdown("""
How big steps to take when adjusting the model:
- Higher: Faster learning but might overshoot
- Lower: Slower but more precise learning
""")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)

st.sidebar.markdown("**Decision Threshold**")
st.sidebar.markdown("""
Probability cutoff for classification:
- Above threshold → Class 1
- Below threshold → Class 0
""")
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5)

st.sidebar.markdown("**Data Separation**")
st.sidebar.markdown("How distinct should the two classes be?")
separation = st.sidebar.slider("Class Separation", 0.1, 2.0, 1.0)

n_points = st.sidebar.slider("Number of Data Points", 50, 200, 100)

# Initialize session state
if 'logistic_losses' not in st.session_state:
    st.session_state.logistic_losses = []
if 'confusion_matrix' not in st.session_state:
    st.session_state.confusion_matrix = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Generate synthetic data
@st.cache_data
def get_classification_data(n_points, separation):
    X, y = generate_classification_data(n_points, noise=1/separation)
    return X, y

X, y = get_classification_data(n_points, separation)

# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Create model and optimizer
model = LogisticRegressionModel()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Create columns for visualizations
col1, col2, col3 = st.columns(3)

def plot_decision_boundary(model, X, y, threshold=0.5):
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
    
    # Plot decision boundary and probability contours
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, levels=[threshold], colors='k', linestyles='--')
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    
    ax.set_title('Decision Boundary and Probability Contours')
    plt.colorbar(scatter, label='True Class')
    return fig

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(losses)
    ax.set_title('Binary Cross-Entropy Loss Over Time')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (BCE)')
    ax.set_yscale('log')
    return fig

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    ax.set_title('Confusion Matrix')
    return fig

# Create placeholder containers
with col1:
    decision_plot = st.empty()
with col2:
    loss_plot = st.empty()
with col3:
    confusion_plot = st.empty()

# Training controls
epochs = st.slider("Number of epochs to train", 1, 100, 20)
train_button = st.button("Train Model")

# Status containers
progress_bar = st.empty()
status_text = st.empty()
metrics_text = st.empty()

if train_button:
    st.session_state.logistic_losses = []
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Store loss
        st.session_state.logistic_losses.append(loss.item())
        
        # Get predictions and compute confusion matrix
        with torch.no_grad():
            predictions = (outputs >= threshold).float()
            conf_matrix = confusion_matrix(y_tensor, predictions)
            st.session_state.confusion_matrix = conf_matrix
            st.session_state.predictions = predictions
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        
        # Calculate accuracy and other metrics
        accuracy = (predictions == y_tensor).float().mean()
        status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy.item():.4f}')
        
        # Update plots
        decision_plot.pyplot(plot_decision_boundary(model, X, y, threshold))
        loss_plot.pyplot(plot_loss(st.session_state.logistic_losses))
        confusion_plot.pyplot(plot_confusion_matrix(conf_matrix))
        plt.close('all')

# Initial plots
if not train_button:
    decision_plot.pyplot(plot_decision_boundary(model, X, y, threshold))
    loss_plot.pyplot(plot_loss([1] if not st.session_state.logistic_losses else st.session_state.logistic_losses))
    if st.session_state.confusion_matrix is not None:
        confusion_plot.pyplot(plot_confusion_matrix(st.session_state.confusion_matrix))
    plt.close('all')

# Display classification report if model has been trained
if st.session_state.predictions is not None:
    st.markdown("### Model Performance Metrics")
    report = classification_report(y_tensor, st.session_state.predictions, 
                                 target_names=['Class 0', 'Class 1'])
    st.code(report)

# Best use cases and references
st.markdown("""
### When to Use Logistic Regression?

#### Best Use Cases:
1. **Binary Classification Problems**
   - Email spam detection
   - Customer churn prediction
   - Disease diagnosis
   - Credit card fraud detection

2. **Probability Estimation**
   - Risk assessment
   - Lead scoring
   - Customer purchase probability

3. **Simple and Interpretable Models Needed**
   - Medical applications
   - Financial decision-making
   - Legal applications

#### Advantages:
- Provides probability scores
- Highly interpretable
- Computationally efficient
- Works well with linearly separable classes
- Less prone to overfitting than complex models

#### Limitations:
- Assumes linear decision boundary
- May underperform on complex relationships
- Requires feature independence
- Needs balanced dataset for best results

### Real-World Applications:
1. **Healthcare**
   - Disease diagnosis
   - Patient readmission prediction
   - Treatment response prediction

2. **Finance**
   - Credit approval
   - Fraud detection
   - Customer default prediction

3. **Marketing**
   - Customer conversion prediction
   - Email campaign response
   - Ad click-through rate prediction

### Learn More:
1. "Introduction to Statistical Learning" - Chapter on Logistic Regression
2. Stanford CS229 - Machine Learning Course Notes
3. StatQuest's Logistic Regression Videos
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary purpose of the sigmoid function in logistic regression?",
    (
        "To predict continuous values.",
        "To convert any input into a probability between 0 and 1.",
        "To calculate the distance between data points."
    )
)

if quiz_answer_1 == "To convert any input into a probability between 0 and 1.":
    st.success("Correct! The sigmoid function maps any input to a probability value between 0 and 1.")
else:
    st.error("Not quite. The sigmoid function is used to produce probabilities, not continuous values or distances.")

# Question 2
quiz_answer_2 = st.radio(
    "What does the decision boundary in logistic regression represent?",
    (
        "The line where the model's prediction is exactly 0.5.",
        "The average of all data points.",
        "The point where the loss function is minimized."
    )
)

if quiz_answer_2 == "The line where the model's prediction is exactly 0.5.":
    st.success("Correct! The decision boundary is where the model predicts a 50% probability, separating the two classes.")
else:
    st.error("Not quite. The decision boundary is the line where the model predicts a probability of 0.5.")