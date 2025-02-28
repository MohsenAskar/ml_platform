import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.data_generators import generate_regression_data

st.set_page_config(page_title="Linear Regression Learning Visualization 📈", page_icon="📈", layout="wide")
st.title("Linear Regression Learning 📈")

# Educational content specific to Linear Regression
st.markdown("""
### What is Linear Regression?
Linear Regression is one of the simplest and most interpretable machine learning algorithms. It tries to find 
the best straight line that describes the relationship between input features (X) and the target variable (y).

Think of it like this: if you're trying to predict house prices based on square footage, Linear Regression 
will find the line that best represents how price typically increases with size.

### The Math Behind It (Simply Explained)
- **Line Equation**: y = mx + b
  - y is what we're trying to predict (like price)
  - x is our input data (like square footage)
  - m is the slope (how much y changes when x changes)
  - b is the y-intercept (the base value when x is 0)

### How Does It Learn?
1. Start with a random line
2. Calculate how wrong the line is (error/loss)
3. Adjust the line to reduce the error
4. Repeat until the line fits well
""")

# Parameters explanation in the sidebar
st.sidebar.header("Model Parameters")

st.sidebar.markdown("**Learning Rate**")
st.sidebar.markdown("""
Controls how much we adjust our line in each step:
- Too high: The line might jump around too much
- Too low: Takes too long to find the best line
""")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)

st.sidebar.markdown("**Noise Level**")
st.sidebar.markdown("""
How much random variation to add to the data:
- Low: Data points close to a perfect line
- High: More scattered data points
""")
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.3)

st.sidebar.markdown("**Number of Data Points**")
st.sidebar.markdown("More points generally give a more stable line fit.")
n_points = st.sidebar.slider("Number of Data Points", 10, 200, 50)

# Initialize session state for loss tracking
if 'lr_losses' not in st.session_state:
    st.session_state.lr_losses = []

# Generate synthetic data
@st.cache_data
def get_regression_data(n_points, noise):
    X, y = generate_regression_data(n_points, noise)
    # Normalize X and y for better training
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
    return X, y

X, y = get_regression_data(n_points, noise_level)

# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return self.weight * x + self.bias

# Create model and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create three columns for visualizations
col1, col2 = st.columns(2)

def plot_regression_line(model, X, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data points
    ax.scatter(X, y, c='blue', alpha=0.5, label='Data Points')
    
    # Plot regression line
    X_sorted = np.sort(X)
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_sorted))
    ax.plot(X_sorted, y_pred, c='red', label='Regression Line')
    
    # Add equation to plot
    weight = model.weight.item()
    bias = model.bias.item()
    equation = f'y = {weight:.2f}x + {bias:.2f}'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title('Linear Regression Fit')
    ax.set_xlabel('Input Feature (X)')
    ax.set_ylabel('Target Variable (y)')
    ax.legend()
    return fig

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(losses)
    ax.set_title('Mean Squared Error Loss Over Time')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (MSE)')
    ax.set_yscale('log')  # Log scale for better loss visualization
    return fig

# Create placeholder containers
with col1:
    regression_plot = st.empty()
with col2:
    loss_plot = st.empty()

# Training controls
epochs = st.slider("Number of epochs to train", 1, 100, 20)
train_button = st.button("Train Model")

# Status containers
progress_bar = st.empty()
status_text = st.empty()

if train_button:
    st.session_state.lr_losses = []
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Store loss
        st.session_state.lr_losses.append(loss.item())
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        
        # Update status
        status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}')
        
        # Update plots
        regression_plot.pyplot(plot_regression_line(model, X, y))
        loss_plot.pyplot(plot_loss(st.session_state.lr_losses))
        plt.close('all')

# Initial plots
if not train_button:
    regression_plot.pyplot(plot_regression_line(model, X, y))
    loss_plot.pyplot(plot_loss([1] if not st.session_state.lr_losses else st.session_state.lr_losses))
    plt.close('all')

# Best use cases and references
st.markdown("""
### When to Use Linear Regression?

#### Best Use Cases:
1. **Simple Predictions with Clear Linear Relationships**
   - House price prediction based on size
   - Sales forecasting based on advertising spend
   - Height prediction based on age (within specific ranges)

2. **Baseline Model Development**
   - Quick initial model to compare more complex algorithms against
   - Understanding basic relationships in data

3. **Feature Importance Analysis**
   - Understanding which factors most strongly influence an outcome
   - Identifying key business drivers

#### Limitations:
- Only captures linear relationships
- Sensitive to outliers
- Assumes independent variables are not highly correlated
- May not work well with non-linear patterns

### Advantages:
- Very interpretable (easy to explain)
- Fast to train
- Requires minimal computing power
- Good for understanding feature importance

### Real-World Applications:
1. **Finance**
   - Predicting stock prices (short-term)
   - Risk assessment
   - Portfolio optimization

2. **Real Estate**
   - Property valuation
   - Rent price estimation

3. **Marketing**
   - Sales forecasting
   - Campaign impact analysis
   - Customer lifetime value prediction

### Learn More:
1. "Introduction to Statistical Learning" by James, Witten, Hastie, Tibshirani
2. Khan Academy's Linear Regression Course
3. StatQuest Linear Regression Videos on YouTube
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What does the slope (m) in the linear regression equation y = mx + b represent?",
    (
        "The base value of y when x is 0.",
        "The rate at which y changes with respect to x.",
        "The amount of noise in the data."
    )
)

if quiz_answer_1 == "The rate at which y changes with respect to x.":
    st.success("Correct! The slope (m) represents how much y changes for a unit change in x.")
else:
    st.error("Not quite. Remember, the slope (m) indicates the rate of change between y and x.")

# Question 2
quiz_answer_2 = st.radio(
    "What happens if the learning rate in gradient descent is set too high?",
    (
        "The model will take too long to converge.",
        "The model might overshoot the optimal solution and fail to converge.",
        "The model will ignore the data and always predict the mean."
    )
)

if quiz_answer_2 == "The model might overshoot the optimal solution and fail to converge.":
    st.success("Correct! A high learning rate can cause the model to jump around and miss the optimal solution.")
else:
    st.error("Not quite. A high learning rate can cause the model to overshoot and fail to converge.")