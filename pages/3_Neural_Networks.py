import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.data_generators import generate_classification_data

st.set_page_config(page_title="Neural Network Visualization", page_icon="🧠", layout="wide")
st.title("Neural Networks 🧠")

# Educational content
st.markdown("""
### What is a Neural Network?
A Neural Network is a machine learning algorithm inspired by how the human brain works. Just like our brain has neurons 
that communicate with each other, artificial neural networks have layers of connected "neurons" that learn patterns from data.

Think of it like this: if you're learning to identify dogs, your brain learns different features - like floppy ears, 
wagging tails, or furry bodies. Similarly, each neuron in our artificial network learns to recognize different patterns 
in the data.

### How Does It Work?
1. The network receives input data (like images or numbers)
2. This data travels through layers of connected neurons
3. Each connection has a "weight" that strengthens or weakens the signal
4. The network adjusts these weights as it learns from examples
5. Finally, it makes predictions based on what it has learned

### Loss Function
The loss function tells us how wrong our network's predictions are. Think of it like a score in a game - the lower 
the score, the better the network is performing. As the network trains, it tries to minimize this loss.
""")

# Parameters explanation in the sidebar
st.sidebar.header("Network Parameters")
st.sidebar.markdown("""
### Parameter Guide
""")

st.sidebar.markdown("**Number of Hidden Neurons**")
st.sidebar.markdown("""
More neurons mean the network can learn more complex patterns, but too many can lead to overfitting 
(like memorizing instead of learning).
""")
n_hidden = st.sidebar.slider("Number of Hidden Neurons", 2, 10, 4)

st.sidebar.markdown("**Learning Rate**")
st.sidebar.markdown("""
This controls how big steps the network takes when learning:
- Too high: might overshoot the best solution
- Too low: learns very slowly
""")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)

st.sidebar.markdown("**Number of Data Points**")
st.sidebar.markdown("More data points usually help the network learn better patterns.")
n_points = st.sidebar.slider("Number of Data Points", 50, 200, 100)

# Initialize session state
if 'losses' not in st.session_state:
    st.session_state.losses = []
if 'model' not in st.session_state:
    st.session_state.model = None
    
# Generate synthetic data
@st.cache_data
def get_data(n_points):
    return generate_classification_data(n_points)

X, y = get_data(n_points)

# Neural Network definition (same as before)
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Create the model
model = SimpleNN(2, n_hidden)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Create three columns for the visualizations
col1, col2, col3 = st.columns(3)

# [Previous plotting functions remain the same]
def plot_decision_boundary(model, X, y):
    fig, ax = plt.subplots(figsize=(6, 6))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    X_grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    Z = model(X_grid).detach().numpy()
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Class 0')
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Class 1')
    ax.set_title('Decision Boundary')
    ax.legend()
    return fig

def plot_network(model, n_hidden):
    # [Previous network plotting code remains the same]
    fig, ax = plt.subplots(figsize=(6, 6))
    weights1 = model.hidden.weight.detach().numpy()
    weights2 = model.output.weight.detach().numpy()
    layer_sizes = [2, n_hidden, 1]
    layer_positions = [0, 1, 2]
    
    for i, size in enumerate(layer_sizes):
        y_positions = np.linspace(0, size-1, size) * 0.5
        for j in range(size):
            ax.scatter([layer_positions[i]], [y_positions[j]], c='gray', s=100)
    
    for i in range(2):
        for j in range(n_hidden):
            ax.plot([0, 1], 
                   [i*0.5, j*0.5],
                   c=plt.cm.RdYlBu(weights1[j, i]/2 + 0.5),
                   alpha=0.5)
    
    for i in range(n_hidden):
        ax.plot([1, 2],
                [i*0.5, 0],
                c=plt.cm.RdYlBu(weights2[0, i]/2 + 0.5),
                alpha=0.5)
    
    ax.set_title('Network Architecture')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(losses)
    ax.set_title('Training Loss Over Time')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    return fig

# Create placeholder containers for plots
with col1:
    decision_plot = st.empty()
with col2:
    network_plot = st.empty()
with col3:
    loss_plot = st.empty()

# Training controls
epochs = st.slider("Number of epochs to train", 1, 50, 10)
train_button = st.button("Train Model")

# Status containers
progress_bar = st.empty()
status_text = st.empty()

if train_button:
    st.session_state.losses = []
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Store loss
        st.session_state.losses.append(loss.item())
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        
        # Update status
        accuracy = ((outputs > 0.5) == y_tensor).float().mean()
        status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy.item():.4f}')
        
        # Update plots using placeholders
        decision_plot.pyplot(plot_decision_boundary(model, X, y))
        network_plot.pyplot(plot_network(model, n_hidden))
        loss_plot.pyplot(plot_loss(st.session_state.losses))
        
        # Clear matplotlib figures to prevent memory issues
        plt.close('all')

# Initial plots
if not train_button:
    decision_plot.pyplot(plot_decision_boundary(model, X, y))
    network_plot.pyplot(plot_network(model, n_hidden))
    loss_plot.pyplot(plot_loss([0] if not st.session_state.losses else st.session_state.losses))
    plt.close('all')

# Best use cases and references
st.markdown("""
### When to Use Neural Networks?
Neural Networks are best suited for:
1. **Complex Pattern Recognition**
   - Image and facial recognition
   - Speech recognition
   - Language translation

2. **Predictions with Many Factors**
   - Weather forecasting
   - Stock market prediction
   - Customer behavior prediction

3. **Data with Hidden Patterns**
   - Fraud detection
   - Medical diagnosis
   - Recommendation systems

### Limitations
- Require large amounts of data
- Can be computationally expensive
- Results can be hard to interpret
- Need careful parameter tuning

### Learn More
For more information, check out these beginner-friendly resources:
1. [3Blue1Brown's Neural Network Video Series](https://www.3blue1brown.com/topics/neural-networks)
2. [MIT's Introduction to Deep Learning](http://introtodeeplearning.com/)
3. "Make Your Own Neural Network" by Tariq Rashid (book)
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary role of the loss function in a neural network?",
    (
        "To measure how well the network is performing.",
        "To determine the number of neurons in the hidden layer.",
        "To decide the learning rate for the network."
    )
)

if quiz_answer_1 == "To measure how well the network is performing.":
    st.success("Correct! The loss function quantifies how wrong the network's predictions are, guiding the learning process.")
else:
    st.error("Not quite. The loss function measures the network's performance, not the number of neurons or the learning rate.")

# Question 2
quiz_answer_2 = st.radio(
    "What happens if you use too many neurons in the hidden layer?",
    (
        "The network will learn faster.",
        "The network may overfit the data.",
        "The network will always perform better."
    )
)

if quiz_answer_2 == "The network may overfit the data.":
    st.success("Correct! Too many neurons can cause the network to memorize the training data instead of learning general patterns.")
else:
    st.error("Not quite. Too many neurons can lead to overfitting, where the network performs poorly on new, unseen data.")