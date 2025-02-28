import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

# Page configuration
st.set_page_config(page_title="MLP Network Flow Visualization", page_icon="🧠", layout="wide")
st.title("Multilayer Perceptron (MLP) Network 🧠")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'losses' not in st.session_state:
    st.session_state.losses = []
if 'model' not in st.session_state:
    st.session_state.model = None

# Educational content
st.markdown("""
### Understanding MLPs
A Multilayer Perceptron (MLP) is like a chain of connected neurons that process information:

1. **Network Structure**
   - Input Layer: Receives data
   - Hidden Layers: Process information
   - Output Layer: Produces final result
   - Connections: Weighted paths between neurons

2. **How Data Flows**
   - Input values enter through input layer
   - Each neuron receives weighted inputs
   - Applies activation function
   - Passes result to next layer
""")

# Network parameters
st.sidebar.header("Network Architecture")
n_input = st.sidebar.slider("Input Neurons", 2, 5, 2)
n_hidden = st.sidebar.slider("Hidden Neurons per Layer", 2, 8, 4)
n_layers = st.sidebar.slider("Number of Hidden Layers", 1, 3, 1)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)

# Create placeholders for visualizations
st.write("### Training Loss")
loss_plot = st.empty()

# Add some spacing between plots
st.markdown("<br><br>", unsafe_allow_html=True)

st.write("### Network Structure")
network_plot = st.empty()





def plot_network(architecture, weights=None, activations=None):
    fig, ax = plt.subplots(figsize=(18, 8))
    layer_sizes = architecture
    layer_positions = np.linspace(0, 1, len(layer_sizes))
    
    # Custom color map - from light blue to deep blue
    def get_color(value):
        return plt.cm.Blues(value * 0.7 + 0.3)
    
    # Draw neurons and connections
    for i, (layer_pos, layer_size) in enumerate(zip(layer_positions, layer_sizes)):
        neuron_positions = np.linspace(0, 1, layer_size)
        for j, neuron_pos in enumerate(neuron_positions):
            # Default color and activation value
            color = '#E6F3FF'  # Light blue default
            act_val = None
            
            if activations is not None and i < len(activations) and j < len(activations[i]):
                act_val = activations[i][j]
                color = get_color(act_val)
            
            # Draw larger circle
            circle = plt.Circle((layer_pos, neuron_pos), 0.08, color=color, 
                              edgecolor='#2C3E50', linewidth=1, zorder=2)
            ax.add_artist(circle)
            
            # Add probability text
            if act_val is not None:
                text_color = "#2C3E50" if act_val < 0.7 else "white"
                ax.text(layer_pos, neuron_pos, f"{act_val:.2f}",
                       ha='center', va='center', fontsize=12, 
                       color=text_color, fontweight='bold')
            
            # Draw connections to next layer
            if i < len(layer_sizes) - 1 and weights is not None:
                next_layer_positions = np.linspace(0, 1, layer_sizes[i+1])
                for k, next_pos in enumerate(next_layer_positions):
                    if j < weights[i].shape[0] and k < weights[i].shape[1]:
                        weight = weights[i][j, k]
                        connection_color = '#95A5A6'
                        ax.plot([layer_pos, layer_positions[i+1]], 
                               [neuron_pos, next_pos], 
                               color=connection_color,
                               alpha=min(abs(weight), 1),
                               linewidth=max(abs(weight)*2, 0.5),
                               zorder=1)
    
    # Add layer labels
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes)-2)] + ['Output']
    for i, (pos, name) in enumerate(zip(layer_positions, layer_names)):
        ax.text(pos, 1.1, name, ha='center', va='bottom', fontsize=12, color='#2C3E50')
    
    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(-0.1, 1.2)
    plt.tight_layout()
    return fig

# Define MLP network class
class MLP(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i+1]))
            
    def forward(self, x):
        activations = [x]
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            activations.append(x)
        x = self.layers[-1](x)
        activations.append(x)
        return x, activations

# Training function
def train_step(model, x, y, optimizer, criterion):
    y_pred, activations = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), activations

# Initialize model and data
architecture = [n_input] + [n_hidden] * n_layers + [1]

# Create new model if architecture changes
current_arch_key = f"{n_input}_{n_hidden}_{n_layers}"
if 'current_arch' not in st.session_state or st.session_state.current_arch != current_arch_key:
    st.session_state.model = MLP(architecture)
    st.session_state.current_arch = current_arch_key
    st.session_state.losses = []
    st.session_state.step = 0

model = st.session_state.model

# Generate example data
x = torch.randn(100, n_input)
y = torch.sin(x.sum(dim=1, keepdim=True))

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training control
st.write("### Training Control")
col1, col2 = st.columns(2)

with col1:
    single_step = st.button("Single Step")
with col2:
    n_steps = st.number_input("Number of Steps", 1, 100, 10)
    multi_step = st.button("Train Multiple Steps")

# Training loop with visualization
if single_step or multi_step:
    steps = 1 if single_step else n_steps
    progress_bar = st.progress(0)
    
    for step in range(steps):
        # Update progress
        progress_bar.progress((step + 1) / steps)
        
        try:
            # Training step
            loss, activations = train_step(model, x, y, optimizer, criterion)
            
            # Get weights and normalize activations
            weights = []
            activation_values = []
            with torch.no_grad():
                for i, layer in enumerate(model.layers):
                    weights.append(layer.weight.detach().numpy())
                    if i < len(activations):
                        act = activations[i].detach().numpy()
                        act_mean = act.mean(axis=0)
                        act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
                        activation_values.append(act_norm)
                
                if len(activations) > len(model.layers):
                    act = activations[-1].detach().numpy()
                    act_mean = act.mean(axis=0)
                    act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
                    activation_values.append(act_norm)
            
            # Update visualizations
            plt.close('all')  # Clean up any existing plots
            
            # Network visualization
            fig = plot_network(architecture, weights, activation_values)
            network_plot.pyplot(fig)
            plt.close(fig)
            
            # Loss plot
            st.session_state.losses.append(loss)
            fig_loss, ax_loss = plt.subplots(figsize=(16, 4))
            ax_loss.plot(st.session_state.losses, marker='o', color='#3498DB')
            ax_loss.set_xlabel('Step')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title(f'Training Loss (Step {st.session_state.step})')
            ax_loss.grid(True, alpha=0.3)
            loss_plot.pyplot(fig_loss)
            plt.close(fig_loss)
            
            st.session_state.step += 1
            
            # Small delay for visualization
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            break

# Display current loss if available
if st.session_state.losses:
    st.write(f"Current Loss: {st.session_state.losses[-1]:.4f}")

# Best practices and tips
st.markdown("""
### Best Practices

1. **Network Size**
   - Start small and increase complexity as needed
   - More neurons don't always mean better performance
   - Consider your data complexity

2. **Training**
   - Watch for overfitting (when loss plateaus)
   - Adjust learning rate if training is unstable
   - Monitor activation patterns

3. **Reading the Visualization**
   - Neuron color intensity shows activation strength
   - Connection thickness indicates weight magnitude
   - Loss curve shows training progress
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary advantage of using multiple layers in an MLP?",
    (
        "It allows the network to learn more complex patterns.",
        "It reduces the number of neurons needed.",
        "It makes the network faster to train."
    )
)

if quiz_answer_1 == "It allows the network to learn more complex patterns.":
    st.success("Correct! Multiple layers enable the network to learn hierarchical features and more complex relationships in the data.")
else:
    st.error("Not quite. While multiple layers can increase complexity, they don't necessarily reduce neurons or speed up training.")

# Question 2
quiz_answer_2 = st.radio(
    "What is the role of activation functions in an MLP?",
    (
        "To normalize the input data.",
        "To introduce non-linearity into the network.",
        "To calculate the loss function."
    )
)

if quiz_answer_2 == "To introduce non-linearity into the network.":
    st.success("Correct! Activation functions allow the network to model non-linear relationships between inputs and outputs.")
else:
    st.error("Not quite. Activation functions are used to introduce non-linearity, not normalize data or calculate loss.")