import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

# Page configuration
st.set_page_config(page_title="RNN Network Flow Visualization", page_icon="🔄", layout="wide")
st.title("Recurrent Neural Network (RNN) 🔄")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'losses' not in st.session_state:
    st.session_state.losses = []
if 'model' not in st.session_state:
    st.session_state.model = None

# Educational content
st.markdown("""
### Understanding RNNs
A Recurrent Neural Network (RNN) processes sequential data by maintaining a hidden state:

1. **Network Structure**
   - Input Layer: Receives sequential data
   - Hidden Layer: Maintains state over time
   - Output Layer: Generates predictions
   - Recurrent Connections: Connect hidden state across time steps

2. **How Data Flows**
   - Input sequence enters one step at a time
   - Hidden state updates using both input and previous state
   - Output generated based on current hidden state
   - State carries information through time
""")

# Network parameters
st.sidebar.header("Network Architecture")
n_input = st.sidebar.slider("Input Size", 1, 5, 1)
n_hidden = st.sidebar.slider("Hidden Size", 2, 8, 4)
n_output = st.sidebar.slider("Output Size", 1, 5, 1)
sequence_length = st.sidebar.slider("Sequence Length", 2, 5, 3)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)

def plot_rnn_network(architecture, weights=None, activations=None, time_steps=3):
    """
    Improved visualization function for RNN to prevent overlapping labels and neurons.
    Dynamically adjusts neuron spacing and label positioning.
    """
    # Create a larger figure with enough space
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Define horizontal (x) positions for time steps using left/right margins
    left_margin = 0.12
    right_margin = 0.9
    time_positions = np.linspace(left_margin, right_margin, time_steps)
    
    # Define vertical space for neuron placement (common for all layers)
    top_margin = 0.88
    bottom_margin = 0.12

    # The architecture is [input_size, hidden_size, output_size]
    sizes = architecture
    layer_labels = ['Input', 'Hidden', 'Output']
    
    # Store neuron coordinates for later connection drawing
    coords = []
    
    # Compute max neurons in any layer to avoid overcrowding
    max_neurons = max(sizes)
    
    # Vertical position adjustments for labels (to avoid overlap)
    label_offsets = { 
        "Input": 0.05,  # Shift labels to the left
        "Hidden": 0.07,
        "Output": 0.09
    }

    # Draw neurons for each time step and each layer
    for t, x_pos in enumerate(time_positions):
        coords_t = []  # Coordinates for time step t
        for layer_idx, n_neurons in enumerate(sizes):
            # Compute y positions evenly within the vertical margins.
            if n_neurons > 1:
                y_positions = np.linspace(top_margin, bottom_margin, n_neurons)
            else:
                y_positions = [(top_margin + bottom_margin) / 2]
            
            # Save positions so we can connect neurons later.
            layer_coords = []
            for i, y in enumerate(y_positions):
                # Determine color based on activation if provided.
                color = '#E6F3FF'
                if activations and len(activations) > t and len(activations[t]) > layer_idx:
                    act_array = activations[t][layer_idx]
                    if i < len(act_array):
                        act_val = act_array[i]
                        # Use a blue colormap: scale activation to a color value.
                        color = plt.cm.Blues(act_val * 0.7 + 0.3)
                # Draw the neuron as a circle.
                circle = plt.Circle((x_pos, y), 0.035, color=color, edgecolor='#2C3E50',
                                    linewidth=1, zorder=2)
                ax.add_artist(circle)
                # Optionally, draw the activation value inside the circle.
                if activations and len(activations[t]) > layer_idx:
                    act_array = activations[t][layer_idx]
                    if i < len(act_array):
                        act_val = act_array[i]
                        text_color = "#2C3E50" if act_val < 0.7 else "white"
                        ax.text(x_pos, y, f"{act_val:.2f}", ha='center', va='center',
                                fontsize=10, color=text_color)
                layer_coords.append((x_pos, y))
            coords_t.append(layer_coords)
        coords.append(coords_t)
    
    # Draw connections within the same time step
    for t in range(time_steps):
        # Input to Hidden connections
        for i, start in enumerate(coords[t][0]):  # Input layer neurons
            for j, end in enumerate(coords[t][1]):  # Hidden layer neurons
                weight = weights[0][j][i] if weights and len(weights) > 0 else 0.5
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color='#95A5A6', alpha=min(abs(weight), 1),
                        linewidth=max(abs(weight)*2, 0.5), zorder=1)
        # Hidden to Output connections
        for i, start in enumerate(coords[t][1]):  # Hidden layer
            for j, end in enumerate(coords[t][2]):  # Output layer
                weight = weights[1][j][i] if weights and len(weights) > 1 else 0.5
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color='#95A5A6', alpha=min(abs(weight), 1),
                        linewidth=max(abs(weight)*2, 0.5), zorder=1)
    
    # Draw recurrent connections between hidden layers at successive time steps
    for t in range(time_steps - 1):
        for i, start in enumerate(coords[t][1]):  # Hidden neurons at time t
            for j, end in enumerate(coords[t+1][1]):  # Hidden neurons at time t+1
                weight = weights[2][j][i] if weights and len(weights) > 2 else 0.5
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color='#E74C3C', alpha=min(abs(weight), 0.7),
                        linewidth=max(abs(weight)*2, 0.5), zorder=1,
                        linestyle='--')

    # **FIXED LABELS: Place them dynamically to avoid overlap**
    for layer_idx, label in enumerate(layer_labels):
        y_positions = [coord[1] for coord in coords[0][layer_idx]]
        y_label_pos = y_positions[0]  # Align with the first neuron
        ax.text(left_margin - label_offsets[label], y_label_pos, label, 
                ha='right', va='center', fontsize=13, color='#2C3E50', fontweight='bold')

    # Add time step labels along the bottom
    for t, x_pos in enumerate(time_positions):
        ax.text(x_pos, bottom_margin - 0.08, f"t={t}", ha='center', va='top',
                fontsize=12, color='#2C3E50')

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig

# Define RNN class
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x_sequence):
        batch_size = x_sequence.size(0)
        seq_length = x_sequence.size(1)
        hidden = torch.zeros(batch_size, self.hidden_size)
        
        activations = []
        for t in range(seq_length):
            x_t = x_sequence[:, t, :]
            hidden_raw = self.input_to_hidden(x_t) + self.hidden_to_hidden(hidden)
            hidden = torch.tanh(hidden_raw)
            output = self.hidden_to_output(hidden)
            
            # Store activations for visualization
            step_activations = [x_t.detach(), hidden.detach(), output.detach()]
            activations.append(step_activations)
        
        return output, activations

# Create placeholders for visualizations with 80/20 split
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.write("### Network Structure")
    network_plot = st.empty()

with col2:
    st.write("### Training Loss")
    loss_plot = st.empty()

# Initialize model and data
if 'current_arch' not in st.session_state or st.session_state.current_arch != f"{n_input}_{n_hidden}_{n_output}_{sequence_length}":
    st.session_state.model = SimpleRNN(n_input, n_hidden, n_output)
    st.session_state.current_arch = f"{n_input}_{n_hidden}_{n_output}_{sequence_length}"
    st.session_state.losses = []
    st.session_state.step = 0

model = st.session_state.model

# Generate example data (simple sine wave sequence)
def generate_sequence_data(batch_size=32):
    t = torch.linspace(0, 4*np.pi, sequence_length)
    x = torch.sin(t).unsqueeze(-1).unsqueeze(0)
    x = x.repeat(batch_size, 1, n_input)
    y = torch.sin(t + 0.1).unsqueeze(-1).unsqueeze(0)
    y = y.repeat(batch_size, 1, n_output)
    return x, y[:, -1, :]  # Return only last target

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
        progress_bar.progress((step + 1) / steps)
        
        try:
            # Generate new sequence data
            x, y = generate_sequence_data()
            
            # Forward pass
            output, activations = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Get weights for visualization
            weights = []
            with torch.no_grad():
                weights.append(model.input_to_hidden.weight.detach().numpy())
                weights.append(model.hidden_to_output.weight.detach().numpy())
                weights.append(model.hidden_to_hidden.weight.detach().numpy())
                
                # Normalize activations
                norm_activations = []
                for t_activations in activations:
                    t_norm = []
                    for layer_act in t_activations:
                        act_mean = layer_act.mean(dim=0).numpy()
                        act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
                        t_norm.append(act_norm)
                    norm_activations.append(t_norm)
            
            # Update visualizations
            plt.close('all')
            
            # Network visualization
            fig = plot_rnn_network([n_input, n_hidden, n_output], weights, norm_activations, sequence_length)
            network_plot.pyplot(fig)
            plt.close(fig)
            
            # Loss plot
            st.session_state.losses.append(loss.item())
            fig_loss, ax_loss = plt.subplots(figsize=(4, 8))
            ax_loss.plot(st.session_state.losses, marker='o', color='#3498DB', linewidth=2)
            ax_loss.set_xlabel('Step', labelpad=10)
            ax_loss.set_ylabel('Loss', labelpad=10)
            ax_loss.set_title(f'Step {st.session_state.step}')
            ax_loss.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            loss_plot.pyplot(fig_loss)
            plt.close(fig_loss)
            
            st.session_state.step += 1
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            break

# Display current loss if available
if st.session_state.losses:
    st.write(f"Current Loss: {st.session_state.losses[-1]:.4f}")

# Additional educational content
st.markdown("""
### Understanding the Visualization

#### Network Structure
- **Neurons**: Circles represent neurons in each layer
- **Colors**: Darker blue indicates higher activation
- **Red Dashed Lines**: Recurrent connections (hidden state memory)
- **Gray Lines**: Standard feed-forward connections
- **Time Steps**: Network unfolded across multiple time steps (t=0,1,2...)

#### Training Tips
1. **Sequence Length**
   - Longer sequences can capture more temporal patterns
   - But may be harder to train
   - Start with shorter sequences and gradually increase

2. **Hidden Size**
   - More hidden units can capture more complex patterns
   - But may lead to overfitting
   - Balance based on your sequence complexity

3. **Learning Rate**
   - RNNs can be sensitive to learning rate
   - If loss oscillates, try reducing it
   - If learning is slow, try increasing it
""")

# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the key feature of an RNN that makes it suitable for sequential data?",
    (
        "It uses convolutional layers to process data.",
        "It has memory to retain information from previous time steps.",
        "It requires fixed-size inputs for all sequences."
    )
)

if quiz_answer_1 == "It has memory to retain information from previous time steps.":
    st.success("Correct! RNNs use hidden states to retain information from previous time steps, making them ideal for sequential data.")
else:
    st.error("Not quite. RNNs are designed to handle sequential data by retaining memory of previous inputs, not through convolutional layers or fixed-size inputs.")

# Question 2
quiz_answer_2 = st.radio(
    "What is a common challenge faced by standard RNNs?",
    (
        "They cannot handle non-linear data.",
        "They struggle with long-term dependencies due to vanishing gradients.",
        "They require too many hidden layers to work effectively."
    )
)

if quiz_answer_2 == "They struggle with long-term dependencies due to vanishing gradients.":
    st.success("Correct! Standard RNNs often struggle with long-term dependencies because gradients can vanish over many time steps.")
else:
    st.error("Not quite. The main challenge for standard RNNs is handling long-term dependencies due to vanishing gradients.")