import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from matplotlib.patches import Rectangle, Arrow

# Page configuration
st.set_page_config(page_title="CNN Network Flow Visualization", page_icon="🖼️", layout="wide")
st.title("Convolutional Neural Network (CNN) 🖼️")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'losses' not in st.session_state:
    st.session_state.losses = []
if 'model' not in st.session_state:
    st.session_state.model = None

# Educational content
st.markdown("""
### Understanding CNNs
A Convolutional Neural Network (CNN) processes spatial data using filters:

1. **Network Components**
   - Convolutional Layers: Extract features using filters
   - Pooling Layers: Reduce spatial dimensions
   - Fully Connected Layers: Final classification/regression

2. **How Data Flows**
   - Input image processed by conv filters
   - Feature maps created from each filter
   - Pooling reduces dimensionality
   - Features combined for final output
""")

# Network parameters
st.sidebar.header("Network Architecture")
input_size = st.sidebar.slider("Input Size", 16, 32, 28)
n_conv_layers = st.sidebar.slider("Number of Conv Layers", 1, 3, 2)
n_filters = st.sidebar.slider("Filters per Layer", 2, 8, 4)
kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
n_classes = st.sidebar.slider("Number of Classes", 2, 10, 5)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)

def plot_feature_map(ax, data, pos, size, title=None):
    """Plot a single feature map with heatmap."""
    # Handle different input dimensions
    if len(data.shape) == 4:  # (batch, channels, height, width)
        data = data[0, 0]  # Take first sample and first channel
    elif len(data.shape) == 3:  # (batch, height, width) or (channels, height, width)
        data = data[0]  # Take first sample or channel
    elif len(data.shape) == 2:  # (height, width)
        data = data
    else:
        raise ValueError(f"Unexpected shape {data.shape}")
        
    im = ax.imshow(data, cmap='viridis', extent=[pos[0], pos[0]+size[0], pos[1], pos[1]+size[1]])
    if title:
        ax.text(pos[0] + size[0]/2, pos[1] + size[1] + 0.02, title,
                ha='center', va='bottom', fontsize=10)
    return im

def plot_kernel(ax, kernel, pos, size):
    """Plot a convolutional kernel."""
    # If the kernel has an extra channel dimension, remove it or average over channels
    if kernel.ndim == 3:
        if kernel.shape[0] == 1:
            kernel = kernel[0]  # Remove the singleton channel dimension
        else:
            kernel = kernel.mean(axis=0)  # Average across multiple channels
    im = ax.imshow(kernel, cmap='coolwarm', extent=[pos[0], pos[0] + size[0], pos[1], pos[1] + size[1]])
    ax.add_patch(Rectangle(pos, size[0], size[1], fill=False, color='gray', linewidth=1))
    return im


def plot_cnn_network(architecture, feature_maps=None, kernels=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Architecture should contain: [input_size, conv_sizes, fc_sizes]
    input_size, conv_sizes, fc_sizes = architecture
    
    # Calculate positions
    total_width = 0.8
    start_x = 0.1
    
    # Track current x position
    current_x = start_x
    layer_spacing = total_width / (len(conv_sizes) + 3)  # +3 for input, flatten, and output
    
    # Plot input
    input_height = 0.6
    input_pos = [current_x, 0.2]
    if feature_maps and len(feature_maps) > 0:
        plot_feature_map(ax, feature_maps[0], input_pos, [layer_spacing*0.8, input_height], "Input")
    else:
        ax.add_patch(Rectangle(input_pos, layer_spacing*0.8, input_height, 
                             fill=True, color='lightgray'))
        ax.text(input_pos[0] + layer_spacing*0.4, input_pos[1] + input_height/2,
                f"{input_size}x{input_size}", ha='center', va='center')
    current_x += layer_spacing

    # Plot convolutional layers
    for i, conv_size in enumerate(conv_sizes):
        # Plot feature maps
        n_maps = conv_size[2]  # number of channels
        map_height = input_height / n_maps
        map_spacing = map_height * 0.2
        
        for j in range(n_maps):
            map_pos = [current_x, 0.2 + j * (map_height + map_spacing)]
            # Plot feature maps
            if feature_maps and len(feature_maps) > i+1:
                # For conv layers, show each filter's output
                current_feature_map = feature_maps[i+1]
                if len(current_feature_map.shape) == 4:  # (batch, channels, height, width)
                    if j < current_feature_map.shape[1]:  # Check if this filter exists
                        plot_feature_map(ax, current_feature_map[:,j:j+1], map_pos,
                                       [layer_spacing*0.8, map_height],
                                       f"Conv{i+1}\nFilter{j+1}")
            else:
                ax.add_patch(Rectangle(map_pos, layer_spacing*0.8, map_height,
                                     fill=True, color='lightgray'))
                ax.text(map_pos[0] + layer_spacing*0.4, map_pos[1] + map_height/2,
                       f"{conv_size[0]}x{conv_size[1]}", ha='center', va='center')
        else:
            ax.add_patch(Rectangle(map_pos, layer_spacing*0.8, map_height,
                                fill=True, color='lightgray'))
            ax.text(map_pos[0] + layer_spacing*0.4, map_pos[1] + map_height/2,
                f"{conv_size[0]}x{conv_size[1]}", ha='center', va='center')
    
        # Plot kernels
        if kernels and len(kernels) > i:
            kernel_size = 0.1
            for j in range(min(n_maps, 3)):  # Show max 3 kernels
                kernel_pos = [current_x - kernel_size/2,
                            0.2 + j * (map_height + map_spacing) - kernel_size/2]
                plot_kernel(ax, kernels[i][j], kernel_pos, [kernel_size, kernel_size])
        
        current_x += layer_spacing

    # Plot flatten layer
    flatten_pos = [current_x, 0.2]
    ax.add_patch(Rectangle(flatten_pos, layer_spacing*0.8, input_height,
                          fill=True, color='lightblue', alpha=0.3))
    ax.text(flatten_pos[0] + layer_spacing*0.4, flatten_pos[1] + input_height/2,
            "Flatten", ha='center', va='center', rotation=90)
    current_x += layer_spacing

            # Plot fully connected output
    fc_height = input_height / n_classes
    for i in range(n_classes):
        fc_pos = [current_x, 0.2 + i * fc_height]
        ax.add_patch(Rectangle(fc_pos, layer_spacing*0.8, fc_height,
                             fill=True, color='lightblue'))
        if feature_maps and len(feature_maps) > len(conv_sizes) + 1:
            val = float(feature_maps[-1][0, i])  # Convert to float scalar
            ax.text(fc_pos[0] + layer_spacing*0.4, fc_pos[1] + fc_height/2,
                   f"Class {i}: {val:.2f}", ha='center', va='center')
        else:
            ax.text(fc_pos[0] + layer_spacing*0.4, fc_pos[1] + fc_height/2,
                   f"Class {i}", ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

# Define CNN class
class SimpleCNN(nn.Module):
    def __init__(self, input_size, n_conv_layers, n_filters, kernel_size, n_classes):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_sizes = []
        
        # Calculate sizes after each conv layer
        current_size = input_size
        current_channels = 1
        
        for i in range(n_conv_layers):
            self.conv_layers.append(nn.Conv2d(current_channels, n_filters, kernel_size, padding=kernel_size//2))
            current_channels = n_filters
            self.conv_sizes.append([current_size, current_size, n_filters])
        
        # Calculate flattened size
        self.flat_size = current_size * current_size * n_filters
        self.fc = nn.Linear(self.flat_size, n_classes)
        
    def forward(self, x):
        # Store input as first feature map
        feature_maps = [x.detach().cpu().numpy()]  # Shape: (batch, channel, height, width)
        kernels = []
        
        # Convolutional layers
        for conv in self.conv_layers:
            kernels.append(conv.weight.detach().cpu().numpy())
            x = F.relu(conv(x))
            # Store feature maps after each conv layer
            feature_maps.append(x.detach().cpu().numpy())
        
        # Flatten and fully connected
        x = x.view(-1, self.flat_size)
        x = self.fc(x)
        feature_maps.append(F.softmax(x, dim=1).detach().numpy())
        
        return x, feature_maps, kernels

# Create placeholders for visualizations with 80/20 split
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.write("### Network Structure")
    network_plot = st.empty()

with col2:
    st.write("### Training Loss")
    loss_plot = st.empty()

# Set input channels
in_channels = 1  # For grayscale images

# Initialize model and data
current_arch = f"{input_size}_{n_conv_layers}_{n_filters}_{kernel_size}_{n_classes}"
if 'current_arch' not in st.session_state or st.session_state.current_arch != current_arch:
    st.session_state.model = SimpleCNN(input_size, n_conv_layers, n_filters, kernel_size, n_classes)
    st.session_state.current_arch = current_arch
    st.session_state.losses = []
    st.session_state.step = 0

model = st.session_state.model

# Generate example data
def generate_sample_data(batch_size=32):
    x = torch.randn(batch_size, 1, input_size, input_size)
    y = torch.randint(0, n_classes, (batch_size,))
    return x, y

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

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
            # Generate new data
            x, y = generate_sample_data()
            
            # Forward pass
            output, feature_maps, kernels = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update visualizations
            plt.close('all')
            
            # Network visualization
            architecture = [input_size, model.conv_sizes, n_classes]
            fig = plot_cnn_network(architecture, feature_maps, kernels)
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
- **Input**: Original image
- **Conv Layers**: Feature maps showing activated patterns
- **Kernels**: Small squares showing learned filters
- **Feature Maps**: Activation patterns after each convolution
- **Output**: Class probabilities

#### Training Tips
1. **Kernel Size**
   - Larger kernels capture bigger patterns
   - But increase computation and parameters
   - 3x3 or 5x5 are common choices

2. **Number of Filters**
   - More filters can capture more features
   - But require more memory and computation
   - Start small and increase if needed

3. **Network Depth**
   - Deeper networks can learn more complex patterns
   - But may be harder to train
   - Add layers gradually as needed
""")


# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary advantage of using Convolutional Neural Networks (CNNs) for image processing?",
    (
        "They require fewer parameters compared to fully connected networks.",
        "They process images faster than traditional neural networks by using larger kernels.",
        "They eliminate the need for backpropagation."
    )
)

if quiz_answer_1 == "They require fewer parameters compared to fully connected networks.":
    st.success("Correct! CNNs use shared weights and local connectivity, reducing the number of parameters while maintaining efficiency.")
else:
    st.error("Not quite. CNNs reduce parameters through shared weights and local connectivity but still rely on backpropagation.")

# Question 2
quiz_answer_2 = st.radio(
    "What is the function of a pooling layer in a CNN?",
    (
        "To increase the resolution of feature maps.",
        "To reduce the spatial size of feature maps, retaining important features.",
        "To directly classify images."
    )
)

if quiz_answer_2 == "To reduce the spatial size of feature maps, retaining important features.":
    st.success("Correct! Pooling layers help downsample feature maps, reducing computational cost while preserving essential information.")
else:
    st.error("Not quite. Pooling layers primarily reduce the size of feature maps while preserving key information, making computations more efficient.")
