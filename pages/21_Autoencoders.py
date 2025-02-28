import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_swiss_roll, make_blobs, make_moons
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Autoencoder Visualization", page_icon="🎭", layout="wide")
st.title("Autoencoder Learning 🎭")

# Educational content
st.markdown("""
### What is an Autoencoder?
An Autoencoder is like a data compression algorithm that learns from examples:

1. **Key Components**
   - Encoder: Compresses input to latent space
   - Decoder: Reconstructs input from latent space
   - Latent Space: Compressed representation
   - Reconstruction Loss: Measures quality

2. **How it Works**
   - Input → Encoder → Latent Space
   - Latent Space → Decoder → Reconstruction
   - Train to minimize reconstruction error
   - Learn efficient representations

3. **Applications**
   - Dimensionality reduction
   - Feature learning
   - Anomaly detection
   - Denoising
""")

# Parameters in sidebar
st.sidebar.header("Autoencoder Parameters")

st.sidebar.markdown("**Data Type**")
data_type = st.sidebar.selectbox("Type of Data", 
    ["Swiss Roll", "Moons", "Blobs"])

st.sidebar.markdown("**Architecture**")
latent_dim = st.sidebar.slider("Latent Dimension", 1, 3, 2)
hidden_dim = st.sidebar.slider("Hidden Layer Size", 4, 32, 16)

st.sidebar.markdown("**Training**")
n_iterations = st.sidebar.slider("Number of Iterations", 10, 500, 100)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)

# Generate data
@st.cache_data
def generate_data(data_type, n_samples=1000):
    if data_type == "Swiss Roll":
        X, _ = make_swiss_roll(n_samples=n_samples)
        return X[:, [0, 2]] / 10
    elif data_type == "Moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.1)
        return X
    else:  # Blobs
        X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5)
        return X

X = generate_data(data_type)
input_dim = X.shape[1]

# Define network
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

def plot_reconstruction(X, X_rec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax1.set_title('Original Data')
    
    # Reconstructed data
    ax2.scatter(X_rec[:, 0], X_rec[:, 1], alpha=0.5, c='r')
    ax2.set_title('Reconstructed Data')
    
    return fig

def plot_latent_space(z, X):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if z.shape[1] == 1:
        # 1D latent space
        ax.scatter(z[:, 0], np.zeros_like(z[:, 0]), alpha=0.5)
        ax.set_ylim(-1, 1)
    else:
        # 2D or 3D latent space (plot first 2 dimensions)
        scatter = ax.scatter(z[:, 0], z[:, 1], alpha=0.5,
                           c=np.sum(X**2, axis=1))
        plt.colorbar(scatter)
    
    ax.set_title('Latent Space Representation')
    return fig

def plot_loss_curve(losses):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    return fig

# Create model
model = Autoencoder(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Create plot placeholders
st.write("### Training Progress")
col1, col2 = st.columns(2)

with col1:
    st.write("Reconstruction Comparison")
    reconstruction_plot = st.empty()

with col2:
    st.write("Training Loss")
    loss_plot = st.empty()

st.write("### Latent Space")
latent_plot = st.empty()

# Training loop
train_button = st.button("Train Model")

if train_button:
    # Convert data to tensor
    X_tensor = torch.FloatTensor(X)
    losses = []
    
    # Progress bar
    progress_bar = st.progress(0)
    
    for i in range(n_iterations):
        # Forward pass
        X_rec, z = model(X_tensor)
        loss = criterion(X_rec, X_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Update progress bar
        progress_bar.progress((i + 1) / n_iterations)
        
        # Update plots periodically
        if (i + 1) % 10 == 0 or i == n_iterations - 1:
            # Get current reconstructions and latent representations
            with torch.no_grad():
                X_rec_np = X_rec.numpy()
                z_np = z.numpy()
            
            # Update reconstruction plot
            fig1 = plot_reconstruction(X, X_rec_np)
            reconstruction_plot.pyplot(fig1)
            plt.close(fig1)
            
            # Update loss plot
            fig2 = plot_loss_curve(losses)
            loss_plot.pyplot(fig2)
            plt.close(fig2)
            
            # Update latent space plot
            fig3 = plot_latent_space(z_np, X)
            latent_plot.pyplot(fig3)
            plt.close(fig3)
    
    st.success(f"Training completed after {n_iterations} iterations!")

    # Final Analysis
    st.write("### Reconstruction Quality")
    with torch.no_grad():
        final_rec, _ = model(X_tensor)
        final_loss = criterion(final_rec, X_tensor).item()
    
    metrics_df = pd.DataFrame({
        'Metric': ['Final Reconstruction Loss', 'Compression Ratio'],
        'Value': [
            final_loss,
            f"{input_dim}:{latent_dim} ({input_dim/latent_dim:.1f}x)"
        ]
    })
    st.write(metrics_df)

# Best use cases and references
st.markdown("""
### When to Use Autoencoders?

#### Best Use Cases:
1. **Dimensionality Reduction**
   - High-dimensional data visualization
   - Feature extraction
   - Data compression

2. **Anomaly Detection**
   - Fault detection
   - Fraud detection
   - Quality control

3. **Denoising**
   - Image denoising
   - Signal processing
   - Data cleaning

#### Advantages:
- Unsupervised learning
- Non-linear dimensionality reduction
- Feature learning
- Flexible architecture
- Data-driven compression

#### Limitations:
- No guaranteed structure in latent space
- May learn trivial solutions
- Requires tuning
- No explicit density estimation
- Training can be unstable

### Real-World Applications:

1. **Image Processing**
   - Image compression
   - Denoising
   - Inpainting

2. **Anomaly Detection**
   - Manufacturing defects
   - Network intrusion
   - Medical diagnosis

3. **Feature Learning**
   - Recommendation systems
   - Information retrieval
   - Transfer learning

### Learn More:
1. "Deep Learning" Book - Chapter on Autoencoders
2. Keras Autoencoder Examples
3. Tutorial on Variational Autoencoders
""")



# Interactive Quiz
st.subheader("Test Your Understanding")

# Question 1
quiz_answer_1 = st.radio(
    "What is the primary purpose of an autoencoder?",
    (
        "To learn an efficient representation of input data by encoding and decoding it.",
        "To classify input data into predefined categories.",
        "To generate entirely new data from random noise."
    )
)

if quiz_answer_1 == "To learn an efficient representation of input data by encoding and decoding it.":
    st.success("Correct! Autoencoders compress input data into a latent representation and then reconstruct it back, capturing essential features.")
else:
    st.error("Not quite. Autoencoders are used for unsupervised learning to encode and decode data, not for classification or generating new data like GANs.")

# Question 2
quiz_answer_2 = st.radio(
    "What is the role of the latent space in an autoencoder?",
    (
        "It stores a compressed representation of the input data.",
        "It assigns labels to input data points.",
        "It is used to apply dropout regularization during training."
    )
)

if quiz_answer_2 == "It stores a compressed representation of the input data.":
    st.success("Correct! The latent space is a lower-dimensional representation of the input, capturing essential features while discarding noise.")
else:
    st.error("Not quite. The latent space is where an autoencoder stores the compressed representation of the input, not for labeling or regularization purposes.")
