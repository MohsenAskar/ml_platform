import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

st.set_page_config(page_title="GAN Visualization", page_icon="🎨", layout="wide")
st.title("Generative Adversarial Networks (GANs) 🎨")

# Educational content
st.markdown("""
### What are GANs?
GANs are like a counterfeiter (Generator) and a detective (Discriminator) playing a game:

1. **The Game**
   - Generator creates fake data
   - Discriminator tries to spot fakes
   - Both improve over time
   - Result: Generator creates realistic data

2. **Key Components**
   - Generator: Creates synthetic data
   - Discriminator: Classifies real/fake
   - Latent Space: Input to Generator
   - Loss Functions: Guide improvement

3. **Training Process**
   - Train Discriminator on real/fake data
   - Train Generator to fool Discriminator
   - Alternate training steps
   - Balance is crucial
""")

# Parameters explanation in the sidebar
st.sidebar.header("GAN Parameters")

st.sidebar.markdown("**Data Type**")
data_type = st.sidebar.selectbox("Type of Data to Generate", 
    ["2D Gaussian", "Swiss Roll", "Circles"])

st.sidebar.markdown("**Network Size**")
n_hidden = st.sidebar.slider("Hidden Layer Size", 4, 32, 8)

st.sidebar.markdown("**Training**")
noise_dim = st.sidebar.slider("Noise Dimension", 2, 10, 2)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
iterations = st.sidebar.slider("Number of Iterations", 10, 1000, 100, step=10)

# Generate real data
@st.cache_data
def generate_real_data(data_type, n_samples=1000):
    if data_type == "2D Gaussian":
        return np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[1, 0.5], [0.5, 1]],
            size=n_samples
        )
    elif data_type == "Swiss Roll":
        t = np.random.uniform(0, 4*np.pi, n_samples)
        x = t * np.cos(t)
        y = t * np.sin(t)
        return np.column_stack([x, y]) / 10
    else:  # Circles
        r = np.random.uniform(0, 1, n_samples)
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

# Define networks
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def plot_data_distribution(real_data, fake_data=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot real data
    ax.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.5, label='Real')
    
    if fake_data is not None:
        # Plot generated data
        ax.scatter(fake_data[:, 0], fake_data[:, 1], c='red', alpha=0.5, label='Generated')
    
    ax.legend()
    ax.set_title('Data Distribution')
    return fig

def plot_losses(g_losses, d_losses):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(g_losses, label='Generator Loss')
    ax.plot(d_losses, label='Discriminator Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    return fig

def plot_discriminator_confidence(discriminator, real_data, fake_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real data confidence
    with torch.no_grad():
        real_conf = discriminator(torch.FloatTensor(real_data)).numpy()
    
    # Fake data confidence
    with torch.no_grad():
        fake_conf = discriminator(torch.FloatTensor(fake_data)).numpy()
    
    # Plot histograms
    ax1.hist(real_conf, bins=30, alpha=0.5, label='Real')
    ax1.hist(fake_conf, bins=30, alpha=0.5, label='Fake')
    ax1.set_title('Discriminator Confidence Distribution')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # Plot scatter with confidence
    scatter1 = ax2.scatter(real_data[:, 0], real_data[:, 1], 
                           c=real_conf.ravel(), cmap='Blues', 
                           label='Real', alpha=0.5)
    scatter2 = ax2.scatter(fake_data[:, 0], fake_data[:, 1], 
                           c=fake_conf.ravel(), cmap='Reds', 
                           label='Fake', alpha=0.5)
    ax2.set_title('Spatial Confidence Distribution')
    ax2.legend()
    
    return fig

# Create containers for visualization
row1 = st.container()
row2 = st.container()

# Generate real data
real_data = generate_real_data(data_type)

# Initialize networks
generator = Generator(noise_dim, n_hidden, 2)
discriminator = Discriminator(2, n_hidden)

# Initialize optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Create placeholders for Data Distribution and Training Loss plots
col1, col2 = st.columns(2)
with col1:
    st.write("### Data Distribution")
    distribution_plot = st.empty()

with col2:
    st.write("### Training Losses")
    loss_plot = st.empty()

# Create a placeholder for Discriminator Analysis in a fixed location
with row2:
    st.write("### Discriminator Analysis")
    disc_analysis_plot = st.empty()

# Initialize loss history in session state if not already done
if 'g_losses' not in st.session_state:
    st.session_state.g_losses = []
    st.session_state.d_losses = []

# Train button with dynamic label
train_button = st.button(f"Train for {iterations} iterations")

if train_button:
    progress_bar = st.progress(0)
    for i in range(iterations):
        # -------------------------------
        # Train Discriminator
        # -------------------------------
        d_optimizer.zero_grad()
        
        # Use real data
        real_tensor = torch.FloatTensor(real_data)
        d_real = discriminator(real_tensor)
        d_real_loss = criterion(d_real, torch.ones(len(real_data), 1))
        
        # Generate fake data
        noise = torch.randn(len(real_data), noise_dim)
        fake_data = generator(noise).detach()
        d_fake = discriminator(fake_data)
        d_fake_loss = criterion(d_fake, torch.zeros(len(real_data), 1))
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()
        
        # -------------------------------
        # Train Generator
        # -------------------------------
        g_optimizer.zero_grad()
        
        noise = torch.randn(len(real_data), noise_dim)
        fake_data = generator(noise)
        d_fake = discriminator(fake_data)
        g_loss = criterion(d_fake, torch.ones(len(real_data), 1))
        
        g_loss.backward()
        g_optimizer.step()
        
        # Store losses
        st.session_state.g_losses.append(g_loss.item())
        st.session_state.d_losses.append(d_loss.item())
        
        # Update progress bar
        progress_bar.progress((i + 1) / iterations)
        
        # Update plots every 10 iterations
        if (i + 1) % 10 == 0:
            # Update Data Distribution plot
            with torch.no_grad():
                noise = torch.randn(len(real_data), noise_dim)
                fake_data = generator(noise).numpy()
            fig = plot_data_distribution(real_data, fake_data)
            distribution_plot.pyplot(fig)
            plt.close(fig)
            
            # Update Loss plot
            fig = plot_losses(st.session_state.g_losses, st.session_state.d_losses)
            loss_plot.pyplot(fig)
            plt.close(fig)
            
            # Update Discriminator Analysis plot in the same placeholder
            fig = plot_discriminator_confidence(discriminator, real_data, fake_data)
            disc_analysis_plot.pyplot(fig)
            plt.close(fig)

# If not training, show current state (initial plots)
if not train_button:
    with row1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Initial Data Distribution")
            fig = plot_data_distribution(real_data)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("### Training Losses")
            if st.session_state.g_losses:
                fig = plot_losses(st.session_state.g_losses, st.session_state.d_losses)
                st.pyplot(fig)
                plt.close(fig)

# Best use cases and references
st.markdown("""
### When to Use GANs?

#### Best Use Cases:
1. **Image Generation**
   - Face generation
   - Art creation
   - Image-to-image translation

2. **Data Augmentation**
   - Training data generation
   - Rare case simulation
   - Balanced dataset creation

3. **Domain Transfer**
   - Style transfer
   - Cross-domain mapping
   - Data translation

#### Advantages:
- Can learn complex distributions
- Generate high-quality samples
- Unsupervised learning
- Flexible architecture
- Creative applications

#### Limitations:
- Training instability
- Mode collapse risk
- Needs careful tuning
- Evaluation challenges
- Computational intensity

### Real-World Applications:

1. **Art and Design**
   - Digital art creation
   - Fashion design
   - Product design

2. **Healthcare**
   - Medical image synthesis
   - Drug discovery
   - Anomaly detection

3. **Entertainment**
   - Game asset creation
   - Virtual characters
   - Special effects


### Learn More:
1. Original GAN Paper by Goodfellow et al.
2. "Deep Learning" Book - GAN Chapter
3. GAN Lab Interactive Demo
""")
st.subheader("Test Your Understanding")

# Quiz Question 1
quiz_answer1 = st.radio(
    "1. What is the main objective of the Generator in a GAN?",
    (
        "To create synthetic data that fools the Discriminator",
        "To classify data as real or fake",
        "To optimize the Discriminator's performance"
    ),
    key="q1"
)

if quiz_answer1 == "To create synthetic data that fools the Discriminator":
    st.success("Correct! The Generator's goal is to produce data that the Discriminator misclassifies as real.")
else:
    st.error("Not quite. Remember, the Generator is designed to generate realistic data that can fool the Discriminator.")

# Quiz Question 2
quiz_answer2 = st.radio(
    "2. What is the primary function of the Discriminator in a GAN?",
    (
        "To generate synthetic data",
        "To evaluate and classify data as real or fake",
        "To add noise to the input data"
    ),
    key="q2"
)

if quiz_answer2 == "To evaluate and classify data as real or fake":
    st.success("Correct! The Discriminator's role is to distinguish between real and generated (fake) data.")
else:
    st.error("Not quite. Recall that the Discriminator evaluates inputs to decide if they are real or generated.")
