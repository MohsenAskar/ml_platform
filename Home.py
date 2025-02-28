import streamlit as st
import base64

st.set_page_config(
    page_title="ML Algorithm Visualizer Platform",
    page_icon="🧠",
    layout="wide"
)

# Convert the image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load your image from a local path
image_path = (r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Introduce_Your_Self\cartoon.JPG")
# Get the base64 string of the image
image_base64 = image_to_base64(image_path)

# Display your image and name in the top right corner
st.markdown(
    f"""
    <style>
    .header {{
        position: absolute;  /* Fix the position */
        top: -60px;  /* Adjust as needed */
        right: -40px;  /* Align to the right */
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 10px;
        flex-direction: column; /* Stack items vertically */
        text-align: center; /* Ensures text is centrally aligned */
    }}
    .header img {{
        border-radius: 50%;
        width: 50px;
        height: 50px;
        margin-bottom: 5px; /* Space between image and text */
    }}
    .header-text {{
        font-size: 12px;
        font-weight: normal; /* Regular weight for text */
        text-align: center;
    }}
    </style>
    <div class="header">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Mohsen Askar">
        <div class="header-text">Developed by: Mohsen Askar</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Machine Learning Algorithm Visualizer")
st.markdown("""
Welcome to the ML Algorithm Visualizer!

This application helps you understand how different machine learning algorithms work in real-time.

### Available Algorithm visualizations:
- **Linear Regression**: Visualize how the regression line adapts to data points
- **Logistic Regression**: See how probabilities are calculated
- **Neural Networks (NN)**: Watch neurons activate and weights adjust during training
- **Multilayer Perceptron (MLP)**: See how the decision boundary changes
- **Recurrent Neural Network (RNN)**: Observe how sequences are processed
- **Convolutional Neural Networks (CNN)**: See how filters activate and weights adjust
- **Support Vector Machine (SVM)**: See the decision boundary evolution
- **Decision Trees (DT)**: See how trees split and make predictions
- **Random Forest (RF)**: Observe tree splitting and ensemble predictions
- **Gradient Boosting Machine (GBM)**: Watch how each tree corrects errors of the previous one
- **Bagging vs Boosting**: Understand the ensemble learning process
- **Principal Component Analysis (PCA)**: Visualize how data is projected
- **Density-Based Spatial Clustering (DBSCAN)**: See how clusters expand and contract based on density
- **Uniform Manifold Approximation and Projection (UMAP)**: Observe manifold learning in action
- **K-Means Clustering**: See how centroids move and clusters form
- **k-Nearest Neighbors**: Watch how the decision boundary changes
- **Hierarchical Clustering**: Observe the dendrogram formation
- **Elastic Net Regression**: Visualize the effect of L1 and L2 regularization
- **Naive Bayes**: See how probabilities are calculated based on Bayes' theorem
- **Generative Adversarial Networks (GANs)**: Observe the generator and discriminator training process
- **Autoencoders**: See how data is compressed and reconstructed
- **Cross-Validation**: Understand how K-Fold CV splits data
- **Evaluation Metrics**: Compare different metrics like accuracy, precision, recall, F1-score
- **Feature Selection**: Compare different methods like filter, wrapper, embedded
- **Global and Local Interpretability**: Understand model predicttions using different techniques


Choose an algorithm from the sidebar to get started!

### How to Use
1. Select an algorithm from the sidebar
2. Adjust the parameters using the controls
3. Watch the algorithm learn and adapt in real-time
4. Experiment with different settings to understand their impact
""")

# Display some sample visualizations or key metrics on the home page
st.sidebar.success("Select an algorithm above.")