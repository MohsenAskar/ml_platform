import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Feature Selection Techniques", page_icon="🎯", layout="wide")
st.title("Feature Selection Techniques 🎯")

# Generate synthetic data
@st.cache_data
def generate_data(n_samples, n_features, n_informative, n_redundant, n_repeated):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        random_state=42
    )
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    return X, y, feature_names

# Data generation parameters
st.sidebar.header("Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
n_features = st.sidebar.slider("Total Features", 10, 50, 20)
n_informative = st.sidebar.slider("Informative Features", 2, 10, 5)
n_redundant = st.sidebar.slider("Redundant Features", 0, 10, 3)
n_repeated = st.sidebar.slider("Repeated Features", 0, 5, 2)

# Generate data
X, y, feature_names = generate_data(n_samples, n_features, n_informative, 
                                    n_redundant, n_repeated)

# Feature selection methods
selection_category = st.selectbox(
    "Select Feature Selection Category",
    ["Filter Methods", "Wrapper Methods", "Embedded Methods"]
)

if selection_category == "Filter Methods":
    method = st.selectbox(
        "Select Filter Method",
        ["ANOVA F-value", "Mutual Information", "Correlation"]
    )
    
    if method == "ANOVA F-value":
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X, y)
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Score': selector.scores_,
            'P-value': selector.pvalues_
        }).sort_values('Score', ascending=False)
        
        # Visualize scores
        fig = px.bar(scores, x='Feature', y='Score',
                     title='ANOVA F-scores for each feature')
        st.plotly_chart(fig)
        
        # Show significant features
        st.write("### Significant Features (p-value < 0.05):")
        st.write(scores[scores['P-value'] < 0.05])
        
    elif method == "Mutual Information":
        mi_scores = mutual_info_classif(X, y)
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Mutual Information': mi_scores
        }).sort_values('Mutual Information', ascending=False)
        
        fig = px.bar(scores, x='Feature', y='Mutual Information',
                     title='Mutual Information Scores')
        st.plotly_chart(fig)
        
    else:  # Correlation
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))
        
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Correlation': correlations
        }).sort_values('Correlation', ascending=False)
        
        fig = px.bar(scores, x='Feature', y='Correlation',
                     title='Absolute Correlation with Target')
        st.plotly_chart(fig)
        
        # Show correlation matrix
        corr_matrix = np.corrcoef(X.T)
        fig = px.imshow(corr_matrix,
                        labels=dict(x="Features", y="Features"),
                        title="Feature Correlation Matrix")
        st.plotly_chart(fig)

elif selection_category == "Wrapper Methods":
    method = st.selectbox(
        "Select Wrapper Method",
        ["Recursive Feature Elimination (RFE)", "Forward Selection", "Backward Selection"]
    )
    
    if method == "Recursive Feature Elimination (RFE)":
        n_select = st.slider("Number of features to select", 1, n_features, 5)
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=n_select)
        rfe.fit(X, y)
        
        # Show selected features
        selected_features = pd.DataFrame({
            'Feature': feature_names,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        fig = px.bar(selected_features, x='Feature', y='Ranking',
                     title='RFE Feature Rankings (1 = Selected)')
        st.plotly_chart(fig)
        
        st.write("### Selected Features:")
        st.write(selected_features[selected_features['Selected']])
        
    elif method in ["Forward Selection", "Backward Selection"]:
        n_select = st.slider("Number of features to select", 1, n_features, 5, key="seq_n_select")
        direction = "forward" if method == "Forward Selection" else "backward"
        sfs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=100, random_state=42),
                                          n_features_to_select=n_select,
                                          direction=direction,
                                          cv=5,
                                          n_jobs=-1)
        sfs.fit(X, y)
        support = sfs.get_support()
        selected_features = pd.DataFrame({
            'Feature': feature_names,
            'Selected': support,
        })
        fig = px.bar(selected_features, x='Feature', y='Selected',
                     title=f'{method} (1 = Selected)',
                     labels={'Selected': 'Selection (1 = Selected, 0 = Not Selected)'})
        st.plotly_chart(fig)
        
        st.write("### Selected Features:")
        st.write(selected_features[selected_features['Selected']])
    
else:  # Embedded Methods
    method = st.selectbox(
        "Select Embedded Method",
        ["Lasso", "Random Forest Importance", "Elastic Net"]
    )
    
    if method == "Lasso":
        alpha = st.slider("Lasso Alpha", 0.001, 1.0, 0.01)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(lasso.coef_)
        }).sort_values('Coefficient', ascending=False)
        
        fig = px.bar(scores, x='Feature', y='Coefficient',
                     title='Lasso Coefficients (Absolute Value)')
        st.plotly_chart(fig)
        
        st.write("### Selected Features (Non-zero coefficients):")
        st.write(scores[scores['Coefficient'] > 0])
        
    elif method == "Random Forest Importance":
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(scores, x='Feature', y='Importance',
                     title='Random Forest Feature Importance')
        st.plotly_chart(fig)
        
        # Show cumulative importance
        scores['Cumulative Importance'] = scores['Importance'].cumsum()
        fig = px.line(scores, x='Feature', y='Cumulative Importance',
                      title='Cumulative Feature Importance')
        st.plotly_chart(fig)
        
    else:  # Elastic Net
        alpha = st.slider("Elastic Net Alpha", 0.001, 1.0, 0.01, key="elastic_alpha")
        l1_ratio = st.slider("Elastic Net L1 Ratio", 0.0, 1.0, 0.5, key="elastic_l1_ratio")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        elastic.fit(X_scaled, y)
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(elastic.coef_)
        }).sort_values('Coefficient', ascending=False)
        
        fig = px.bar(scores, x='Feature', y='Coefficient',
                     title='Elastic Net Coefficients (Absolute Value)')
        st.plotly_chart(fig)
        
        st.write("### Selected Features (Non-zero coefficients):")
        st.write(scores[scores['Coefficient'] > 0])

# ---------------- Performance Comparison ----------------
st.write("### Performance Comparison")
k_features = st.slider("Number of top features to use", 1, n_features, 5)

def get_cv_score(X, y, selected_features, model=RandomForestClassifier()):
    # If selected_features is a boolean mask:
    if isinstance(selected_features, np.ndarray) and selected_features.dtype == bool:
        if np.sum(selected_features) == 0:
            st.warning("No features were selected. Please adjust your parameters.")
            return np.nan, np.nan
        X_selected = X[:, selected_features]
    else:
        if len(selected_features) == 0:
            st.warning("No features were selected. Please adjust your parameters.")
            return np.nan, np.nan
        X_selected = X[:, selected_features]
    scores = cross_val_score(model, X_selected, y, cv=5, error_score='raise')
    return scores.mean(), scores.std()

comparison_results = []

# Filter methods
if selection_category == "Filter Methods":
    # ANOVA
    selector = SelectKBest(f_classif, k=k_features)
    selector.fit(X, y)
    score_anova = get_cv_score(X, y, selector.get_support())
    comparison_results.append(('ANOVA', score_anova[0], score_anova[1]))
    
    # Mutual Information
    selector = SelectKBest(mutual_info_classif, k=k_features)
    selector.fit(X, y)
    score_mi = get_cv_score(X, y, selector.get_support())
    comparison_results.append(('Mutual Info', score_mi[0], score_mi[1]))

# Wrapper methods
elif selection_category == "Wrapper Methods":
    # RFE or Sequential Feature Selector
    if method == "Recursive Feature Elimination (RFE)":
        rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=k_features)
        rfe.fit(X, y)
        score_rfe = get_cv_score(X, y, rfe.support_)
        comparison_results.append(('RFE', score_rfe[0], score_rfe[1]))
    else:
        # For Forward/Backward Selection using SequentialFeatureSelector
        direction = "forward" if method == "Forward Selection" else "backward"
        sfs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=100, random_state=42),
                                          n_features_to_select=k_features,
                                          direction=direction,
                                          cv=5,
                                          n_jobs=-1)
        sfs.fit(X, y)
        support = sfs.get_support()
        score_sfs = get_cv_score(X, y, support)
        comparison_results.append((f'{method}', score_sfs[0], score_sfs[1]))

# Embedded methods
else:
    # Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    top_features = np.argsort(rf.feature_importances_)[-k_features:]
    score_rf = get_cv_score(X, y, top_features)
    comparison_results.append(('Random Forest', score_rf[0], score_rf[1]))
    
    # Lasso
    lasso_model = Lasso(alpha=0.01, random_state=42)
    lasso_selector = SelectFromModel(lasso_model, max_features=k_features)
    lasso_selector.fit(X, y)
    selected_mask = lasso_selector.get_support()
    if np.sum(selected_mask) == 0:
        st.warning("Lasso did not select any features. Consider lowering the alpha parameter.")
    else:
        score_lasso = get_cv_score(X, y, selected_mask)
        comparison_results.append(('Lasso', score_lasso[0], score_lasso[1]))
    
    # Elastic Net
    alpha_perf = st.slider("Elastic Net Alpha (Performance Comparison)", 0.001, 1.0, 0.01, key="elastic_alpha_perf")
    l1_ratio_perf = st.slider("Elastic Net L1 Ratio (Performance Comparison)", 0.0, 1.0, 0.5, key="elastic_l1_ratio_perf")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    elastic = ElasticNet(alpha=alpha_perf, l1_ratio=l1_ratio_perf, random_state=42)
    elastic.fit(X_scaled, y)
    selected_mask = np.abs(elastic.coef_) > 1e-6
    if np.sum(selected_mask) == 0:
        st.warning("Elastic Net did not select any features. Consider lowering the alpha parameter or adjusting l1_ratio.")
    else:
        score_elastic = get_cv_score(X, y, selected_mask)
        comparison_results.append(('Elastic Net', score_elastic[0], score_elastic[1]))

# Plot comparison
comparison_df = pd.DataFrame(comparison_results, 
                             columns=['Method', 'CV Score', 'Std'])
fig = px.bar(comparison_df, x='Method', y='CV Score',
             error_y='Std',
             title=f'Cross-validation Scores with Top {k_features} Features')
st.plotly_chart(fig)

# ---------------- Educational Content ----------------
st.markdown("""
### Understanding Feature Selection Methods

#### Filter Methods
- **Advantages**: Fast, independent of model  
- **Disadvantages**: Ignores feature interactions  
- **Best for**: Quick initial feature screening  
- Works well with high-dimensional data

#### Wrapper Methods
- **Advantages**: Considers feature interactions  
- **Disadvantages**: Computationally intensive  
- **Best for**: Small to medium feature sets  
- Can find the optimal feature subset

#### Embedded Methods
- **Advantages**: Balance between filter and wrapper  
- **Disadvantages**: Model-specific  
- **Best for**: Medium to large feature sets  
- Combines feature selection with model training

### Best Practices:
1. Start with filter methods for initial screening  
2. Use domain knowledge when possible  
3. Consider computational resources  
4. Validate selection with cross-validation  
5. Compare multiple methods
""")

# ==================== Interactive Quiz ====================
st.subheader("Test Your Understanding")

# Question 1: Feature Selection Methods
quiz_answer_1 = st.radio(
    "Which feature selection method evaluates individual features independently without considering interactions?",
    (
        "Recursive Feature Elimination (RFE)",
        "Mutual Information",
        "Forward Selection"
    ),
    key="quiz_q1"
)

if quiz_answer_1 == "Mutual Information":
    st.success("✅ Correct! Mutual Information evaluates each feature's relevance independently, without considering feature interactions.")
else:
    st.error("❌ Not quite. RFE and Forward Selection take interactions into account, while Mutual Information does not.")

# Question 2: Embedded vs Wrapper Methods
quiz_answer_2 = st.radio(
    "What is a key advantage of Embedded Methods like Lasso and Elastic Net compared to Wrapper Methods?",
    (
        "They are faster and computationally efficient.",
        "They always find the best possible feature subset.",
        "They work better with non-numeric data."
    ),
    key="quiz_q2"
)

if quiz_answer_2 == "They are faster and computationally efficient.":
    st.success("✅ Correct! Embedded methods perform feature selection while training the model, making them faster than wrapper methods.")
else:
    st.error("❌ Not quite. Embedded methods optimize performance but do not guarantee the absolute best subset. They also require numeric data.")

