# SmartML: Lightweight AutoML Web App for Model Training and Evaluation
A beginner-friendly AutoML web application built using Streamlit that enables users to upload datasets, explore data, engineer features, train machine learning models, evaluate performance, and export results — all from a simple UI, with support for both supervised and unsupervised learning.

# Project Overview
SmartML is designed for users with minimal coding experience who want to experiment with machine learning workflows. It runs efficiently on low-resource systems and provides step-by-step EDA, model selection, training, and evaluation for classification, regression, clustering, and dimensionality reduction (PCA) tasks.

# Features
- Automatic ML Pipeline: Train classification, regression, and clustering models.
- Exploratory Data Analysis: Visualize distributions, correlations, and missing data.
- Feature Engineering: Handle missing values, encoding, scaling, and more.
- Model Evaluation: Accuracy, confusion matrix, MAE, MSE, silhouette score, etc.
- Model Export: Save trained models as .pkl files.
- Data Support: Upload CSV files directly through the browser.
- Unsupervised Learning: K-Means Clustering, PCA visualization.

# Technologies Used
- Frontend/UI: Streamlit
- Backend: Python
- Machine Learning: scikit-learn, pandas, NumPy
- Visualization: matplotlib, seaborn, plotly
- Others: joblib, io, base64, warnings

# Export Options
- Trained models saved as .pkl files
- Visualizations rendered live in-app
- Option to download the model after training

# Ideal Use Cases
- Beginners wanting to learn ML workflows
- Educators teaching EDA/modeling
- Analysts working on low-spec machines
- Quick model testing without coding

# License
This project is open-source under the MIT License.
