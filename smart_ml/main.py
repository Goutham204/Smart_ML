import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import io
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

st.title("Smart ML")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def drop_unwanted_columns(df):
    drops_col = st.sidebar.multiselect("Drop Columns (e.g. ID, Date)", df.columns)
    return df.drop(columns=drops_col) if drops_col else df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data")
    st.write(df)

    df = drop_unwanted_columns(df)

    st.subheader("Missing Value Check")
    missing_info = df.isnull().sum()
    st.write(missing_info[missing_info > 0])

    st.subheader("Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed.")

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    st.subheader("Exploratory Data Analysis")
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] >= 2:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.sidebar.subheader("Choose Learning Type")
    learning_type = st.sidebar.selectbox("Choose one:", ["Supervised Learning", "Unsupervised Learning"])

    if learning_type == "Supervised Learning":
        task_type = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
        target_column = st.sidebar.selectbox("Target Column", df.columns)
        feature_columns = st.sidebar.multiselect("Select Features to Train", [col for col in df.columns if col != target_column], default=[col for col in df.columns if col != target_column])

        X = df[feature_columns]
        y = df[target_column]

        X = pd.get_dummies(X, drop_first=True)
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model_choice = st.selectbox("Select Model", [
            "Logistic Regression", "Random Forest", "Gradient Boosting", "Decision Tree", "KNN", "Naive Bayes", "Linear Regression"])

        if st.button("Train Model"):
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier() if task_type == "Classification" else RandomForestRegressor()
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingClassifier() if task_type == "Classification" else GradientBoostingRegressor()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier() if task_type == "Classification" else DecisionTreeRegressor()
            elif model_choice == "KNN":
                model = KNeighborsClassifier() if task_type == "Classification" else KNeighborsRegressor()
            elif model_choice == "Naive Bayes":
                model = GaussianNB()
            elif model_choice == "Linear Regression":
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Model Trained Successfully")

            if task_type == "Classification":
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
            else:
                st.write("MSE:", mean_squared_error(y_test, y_pred))
                st.write("R2 Score:", r2_score(y_test, y_pred))

            model_name = st.text_input("Enter model file name:", "my_model.pkl")
            if st.button("Save Model to Disk"):
                joblib.dump(model, model_name)
                st.success(f"Model saved locally as {model_name}")

            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            buffer.seek(0)
            st.download_button("Download Trained Model", buffer, file_name=model_name, mime="application/octet-stream")

    elif learning_type == "Unsupervised Learning":
        st.subheader("Unsupervised Learning")
        unsupervised_model = st.selectbox("Choose Unsupervised Model", ["KMeans", "PCA"])

        unsupervised_columns = st.multiselect("Select columns for unsupervised learning", df.select_dtypes(include=['number']).columns.tolist())

        if len(unsupervised_columns) > 0:
            unsupervised_data = df[unsupervised_columns]

            if unsupervised_model == "KMeans":
                max_k = st.slider("Max K for Elbow Method", 1, 10, 5)
                distortions = []
                for k in range(1, max_k+1):
                    km = KMeans(n_clusters=k, random_state=42)
                    km.fit(unsupervised_data)
                    distortions.append(km.inertia_)

                st.write("Elbow Method Plot")
                fig, ax = plt.subplots()
                ax.plot(range(1, max_k+1), distortions, marker='o')
                ax.set_xlabel("K")
                ax.set_ylabel("Inertia")
                st.pyplot(fig)

            elif unsupervised_model == "PCA":
                n_components = st.slider("Components", 1, min(len(unsupervised_columns), 10), 2)
                pca = PCA(n_components=n_components)
                pca_data = pca.fit_transform(unsupervised_data)
                st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
                fig = px.scatter(x=pca_data[:, 0], y=pca_data[:, 1], title="PCA Result")
                st.plotly_chart(fig)

    st.download_button("Download Cleaned CSV", df.to_csv(index=False), file_name="cleaned_data.csv")
