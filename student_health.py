import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Load dataset from uploaded file
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

st.title("Student Mental Health Survey Clustering")

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.write("### Dataset Overview")
    st.write(df.head())
    st.write(df.describe())
    st.write(f"Data shape: {df.shape}")

    # Feature selection
    st.sidebar.header("Select Features")

    num_features = [col for col in df.columns if df[col].dtype != 'object']
    cat_features = [col for col in df.columns if df[col].dtype == 'object']

    selected_num_features = st.sidebar.multiselect("Select numerical features", num_features, default=num_features)
    selected_cat_features = st.sidebar.multiselect("Select categorical features", cat_features, default=cat_features)

    if selected_num_features or selected_cat_features:
        # Encoding categorical features
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in selected_cat_features:
            df_encoded[col] = le.fit_transform(df_encoded[col])

        # Scaling features
        scaler = MinMaxScaler()
        df_scaled = df_encoded.copy()
        df_scaled[selected_num_features] = scaler.fit_transform(df_encoded[selected_num_features])

        data = df_scaled[selected_num_features + selected_cat_features].values

        # Split data
        x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)

        # KMeans clustering
        num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=2)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)

        st.write(f"### Silhouette Score: {score:.2f}")

        # Display clustering results
        st.write("### Clustering Results:")
        df_scaled['Cluster'] = labels
        st.write(df_scaled.head())

        # Plot clusters
        if len(selected_num_features) >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(df_scaled[selected_num_features[0]], df_scaled[selected_num_features[1]], c=df_scaled['Cluster'], cmap='viridis')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            ax.set_xlabel(selected_num_features[0])
            ax.set_ylabel(selected_num_features[1])
            st.pyplot(fig)

        # New data input form
        st.sidebar.header("Classify New Data")
        new_data = {}
        
        for feature in selected_num_features:
            value = st.sidebar.number_input(f"Enter value for {feature}", value=0.0)
            new_data[feature] = value
        
        for feature in selected_cat_features:
            options = df[feature].unique()
            selected_option = st.sidebar.selectbox(f"Select value for {feature}", options)
            new_data[feature] = selected_option

        if st.sidebar.button("Classify New Data"):
            # Prepare new data for prediction
            new_df = pd.DataFrame([new_data])
            new_df_encoded = new_df.copy()
            
            # Encoding new data
            for col in selected_cat_features:
                new_df_encoded[col] = le.transform(new_df_encoded[col])
            
            # Scaling new data
            new_df_scaled = new_df_encoded.copy()
            new_df_scaled[selected_num_features] = scaler.transform(new_df_encoded[selected_num_features])

            new_data_scaled = new_df_scaled[selected_num_features + selected_cat_features].values
            cluster = kmeans.predict(new_data_scaled)
            
            st.write(f"### New Data Classification")
            st.write(f"The new data point belongs to cluster: {cluster[0]}")
else:
    st.info("Please upload a CSV file to get started.")
