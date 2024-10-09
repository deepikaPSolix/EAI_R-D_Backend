import pandas as pd
import numpy as np
import sklearn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

def find_optimal_clusters(data, max_clusters=20):
    inertia = []
    silhouette_scores = []

    cluster_range = range(2, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        inertia.append(kmeans.inertia_)  # For the Elbow method
        silhouette_avg = silhouette_score(data, cluster_labels)  # Silhouette score
        silhouette_scores.append(silhouette_avg)
    
    elbow_k = np.argmin(np.gradient(inertia)) + 2  # Adding 2 to account for 0-based indexing
    
    silhouette_k = cluster_range[np.argmax(silhouette_scores)]
    
    best_k = silhouette_k
    
    kmeans_model = KMeans(n_clusters=best_k, random_state=42)
    kmeans_model.fit(data)
    
    # Plotting Elbow Method and Silhouette Scores for reference
    plt.figure(figsize=(12, 6))

    # Plot inertia (Elbow Method)
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.show()
    
    return best_k, kmeans_model

def cluster_classify(data, partition_size):
    df = pd.read_csv(data)

    df['combined_text'] = df[df.columns].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)

    data_partition = int(np.floor(len(df) * partition_size))

    clustering_df = df.iloc[:data_partition]
    classfication_df = df.iloc[data_partition:]
    print('Clustering_df: ', clustering_df.shape,'\n', 'Classfication_df: ', classfication_df.shape)

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(clustering_df['combined_text'])

    optimal_k, kmeans_model = find_optimal_clusters(tfidf_matrix, max_clusters=20)
    print(f"Optimal number of clusters: {optimal_k}")

    clustering_df['cluster_labels'] = kmeans_model.labels_

    cluster_X = clustering_df['combined_text']
    cluster_y = clustering_df['cluster_labels']

    tfidf_vectorizer = TfidfVectorizer()
    cluster_X_tfidf = tfidf_vectorizer.fit_transform(cluster_X)

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(cluster_X_tfidf, cluster_y)

    classify_X = classfication_df['combined_text']
    classify_X_tfidf = tfidf_vectorizer.transform(classify_X)

    predicted_labels = classifier.predict(classify_X_tfidf)

    # Store the predicted labels back to the DataFrame
    classfication_df['cluster_labels'] = predicted_labels

    final_df = pd.concat([clustering_df, classfication_df], ignore_index=True)

    return final_df
