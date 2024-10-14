from typing import List
import pandas as pd
import numpy as np
from doc_file import DocFile
from llm_model import LLMModel
from models.label_governance_model import LabelGovernanceModel
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import pickle


class ClusterAndClassify:

    def _find_optimal_clusters(self, data, max_clusters=20):
        inertia = []
        silhouette_scores = []

        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            
            inertia.append(kmeans.inertia_)  # For the Elbow method
            silhouette_avg = silhouette_score(data, cluster_labels)  # Silhouette score
            silhouette_scores.append(silhouette_avg)
        
        # elbow_k = np.argmin(np.gradient(inertia)) + 2  # Adding 2 to account for 0-based indexing
        
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

    def cluster_classify(self, data: List[DocFile], partition_size = 0.7):
        data_dict = [d.to_dict() for d in data]
        data_partition = int(np.floor(len(data_dict) * partition_size))
        cluster_data = data_dict[:data_partition]
        classification_data = data_dict[data_partition:]

        # df['combined_text'] = df[df.columns].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)


        clustering_df = pd.DataFrame(cluster_data)
        classfication_df = pd.DataFrame(classification_data)
        print('Clustering_df: ', clustering_df.shape,'\n', 'Classfication_df: ', classfication_df.shape)

        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform(clustering_df['data'])

        optimal_k, kmeans_model = self._find_optimal_clusters(tfidf_matrix, max_clusters=20)
        print(f"Optimal number of clusters: {optimal_k}")

        clustering_df['cluster_labels'] = kmeans_model.labels_

        cluster_X = clustering_df['data']
        cluster_y = clustering_df['cluster_labels']

    
        self.train_classifier(cluster_X, cluster_y)

        classfication_df = self.classify_data(classification_data)

        final_df = pd.concat([clustering_df, classfication_df], ignore_index=True)

        return final_df
    
    def train_classifier(self, data, labels):
        tfidf_vectorizer = TfidfVectorizer()
        cluster_X_tfidf = tfidf_vectorizer.fit_transform(data)

        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(cluster_X_tfidf, labels)

        with open('ml_model/doc_classifier_model.pkl', 'wb') as file:
            pickle.dump(classifier, file)

    def classify_data(self, data: List[DocFile]):
        classfication_df = [d.to_dict() for d in data]
        classify_X = classfication_df['data']

        tfidf_vectorizer = TfidfVectorizer()
        classify_X_tfidf = tfidf_vectorizer.transform(classify_X)

        with open('ml_model/doc_classifier_model.pkl', 'rb') as file:
            classifier = pickle.load(file)

        predicted_labels = classifier.predict(classify_X_tfidf)
        classfication_df['cluster_labels'] = predicted_labels
        return classfication_df

    
    def generate_cluster_labels(self, cluster_df):
        model = LLMModel()
        unique_classes = cluster_df["cluster_labels"].unique()
        res = []
        for c in unique_classes:
            filtered_rows = cluster_df[cluster_df['cluster_labels'] == c]
            query = f'''
            You are given a set of documents from the same cluster. Your task is to generate a label for this cluster and determine the sensitivity level based on the following categories: Public Data, Internal Data, Confidential Data, Restricted Data, Private Data, Critical Data, Regulatory Data.

            Only output the label name and the sensitivity level, nothing else.
            Generate one label for the whole cluster and one sensitivity level for the cluster based on the provided categories.
            Here is the data from the cluster:
            {filtered_rows['text']}
            '''
            q_res = model.infer_model(query, LabelGovernanceModel)
            filtered_rows['label'] = q_res.label
            filtered_rows['sensitivity'] = q_res.sensitivity
            res.append(filtered_rows)

        return pd.concat(res, ignore_index=True)

