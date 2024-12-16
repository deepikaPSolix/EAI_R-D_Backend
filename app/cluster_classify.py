from typing import List
import pandas as pd
import numpy as np
from app.models.doc_file import DocFile
from app.llm_model import LLMModel
from app.models.label_governance_model import GovernanceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


class ClusterAndClassify:
    
    def __init__(self) -> None:
        self.ml_model_path = "cache/ml-model"

    def _find_optimal_clusters(self, data, max_clusters=5):
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
        
        return best_k, kmeans_model

    def cluster_classify(self, data: List[DocFile], partition_size = 0.7):
        data_partition = int(np.floor(len(data) * partition_size))
        cluster_data = data[:data_partition]
        classification_data = data[data_partition:]

        clustering_df = pd.DataFrame([d.model_dump(exclude={'chunks'}) for d in cluster_data])
        classfication_df = pd.DataFrame([d.model_dump(exclude={'chunks'})for d in classification_data])
        print('Clustering_df: ', clustering_df.shape,'\n', 'Classfication_df: ', classfication_df.shape)

        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform(clustering_df['data'])

        optimal_k, kmeans_model = self._find_optimal_clusters(tfidf_matrix)
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
        with open(os.path.join(self.ml_model_path, 'doc_classifier_model.pkl'), 'wb') as file:
            pickle.dump(classifier, file)

        with open(os.path.join(self.ml_model_path, 'tfidf_vectorizer.pkl'), 'wb') as file:
            pickle.dump(tfidf_vectorizer, file)

    def classify_data(self, data: List[DocFile]):
        classfication_df = pd.DataFrame([d.model_dump(exclude={'chunks'}) for d in data])
        classify_X = classfication_df['data']

        with open(os.path.join(self.ml_model_path, 'tfidf_vectorizer.pkl'), 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        classify_X_tfidf = tfidf_vectorizer.transform(classify_X)

        with open(os.path.join(self.ml_model_path, 'doc_classifier_model.pkl'), 'rb') as file:
            classifier = pickle.load(file)

        predicted_labels = classifier.predict(classify_X_tfidf)
        classfication_df['cluster_labels'] = predicted_labels
        return classfication_df

    
    def generate_cluster_labels(self, cluster_df: pd.DataFrame):
        model = LLMModel()
        unique_classes = cluster_df["cluster_labels"].unique()
        res = []
        for c in unique_classes:
            filtered_rows = cluster_df[cluster_df['cluster_labels'] == c].sample(n=10)
            rows = []
            for row in filtered_rows.itertuples():
                rows.append(f'file_name: {row.file_name}, data: {row.data}')
            query = f'''
            You are provided with 5 chunks of text from each of 10 documents belonging to the same cluster. These chunks are a representative sample of the cluster's content. Your job is to analyze the provided data and generate a single, concise, and meaningful label that represents the central theme or topic of the entire cluster.

            Guidelines:

            Consider all the chunks holistically to identify the overarching theme shared by the documents.
            The label should be clear, specific, and concise (e.g., "Sustainable Energy Solutions" or "Trends in Digital Marketing").
            Avoid overly broad or overly specific labels; focus on capturing the central idea of the cluster based on the given chunks.
            Disregard any minor details or outliers that do not align with the main topic.

            Cluster Data:
            {rows}
            '''
            q_res = model.infer_model(query, GovernanceModel)
            print("QRES: " + str(c))
            print(q_res.dict())
            res_dict = q_res.dict()
            df = pd.DataFrame(res_dict['attributes'])
            result = pd.merge(filtered_rows, df, on='file_name', how='left') 
            res.append(result)

        return pd.concat(res, ignore_index=True)

