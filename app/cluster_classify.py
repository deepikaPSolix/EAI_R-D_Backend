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
    
    def cluster_data(self, data: pd.DataFrame):
        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform(data['data'])

        optimal_k, kmeans_model = self._find_optimal_clusters(tfidf_matrix)
        print(f"Optimal number of clusters: {optimal_k}")

        return kmeans_model.labels_

    def cluster_classify(self, data: List[DocFile], partition_size = 0.7) -> list[DocFile]:
        data_partition = int(np.floor(len(data) * partition_size))
        cluster_data = data[:data_partition]
        classification_data = data[data_partition:]

        clustering_df = pd.DataFrame([d.model_dump(include={'file_name', 'data'}) for d in cluster_data])
        classfication_df = pd.DataFrame([d.model_dump(include={'file_name', 'data'})for d in classification_data])
        
        clusters = self.cluster_data(clustering_df)

        clustering_df['cluster_labels'] = clusters

        cluster_X = clustering_df['data']
        cluster_y = clustering_df['cluster_labels']

    
        self.train_classifier(cluster_X, cluster_y)

        classfication_df = self.classify_data(classification_data)

        final_df = pd.concat([clustering_df, classfication_df], ignore_index=True)

        results = {
            row['file_name']: row['cluster_labels']
            for _, row in final_df.iterrows()
        }


        return results
    
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
        classfication_df = pd.DataFrame([d.model_dump(include={'file_name', 'data'}) for d in data])
        classify_X = classfication_df['data']

        with open(os.path.join(self.ml_model_path, 'tfidf_vectorizer.pkl'), 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        classify_X_tfidf = tfidf_vectorizer.transform(classify_X)

        with open(os.path.join(self.ml_model_path, 'doc_classifier_model.pkl'), 'rb') as file:
            classifier = pickle.load(file)

        predicted_labels = classifier.predict(classify_X_tfidf)
        classfication_df['cluster_labels'] = predicted_labels
        return classfication_df

    
    def generate_cluster_labels(self, data:list[DocFile]):
        model = LLMModel()
        cluster_df = pd.DataFrame([d.model_dump() for d in data])
        res = []
        for c in cluster_df["cluster"].unique():
            # filtered_rows = cluster_df[cluster_df['cluster'] == c].sample(n=10)
            rows = [
                f'file_name: {row.file_name}, chunks: {" ".join([c.text for c in row.chunks[:min(10, len(row.chunks))]])}'
                for row in cluster_df[cluster_df['cluster'] == c].sample(n=min(10, len(cluster_df[cluster_df['cluster'] == c]))).itertuples()
            ]
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
            print("Cluster_Value: " + str(c))
            print(q_res.model_dump())
            cluster_df.loc[cluster_df['cluster'] == c, 'cluster_label'] = q_res.label

        for original_doc, cluster_label in zip(data, cluster_df['cluster_label']):
            original_doc.cluster_label = cluster_label

        return data


class Clustering:
    def __init__(self, max_clusters=5):
        self.max_clusters = max_clusters

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
                
        silhouette_k = cluster_range[np.argmax(silhouette_scores)]
        
        best_k = silhouette_k
        
        kmeans_model = KMeans(n_clusters=best_k, random_state=42)
        kmeans_model.fit(data)
        
        return best_k, kmeans_model
    
    def cluster_data(self, data: pd.DataFrame):
        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform(data['data'])

        optimal_k, kmeans_model = self._find_optimal_clusters(tfidf_matrix)
        print(f"Optimal number of clusters: {optimal_k}")

        return kmeans_model.labels_


class Classification:
    def __init__(self, model_path="cache/ml-model"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

    def train(self, data, labels):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data)

        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(tfidf_matrix, labels)

        self._save_model(classifier, 'doc_classifier_model.pkl')
        self._save_model(vectorizer, 'tfidf_vectorizer.pkl')

    def classify(self, text_data: pd.Series):
        vectorizer = self._load_model('tfidf_vectorizer.pkl')
        classifier = self._load_model('doc_classifier_model.pkl')

        tfidf_matrix = vectorizer.transform(text_data)
        return classifier.predict(tfidf_matrix)

    def _save_model(self, model, filename):
        try:
            with open(os.path.join(self.model_path, filename), 'wb') as file:
                pickle.dump(model, file)
        except Exception as e:
            print(f"Error saving model: {e}")

    def _load_model(self, filename):
        try:
            with open(os.path.join(self.model_path, filename), 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


class LabelGenerator:
    def __init__(self):
        self.model = LLMModel()

    def generate_labels(self, data: pd.DataFrame, cluster_column="cluster"):

        for cluster_id in data[cluster_column].unique():
            cluster_samples = data[data[cluster_column] == cluster_id].sample(
                n=min(10, len(data[data[cluster_column] == cluster_id]))
            )
            rows = [
                f'file_name: {row.file_name}, chunks: {" ".join([c.text for c in row.chunks[:min(10, len(row.chunks))]])}'
                for row in cluster_samples.itertuples()
            ]

            query = f"""
            You are provided with 5 chunks of text from each of 10 documents belonging to the same cluster. 
            These chunks are a representative sample of the cluster's content. Your job is to analyze the provided 
            data and generate a single, concise, and meaningful label that represents the central theme or topic 
            of the entire cluster.

            Cluster Data:
            {rows}
            """
            response = self.model.infer_model(query, GovernanceModel)
            label = response.model_dump().get("label", "Unknown")
            data.loc[data[cluster_column] == cluster_id, "cluster_label"] = label

        return data