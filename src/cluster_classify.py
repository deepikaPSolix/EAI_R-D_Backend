from typing import List
import pandas as pd
import numpy as np
from doc_file import DocFile
from llm_model import LLMModel
from models.label_governance_model import GovernanceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


class ClusterAndClassify:

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

        # df['combined_text'] = df[df.columns].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)


        clustering_df = pd.DataFrame([d.to_dict() for d in cluster_data])
        classfication_df = pd.DataFrame([d.to_dict() for d in classification_data])
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
        with open(os.path.join("ml-model", 'doc_classifier_model.pkl'), 'wb') as file:
            pickle.dump(classifier, file)

        with open(os.path.join("ml-model", 'tfidf_vectorizer.pkl'), 'wb') as file:
            pickle.dump(tfidf_vectorizer, file)

    def classify_data(self, data: List[DocFile]):
        classfication_df = pd.DataFrame([d.to_dict() for d in data])
        classify_X = classfication_df['data']

        with open(os.path.join("ml-model", 'tfidf_vectorizer.pkl'), 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        classify_X_tfidf = tfidf_vectorizer.transform(classify_X)

        with open(os.path.join("ml-model", 'doc_classifier_model.pkl'), 'rb') as file:
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
            rows = []
            for row in filtered_rows.itertuples():
                rows.append(f'file_name: {row.file_name}, data: {row.data}')
            query = f'''
            You are provided with a set of documents from a cluster. Complete the following tasks:

            1. Generate a label for this cluster. The label must be relevant and very specific to the data in the cluster.
            2. For each document, assign a sensitivity level based on these categories:
            - Public Data: 1, Internal Data: 2, Confidential Data: 3, Restricted Data: 4, Private Data: 5, Critical Data: 6, Regulatory Data: 7.
            - Explicitly analyze each document to assign sensitivity individually.
            3. Provide a reason for each document's sensitivity classification mentioning the data points responsible for the sensitivity.
            4. Determine the retention period for each document based on its type. Provide a specific time frame in years and months.

            Only return the output in the requested JSON format. No extra information.

            Cluster Data:  
            {rows}
            '''
            q_res = model.infer_model(query, GovernanceModel)
            print("QRES")
            print(q_res.dict())
            res_dict = q_res.dict()
            df = pd.DataFrame(res_dict['attributes'])
            result = pd.merge(filtered_rows, df, on='file_name', how='left') 
            res.append(result)

        return pd.concat(res, ignore_index=True)

