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
            Task: Cluster Document Analysis

You are provided with a set of documents in a cluster. Perform the following tasks accurately:

1. Cluster Labeling: Generate a specific and accurate label that reflects the content of the entire cluster. Only return the cluster name and nothing else.

2. Sensitivity Classification: For each document, assign a sensitivity level as an integer from the following categories:
   - 1: Public Data (General public access, e.g., reports, statistics)
   - 2: Internal Data (Operational documents for internal use)
   - 3: Confidential Data (Personal/sensitive data shared on a need-to-know basis)
   - 4: Restricted Data (Highly sensitive information, access limited to specific personnel)
   - 5: Private Data (Detailed personal records, protected by privacy laws)
   - 6: Critical Data (Vital for urgent care or life-saving decisions)
   - 7: Regulatory Data (Compliance with legal/regulatory requirements)

   Carefully analyze any document containing PII, trade secrets, legal, medical, or intellectual property. These should be classified as level 3 or higher. Ensure critical or regulatory data is assigned levels 6 or 7.

3. List the attributes that are found in the data that justify the sensitivity level. Only give the list in the form CSV values and nothing else.

4. Retention Period: Assign the appropriate retention period for each document based on its type, specifying the time in years and months.

Return the output strictly in the requested JSON format, without additional information.

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

