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

            1. **Cluster Labeling**: Generate a clear and specific label that accurately represents the cluster. Return only the cluster name.

            2. **Sensitivity Classification**: For each document, assign a sensitivity level (integer) based on the following categories:
            - 1: Public Data (e.g., public reports, statistics)
            - 2: Internal Data (internal use, not for external sharing)
            - 3: Confidential Data (personal or sensitive information)
            - 4: Restricted Data (highly sensitive, access limited)
            - 5: Private Data (personal data protected by privacy laws)
            - 6: Critical Data (vital for urgent care or life-saving actions)
            - 7: Regulatory Data (compliance with legal/regulatory rules)

            Ensure documents containing PII, trade secrets, legal, medical, or intellectual property are classified as level 3 or higher. Assign levels 6 or 7 for critical or regulatory data.

            3. **Top Data Types**: Analyze the document and identify 3 to 5 **distinct** and meaningful data types that justify the sensitivity classification. Do not rely on the example data types provided below—these are only for reference. The identified data types must be directly related to the actual content of the document. Return these data types in CSV format. The data types should be relevant to the document's content, and similar types must not be repeated.

            Reference examples (for understanding only, do not use as output unless relevant):
                - PII
                - PHI
                - EHR Data
                - Medical History
                - Lab Results
                - Prescription Data
                - Patient Satisfaction Surveys
                - Appointment Records
                - Demographic Information
                - Contact Information
                - Health Insurance Details
                - Caregiver Information
                - Clinical Trial Data
                - Adverse Event Reports
                - Imaging Data
                - Genetic Information
                - Diagnosis Codes (ICD-10)
                - Treatment Protocols
                - Medical Devices Information
                - Immunization Records
                - Clinical Notes
                - Anonymized Clinical Notes
                - Research Findings
                - Billing Information
                - Insurance Claims Data
                - Compliance Reports
                - Audit Trails
                - Internal Policies
                - Staff Scheduling Information
                - Facility Management Data
                - Equipment Inventory
                - Financial Reports
                - Strategic Plans
                - Billing Disputes
                - Operational Efficiency Metrics
                - Staffing Levels
                - Risk Management Reports


            4. **Data Points**: Provide up to 5 key-value pairs in dictionary format that contribute to the sensitivity level. Use this format: {{"name1": "value1", "name2": "value2"}}.

            5. **Retention Period**: Assign a retention period for each document based on its type, specifying the time in years and months.

            Return the output strictly in the required JSON format, without any additional information.

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

