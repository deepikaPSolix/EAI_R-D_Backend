# **Project Documentation**

## **1. DocFile Class**

### **Overview**
The `DocFile` class is responsible for processing a document file, extracting its content, chunking the data, and storing the result in a structured format.

### **Methods**
1. **`_extract_data_chunks(self, file_path: str)`**
   - **Purpose:** Uses the `unstructured.partition.auto.partition` method to extract elements from the document and chunks the data.
   - **Parameters:** 
     - `file_path (str)`: Path to the document.
   - **Returns:** A list of text chunks.

2. **`to_dict(self)`**
   - **Purpose:** Converts the `DocFile` object into a dictionary with the file name and data.
   - **Returns:** A dictionary with the following keys:
     - `"file_name"`: The name of the file.
     - `"data"`: The combined text data extracted from the file.

---

## **2. LLMModel Class**

### **Overview**
The `LLMModel` class interacts with a Language Learning Model (LLM) using the `ChatTogether` API. It initializes the model and processes user queries to infer structured responses.

### **Methods**
1. **`_initialize_model(self)`**
   - **Purpose:** Initializes the `ChatTogether` model with a specified temperature setting.

2. **`infer_model(self, query, data_model)`**
   - **Purpose:** Generates a response from the model based on the provided query and parses the output into a structured format.
   - **Parameters:**
     - `query (str)`: The user's query to the LLM.
     - `data_model (Pydantic Model)`: The expected output format for the LLM's response.
   - **Returns:** Parsed output in the form of the `data_model`.

---

## **3. ClusterAndClassify Class**

### **Overview**
The `ClusterAndClassify` class is designed to handle document clustering and classification. It clusters documents using K-Means and classifies remaining documents using a RandomForestClassifier.

### **Methods**
1. **`_find_optimal_clusters(self, data, max_clusters=5)`**
   - **Purpose:** Determines the optimal number of clusters by evaluating inertia (for the elbow method) and silhouette scores.
   - **Parameters:**
     - `data`: The TF-IDF matrix of the documents.
     - `max_clusters (int)`: Maximum number of clusters to evaluate.
   - **Returns:** Optimal number of clusters and the trained K-Means model.

2. **`cluster_classify(self, data: List[DocFile], partition_size=0.7)`**
   - **Purpose:** Splits the input data into a clustering and classification set, clusters a portion of the data, and classifies the rest.
   - **Parameters:**
     - `data (List[DocFile])`: A list of `DocFile` objects.
     - `partition_size (float)`: Proportion of data to use for clustering.
   - **Returns:** A DataFrame containing both clustering and classification results.

3. **`train_classifier(self, data, labels)`**
   - **Purpose:** Trains a RandomForestClassifier using the TF-IDF features and cluster labels.
   - **Parameters:**
     - `data`: The text data used for training.
     - `labels`: The cluster labels.
   - **Side Effects:** Saves the trained classifier and TF-IDF vectorizer as pickle files.

4. **`classify_data(self, data: List[DocFile])`**
   - **Purpose:** Classifies new documents using a pre-trained classifier.
   - **Parameters:**
     - `data (List[DocFile])`: The documents to classify.
   - **Returns:** A DataFrame with classified documents and their predicted cluster labels.

5. **`generate_cluster_labels(self, cluster_df)`**
   - **Purpose:** Generates labels for clusters using an LLM and assigns sensitivity levels to each cluster.
   - **Parameters:**
     - `cluster_df (DataFrame)`: A DataFrame containing clustered documents.
   - **Returns:** A DataFrame with cluster labels and sensitivity levels.

---

## **4. AttributeExtractor Class**

### **Overview**
The `AttributeExtractor` class is responsible for extracting specific attributes from document text chunks using an LLM and processing the extracted data into structured formats.

### **Methods**
1. **`extract_from_files(self, files: List[DocFile])`**
   - **Purpose:** Processes a list of document files, extracts attributes from the file chunks, and returns the structured data as a DataFrame.
   - **Parameters:**
     - `files (List[DocFile])`: A list of `DocFile` objects.
   - **Returns:** A DataFrame containing extracted attributes for each file.

2. **`extract_data_from_chunks(self, chunks)`**
   - **Purpose:** Extracts attributes from each chunk of text in the document by sending queries to the LLM.
   - **Parameters:**
     - `chunks`: A list of text chunks.
   - **Returns:** A list of extracted attributes.

3. **`combine_data(self, data, file_name, file_type)`**
   - **Purpose:** Merges the extracted data and adds metadata such as file name and file type.
   - **Parameters:**
     - `data`: The extracted data from the LLM.
     - `file_name`: The name of the file.
     - `file_type`: The type of the file (e.g., PDF, DOCX).
   - **Returns:** A merged dictionary of attributes and file metadata.

4. **`_merge_json_objects(self, json_objects: dict)`**
   - **Purpose:** Merges multiple JSON objects into a single dictionary, handling key conflicts and combining values intelligently.
   - **Parameters:**
     - `json_objects (dict)`: A list of JSON objects to merge.
   - **Returns:** A single merged JSON dictionary.

---

### **Dependencies**
- **Libraries:**
  - `pandas`: Used for DataFrame operations and structured data handling.
  - `numpy`: For numerical operations.
  - `sklearn`: Provides TF-IDF vectorization, K-Means clustering, silhouette scoring, and RandomForest classification.
  - `pickle`: Used to save and load the trained models (classifier and vectorizer).
  - `LLMModel`: Custom class used to interact with the LLM for inference.
  - `DocFile`: Custom class for document text extraction and chunking.
  - `LabelGovernanceModel`, `ExtractedData`: Custom Pydantic models used for output parsing from LLM.
