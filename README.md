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
The `LLMModel` class interacts with a Language Learning Model (LLM) using the ChatTogether API. It initializes a rate-limited model and processes user queries to produce structured responses.

### **Methods**

1. **`_initialize_model(self)`**
   - **Purpose:** Initializes the ChatTogether model with a specified temperature and rate limiter to control request frequency.
   - **Rate Limiter Configuration:**
     - **requests_per_second:** Limits the model to one request per second.
     - **check_every_n_seconds:** Sets the rate limiter to check every 100 ms.
     - **max_bucket_size:** Allows a maximum burst size of up to 10 requests.

2. **`infer_model(self, query, data_model)`**
   - **Purpose:** Generates a structured response from the model based on the provided query and formats the output using a Pydantic model.
   - **Parameters:**
     - `query (str)`: The user’s query input to the LLM.
     - `data_model (Pydantic Model)`: The structure expected for the LLM's response.
   - **Returns:** Parsed output in the form of the specified `data_model`, or `None` if an error occurs.


---

## **3. ClusterAndClassify Class**

### **Overview**
The `ClusterAndClassify` class is designed to handle document clustering and classification. It splits documents into sets for clustering and classification, determines the optimal clusters, and applies an LLM to generate cluster labels and classify sensitivity levels.

### **Methods**

1. **`_find_optimal_clusters(self, data, max_clusters=5)`**
   - **Purpose:** Determines the optimal number of clusters by evaluating inertia (Elbow method) and silhouette scores.
   - **Parameters:**
     - `data`: The TF-IDF matrix of the documents.
     - `max_clusters (int)`: Maximum number of clusters to evaluate.
   - **Returns:** Optimal number of clusters and the trained KMeans model.

2. **`cluster_classify(self, data: List[DocFile], partition_size=0.7)`**
   - **Purpose:** Splits the data for clustering and classification, clusters part of the data, and classifies the rest using a pre-trained classifier.
   - **Parameters:**
     - `data (List[DocFile])`: List of `DocFile` objects to be clustered and classified.
     - `partition_size (float)`: Fraction of data to allocate for clustering.
   - **Returns:** A combined dataframe containing clustered and classified documents with their assigned labels.

3. **`train_classifier(self, data, labels)`**
   - **Purpose:** Trains a RandomForestClassifier on clustered data and saves the model and TF-IDF vectorizer.
   - **Parameters:**
     - `data`: Clustered document data for training.
     - `labels`: Cluster labels assigned to the documents.

4. **`classify_data(self, data: List[DocFile])`**
   - **Purpose:** Classifies the remaining documents in the dataset using a trained model.
   - **Parameters:**
     - `data (List[DocFile])`: List of documents to be classified.
   - **Returns:** A dataframe of classified documents with their predicted cluster labels.

5. **`generate_cluster_labels(self, cluster_df)`**
   - **Purpose:** Utilizes an LLM to generate a cluster label, assign sensitivity levels, identify data types, and assign retention periods for each document in a cluster.
   - **Parameters:**
     - `cluster_df`: Dataframe containing documents and their cluster labels.
   - **Returns:** A dataframe with detailed cluster labels, sensitivity classifications, data types, key data points, and retention periods for each document.


---

## **4. AttributeExtractor Class**

### **Overview**
The `AttributeExtractor` class is designed to process files, extract specific attributes from chunks of text data, and store the extracted attributes in a structured format. It uses a Language Learning Model (LLM) to analyze the text data and provides structured JSON output for each document.

### **Methods**

1. **`extract_from_files(self, files: List[DocFile])`**
   - **Purpose:** Processes a list of `DocFile` objects and extracts structured data from each file.
   - **Parameters:**
     - `files (List[DocFile])`: List of `DocFile` objects containing the document data.
   - **Returns:** A Pandas DataFrame with the combined extracted data for each file.

2. **`extract_data_from_chunks(self, chunks)`**
   - **Purpose:** Analyzes chunks of text data to extract structured attributes by querying the LLM with each chunk.
   - **Parameters:**
     - `chunks`: List of text chunks from a `DocFile`.
   - **Returns:** A list of extracted attribute results from each chunk in JSON format.

3. **`combine_data(self, data, file_name, file_type)`**
   - **Purpose:** Combines extracted data from multiple chunks into a single JSON object for each file.
   - **Parameters:**
     - `data`: List of JSON objects with extracted attributes.
     - `file_name (str)`: Name of the file.
     - `file_type (str)`: Type of the file (e.g., "pdf").
   - **Returns:** A merged JSON object with consolidated data for the file.

4. **`_merge_json_objects(self, json_objects: dict)`**
   - **Purpose:** Merges multiple JSON objects into a single dictionary, handling conflicts and combining lists or boolean values appropriately.
   - **Parameters:**
     - `json_objects (dict)`: List of JSON objects containing attributes to merge.
   - **Returns:** A merged dictionary with consolidated attribute data.


---

## **5. ChromaDB Class**

### **Overview**
The `ChromaDB` class is a singleton class for managing a ChromaDB collection. It provides methods to create and manage collections of document data, with functions to add, update, retrieve, and query documents based on metadata attributes.

### **Methods**

1. **`create_collection(self, name='documents')`**
   - **Purpose:** Creates or retrieves a ChromaDB collection.
   - **Parameters:**
     - `name (str)`: Name of the collection to create or retrieve.
   - **Returns:** A ChromaDB collection object.

2. **`add_documents(self, data_df)`**
   - **Purpose:** Adds new documents to the ChromaDB collection.
   - **Parameters:**
     - `data_df (DataFrame)`: DataFrame containing document data, with columns such as `file_name`, `data`, `label`, `sensitivity`, `data_classifiers`, `responsible_values`, `retention_time`, and `attributes`.
   - **Returns:** `True` if documents are successfully added, `False` otherwise.

3. **`update_documents(self, data_dict)`**
   - **Purpose:** Updates existing documents in the ChromaDB collection.
   - **Parameters:**
     - `data_dict (list)`: List of dictionaries, each containing `id`, `data`, `label`, `sensitivity`, and `attributes` for each document to update.
   - **Returns:** `True` if the update is successful, `False` otherwise.

4. **`get(self, ids=None, where=None)`**
   - **Purpose:** Retrieves documents from the collection based on document IDs or specified conditions.
   - **Parameters:**
     - `ids (list)`: Optional; List of document IDs to retrieve.
     - `where (dict)`: Optional; Query conditions to filter the results.
   - **Returns:** List of dictionaries with document metadata and data content.

5. **`query_db(self, query_text, user_role, k=20)`**
   - **Purpose:** Queries the ChromaDB collection based on a query text and filters results by sensitivity level.
   - **Parameters:**
     - `query_text (str)`: The search query text.
     - `user_role (int)`: User's sensitivity level to filter accessible documents.
     - `k (int)`: Optional; The number of results to return, default is 20.
   - **Returns:** List of dictionaries containing relevant document metadata and content.
---

## **6. Server.py**

## Overview
`server.py` is a Flask application designed to handle document uploads, clustering, classification, and attribute extraction. The application utilizes Celery for asynchronous task management and ChromaDB for document storage and retrieval.

## Features
- **Upload and Process Documents**: Supports multiple file uploads for processing.
- **Clustering and Classification**: Uses machine learning techniques to classify documents.
- **Attribute Extraction**: Extracts relevant attributes from the uploaded documents.
- **RAG (Retrieval-Augmented Generation)**: Facilitates user queries with document retrieval based on access levels.
- **Task Management**: Provides task status tracking via Celery.

## Technologies Used
- **Flask**: Web framework for building the API.
- **Celery**: Asynchronous task queue for handling long-running processes.
- **ChromaDB**: Database for storing document metadata and attributes.
- **Pandas**: Data manipulation and analysis.
- **Redis**: Message broker for Celery tasks.


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

