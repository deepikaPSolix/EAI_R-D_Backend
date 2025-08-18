import sqlalchemy
import pandas as pd
from ydata_profiling import ProfileReport
import json
import numpy as np
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoids HF tokenizer thread storms

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
import itertools
import psutil, gc

def get_all_results(chroma_client, engine, limit = 1000):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    profiling_result = []
    # Get all tables
    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()

    for table in tables:
        ### Embedding
        embed_table_columns(table, chroma_client, engine, model, limit)
        print(f"Embedding for table '{table}' completed")
        ### Profiling
        profiling_result.append({table: generate_profiling_report(table, engine, limit)})
        print(f"Profiling for table '{table}' completed\n")

    embedding_result = compare_all_tables(tables, chroma_client)
    print(f"Embedding comparison across all tables completed")
    return embedding_result, profiling_result

def get_results_for_tables(tables, chroma_client, engine, limit = 1000):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    profiling_result = []
    for table in tables:
        ### Embedding
        embed_table_columns(table, chroma_client, engine, MODEL, limit)
        print(f"Embedding for table '{table}' completed")
        ### Profiling
        profiling_result.append({table: generate_profiling_report(table, engine, limit)})
        print(f"Profiling for table '{table}' completed")

    embedding_result = compare_all_tables(tables, chroma_client)
    print(f"Embedding comparison across all tables completed")
    return embedding_result, profiling_result

#---
# Global in worker
_WORKER_MODEL = None

def _init_model_worker(model_name: str, device: str = "cpu"):
    """Initializer runs once per worker."""
    global _WORKER_MODEL
    _WORKER_MODEL = SentenceTransformer(model_name, device=device)

def _embed_text_worker(text: str):
    """Runs inside worker; model is preloaded."""
    global _WORKER_MODEL
    emb = _WORKER_MODEL.encode([text])[0]
    # return list (JSON/pickling safe)
    return emb.tolist()

@contextmanager
def embedding_pool(model_name: str, processes: int = 1, device: str = "cpu"):
    """
    Spawn-based pool that preloads the model once per worker.
    Use context manager so we always close/terminate cleanly.
    """
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=processes, initializer=_init_model_worker, initargs=(model_name, device))
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def _sample_column_values(series: pd.Series, max_values: int = 2000):
    """
    Keep it bounded and representative.
    - Dropna, cast to str
    - Take up to max_values (unique then head)
    """
    s = series.dropna().astype(str)
    # Prefer unique sample first to avoid one value dominating
    uniq = pd.unique(s.values)
    if len(uniq) > max_values:
        uniq = uniq[:max_values]
    return " ".join(uniq)

def _chunks(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

#---
def _summarize_column(series: pd.Series, max_values: int = 512, max_chars: int = 4000) -> str:
    # representative, bounded, deterministic
    s = series.dropna().astype(str)
    uniq = pd.unique(s.values)
    if len(uniq) > max_values:
        uniq = uniq[:max_values]
    txt = " ".join(uniq)
    return txt[:max_chars]

def embed_table_columns(table_name, chroma_client, engine, model, limit=1000):
    print(f"Creating embeddings for {table_name}")
    collection = chroma_client.get_or_create_collection(table_name)

    df = pd.read_sql_table(table_name, engine).head(limit)
    ncols = len(df.columns)
    print(f"Embedding {len(df)} rows of {ncols} columns for table {table_name}")

    # enforce small context; MiniLM defaults to 256 tokens but make it explicit
    try:
        model.max_seq_length = 256
    except Exception:
        pass

    for i, col in enumerate(df.columns, start=1):
        print(f"[{i}/{ncols}] {table_name}.{col}")
        s = df[col]
        if s.isnull().all():
            print("null")
            continue

        text = _summarize_column(s, max_values=512, max_chars=4000)
        if not text:
            print("no values")
            continue

        emb = model.encode([text], show_progress_bar=False)[0]  # sync, no tqdm

        # collection.add(
        #     documents=[text],
        #     embeddings=[emb.tolist()],
        #     metadatas=[{"column": col, "table": table_name}],
        #     ids=[f"{table_name}_{col}"]
        # )

        process = psutil.Process(os.getpid())
        print(f"DEBUG: Mem after {table_name}.{col}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        del text, emb
        gc.collect()

    print(f"Column embeddings for table '{table_name}' stored in ChromaDB.")
    
#---

# def embed_table_columns(table_name, chroma_client, engine, model, limit = 1000):
#     print(f"Creating embeddings for {table_name}")
#     # Get or create collection for the table
#     collection = chroma_client.get_or_create_collection(table_name)

#     # Initialize embedding model
#     model = model

#     # Get table data
#     table_df = pd.read_sql_table(table_name, engine)
#     table_df = table_df[0:limit]  # Limit to specified rows for performance

#     # For each column, fetch all values, concatenate as string, and embed
#     print(f"Embedding {len(table_df)} rows of {len(table_df.columns)} columns for table {table_name}")
#     for col in table_df.columns:
#         print(f"Processing column: {table_name}.{col}")
#         # If col is only null, skip column
#         if table_df[col].isnull().all():
#             print("null")
#             continue
#         values = table_df[col].to_list()
#         if not values:
#             continue
#         text = " ".join(map(str, values))
#         print(f"DEBUG: Encoding type: {type(text)}, length: {len(text)}")
#         embedding = model.encode([text])[0]
        
#         # Store the embedding in ChromaDB
#         # # print(f"Values for column '{col}' in table '{table_name}': {values}")
#         # collection.add(
#         #     documents=[text],
#         #     embeddings=[embedding.tolist()],
#         #     metadatas=[{"column": col, "table": table_name}],
#         #     ids=[f"{table_name}_{col}"]
#         # )

#         # Print memory usage and force garbage collection
#         process = psutil.Process(os.getpid())
#         print(f"DEBUG: Memory usage after embedding: {process.memory_info().rss / 1024 ** 2:.2f} MB")
#         gc.collect()
#         print("DEBUG: Garbage collection complete")
#     print(f"Column embeddings for table '{table_name}' stored in ChromaDB.")

# ---
# def embed_table_columns(table_name, chroma_client, engine, limit=1000,
#                         model_name='all-MiniLM-L6-v2',
#                         processes: int = 1,
#                         timeout_sec: int = 60,
#                         per_col_max_values: int = 2000,
#                         chunk_chars: int = 4000):
#     """
#     Embed each column using a persistent spawn pool.
#     - Preload model once per worker
#     - Sample/limit per column
#     - Chunk long text to avoid huge single strings (then average)
#     """
#     print(f"Creating embeddings for {table_name}")
#     collection = chroma_client.get_or_create_collection(table_name)

#     df = pd.read_sql_table(table_name, engine).head(limit)
#     print(f"Embedding {len(df)} rows of {len(df.columns)} columns for table {table_name}")

#     with embedding_pool(model_name=model_name, processes=processes, device="cpu") as pool:
#         for col in df.columns:

#             print(f"Processing column: {table_name}.{col}")
#             s = df[col]
#             if s.isnull().all():
#                 print("null")
#                 continue

#             # Build bounded representation
#             base_text = _sample_column_values(s, max_values=per_col_max_values)
#             if not base_text:
#                 print("no values")
#                 continue

#             # Chunk long text, embed each chunk, average for stability
#             texts = []
#             if len(base_text) <= chunk_chars:
#                 texts = [base_text]
#             else:
#                 # rough splitting without breaking words too much
#                 for i in range(0, len(base_text), chunk_chars):
#                     texts.append(base_text[i:i+chunk_chars])

#             async_results = [pool.apply_async(_embed_text_worker, (t,)) for t in texts]

#             embeddings = []
#             for ar in async_results:
#                 try:
#                     emb = ar.get(timeout=timeout_sec)
#                     if emb is not None:
#                         embeddings.append(np.array(emb, dtype=np.float32))
#                 except mp.TimeoutError:
#                     print(f"ERROR: Embedding chunk timeout for {table_name}.{col}")
#                 except Exception as e:
#                     print(f"ERROR during chunk embedding for {table_name}.{col}: {e}")

#             if not embeddings:
#                 print("embedding none")
#                 continue

#             # Average chunk embeddings for the column
#             col_embedding = np.mean(embeddings, axis=0)
#             print(f"DEBUG: Embedding length: {len(col_embedding)}")
#             print("checkpoint 1")

#             # Store in ChromaDB (UNCOMMENT when ready)
#             # collection.add(
#             #     documents=[" ".join(texts)],
#             #     embeddings=[col_embedding.tolist()],
#             #     metadatas=[{"column": col, "table": table_name}],
#             #     ids=[f"{table_name}_{col}"]
#             # )
#             print("checkpoint 2")
#             process = psutil.Process(os.getpid())
#             print(f"DEBUG: Memory usage after embedding: {process.memory_info().rss / 1024 ** 2:.2f} MB")
#             gc.collect()
#             print("DEBUG: Garbage collection complete")

#     print(f"Column embeddings for table '{table_name}' stored in ChromaDB.")

#---
# def embed_column_worker(text, model_name, conn):
#     try:
#         model = SentenceTransformer(model_name)
#         embedding = model.encode([text])[0]
#         conn.send(embedding.tolist())  # send as list for pickling safety
#     except Exception as e:
#         conn.send(None)
#     finally:
#         conn.close()

# def embed_table_columns(table_name, chroma_client, engine, limit = 1000):
#     if table_name == "activity_stds_lookup":
#         return
#     print(f"Creating embeddings for {table_name}")
#     # Get or create collection for the table
#     collection = chroma_client.get_or_create_collection(table_name)
#     model_name = 'all-MiniLM-L6-v2'

#     # Get table data
#     table_df = pd.read_sql_table(table_name, engine)
#     table_df = table_df[0:limit]  # Limit to specified rows for performance

#     # For each column, fetch all values, concatenate as string, and embed
#     print(f"Embedding {len(table_df)} rows of {len(table_df.columns)} columns for table {table_name}")
#     for col in table_df.columns:
#         if (col == "normal_range_max" or col == "normal_range_min") and table_name == "activity_stds_lookup": # skip this column
#             continue
#         print(f"Processing column: {table_name}.{col}")
#         # If col is only null, skip column
#         if table_df[col].isnull().all():
#             print("null")
#             continue
#         print("checkpoint .1")
#         values = table_df[col].to_list()
#         print("checkpoint .2")
#         if not values:
#             print("no values")
#             continue
#         print("checkpoint .3")
#         text = " ".join(map(str, values))
#         print("checkpoint .4")
#         print(f"DEBUG: Length of text for {table_name}.{col}: {len(text)}")
#         print(f"DEBUG: Encoding type: {type(text)}, length: {len(text)}")
#         try:
#             print(f"DEBUG: Starting embedding subprocess for {table_name}.{col}")
#             parent_conn, child_conn = multiprocessing.Pipe()
#             p = multiprocessing.Process(target=embed_column_worker, args=(text, model_name, child_conn))
#             p.start()
#             p.join(timeout=30)  # Reduced timeout for testing
#             if p.is_alive():
#                 print("is alive")
#                 print(f"ERROR: Embedding process for {table_name}.{col} timed out. Terminating.")
#                 p.terminate()
#                 p.join()
#                 embedding = None
#             else:
#                 print("not alive")
#                 embedding = parent_conn.recv() if parent_conn.poll() else None
#             print(f"DEBUG: Embedding subprocess finished for {table_name}.{col}")
#             if embedding is None:
#                 print("embedding none")
#                 print(f"ERROR: No embedding returned for {table_name}.{col}")
#                 continue
#             print(f"DEBUG: Embedding length: {len(embedding)}")
#             print("checkpoint 1")
#         except Exception as e:
#             print(f"ERROR during embedding for {table_name}.{col}: {e}")
#             continue
        
#         # Store the embedding in ChromaDB
#         # print(f"Values for column '{col}' in table '{table_name}': {values}")
#         # collection.add(
#         #     documents=[text],
#         #     embeddings=[embedding.tolist()],
#         #     metadatas=[{"column": col, "table": table_name}],
#         #     ids=[f"{table_name}_{col}"]
#         # )
#         print("checkpoint 2")

#         # Print memory usage and force garbage collection
#         process = psutil.Process(os.getpid())
#         print(f"DEBUG: Memory usage after embedding: {process.memory_info().rss / 1024 ** 2:.2f} MB")
#         gc.collect()
#         print("DEBUG: Garbage collection complete")

#     print(f"Column embeddings for table '{table_name}' stored in ChromaDB.")

#---

def compare_all_tables(tables, chroma_client):
    # Compare embeddings across all tables without duplicates
    print("Comparing all tables")

    result = []
    for i, table1 in enumerate(tables):
        for table2 in tables[i + 1:]:
            result.extend(compare_tables(table1, table2, chroma_client))

    return result

def compare_one_table(table_name, tables, chroma_client):
    # Compare a specific table with all others
    print(f"Comparing table {table_name} with all other tables")
    for other_table in tables:
        if other_table != table_name:
            compare_tables(table_name, other_table, chroma_client)

def compare_tables(table1, table2, chroma_client):
    # Compare embeddings between two tables
    print(f"Comparing {table1} with {table2}")
    try:
        collection1 = chroma_client.get_collection(table1)
        collection2 = chroma_client.get_collection(table2)
    except Exception as e:
        print(f"Error getting collections: {e}")
        return []

    # Get all documents, embeddings, and metadatas from collection1
    try:
        data1 = collection1.get(include=["documents", "embeddings", "metadatas"])
    except Exception as e:
        print(f"Error fetching data from collection '{table1}': {e}")
        return []

    docs1 = data1.get("documents", [])
    embeds1 = data1.get("embeddings", [])
    metas1 = data1.get("metadatas", [])

    if not embeds1.size:
        print(f"No embeddings found in collection '{table1}'. Skipping comparison.")
        return []

    result = []

    # Compare embeddings
    for idx, embedding in enumerate(embeds1):
        try:
            query_result = collection2.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Error querying collection '{table2}': {e}")
            continue

        # Print results if any found
        if query_result.get("ids") and query_result["ids"][0] and query_result["distances"][0][0] < 0.5:
            doc2_meta = query_result["metadatas"][0][0]
            doc2_doc = query_result["documents"][0][0]
            res_str = ''
            # res_str += f"Found similar column: {table1}.{metas1[idx]['column']} == {table2}.{doc2_meta['column']}" + "\n"
            res_str += f"{table1}.{metas1[idx]['column']} == {table2}.{doc2_meta['column']}" + "\n"
            res_str += f"Distance: {query_result['distances'][0][0]}" + "\n"
            # res_str += f"{table1}.{metas1[idx]['column']} content preview: " + docs1[idx][:100] + "\n"
            # res_str += f"{table2}.{doc2_meta['column']} content preview: " + doc2_doc[:100]
            print(res_str + "\n")
            result.append(res_str)
    
    return result

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# Generate profiling report
def generate_profiling_report(table_name, engine, limit = 1000):
    result = generate_profiling_report_custom(table_name, engine, limit)
    # result = generate_profiling_report_ydata(table_name, engine, limit)
    return result

def generate_profiling_report_custom(table_name, engine, limit):
    df = pd.read_sql_table(table_name, engine)
    df = df[0:limit]  # Limit to specified rows for performance
    print(f"Generating profiling report for {len(df)} rows of {len(df.columns)} columns table {table_name}")

    report = {}
    # determine if values are unique, how many values are missing, the data type of the values (numeric or text),
    # count of values, max value or length, min value or length
    for column in df.columns:
        report[column] = {}
        report[column]['n'] = len(df[column])
        if report[column]['n'] == 0:
            continue  # Skip empty columns

        # report[column]['data_type'] = str(df[column].dtype)
        if pd.api.types.is_numeric_dtype(df[column]):
            report[column]['data_type'] = 'numeric'
        elif pd.api.types.is_string_dtype(df[column]):
            report[column]['data_type'] = 'text'
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            report[column]['data_type'] = 'datetime'
        else:
            report[column]['data_type'] = 'other'

        report[column]['n_missing'] = df[column].isnull().sum()
        # report[column]['p_missing'] = report[column]['n_missing'] / report[column]['n']
        # report[column]['count'] = df[column].count()
        # report[column]['memory_size'] = df[column].memory_usage(deep=True)
        # first 5 non null rows
        # report[column]['first_rows'] = df[column].dropna().head(5).tolist()
        # if df[column].dtype == 'object':
        #     report[column]['max_length'] = df[column].str.len().max()
        #     report[column]['mean_length'] = df[column].str.len().mean()
        #     report[column]['median_length'] = df[column].str.len().median()
        #     report[column]['min_length'] = df[column].str.len().min()
        # else:
        #     report[column]['max_length'] = df[column].max()
        #     report[column]['mean_length'] = df[column].mean()
        #     report[column]['median_length'] = df[column].median()
        #     report[column]['min_length'] = df[column].min()
        report[column]['n_distinct'] = df[column].nunique()
        # report[column]['p_distinct'] = report[column]['n_distinct'] / report[column]['n']
        report[column]['is_unique'] = df[column].is_unique

    return report


# def generate_profiling_report_ydata(table_name, engine, limit):
#     df = pd.read_sql_table(table_name, engine)
#     df = df[0:limit]  # Limit to specified rows for performance
#     print(f"Generating profiling report for {len(df)} rows of {len(df.columns)} columns table {table_name}")
#     profile = ProfileReport(df, minimal=True)

#     # Save report to JSON
#     json_data = profile.to_json()
#     # Convert JSON string to dictionary
#     json_data = json.loads(json_data)

#     # Get only the relevant sections
#     result_data = {
#         "alerts": json_data["alerts"],
#         "correlations": json_data["correlations"]
#     }
#     variables = {}
#     for column in json_data["variables"]:
#         # print(json_data["variables"][column])
#         # List of keys to extract if they exist
#         keys = [
#             "n_distinct", "p_distinct", "is_unique", "n_unique", "p_unique", "type", "hashable",
#             "n_missing", "n", "p_missing", "count", "memory_size", "first_rows",
#             "max_length", "mean_length", "median_length", "min_length"
#         ]
#         variables[column] = {}
#         for key in keys:
#             if key in json_data["variables"][column]:
#                 variables[column][key] = json_data["variables"][column][key]
#     result_data["variables"] = variables
#     return result_data

