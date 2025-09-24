from flask import current_app
import pandas as pd
import json
import numpy as np
import os

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
import itertools
import psutil, gc
import sqlalchemy
from sqlalchemy import text, select, MetaData, Table
from sqlalchemy.sql import expression

def get_all_results(chroma_client, engine, limit):
    # Get all tables
    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()

    return get_embeddings_for_tables(tables, chroma_client, engine, limit), get_profiling_for_tables(tables, engine, limit)

def get_embeddings_for_tables(tables, chroma_client, engine, limit):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = MODEL
    for table in tables:
        ### Embedding
        embed_table_columns(table, chroma_client, engine, model, limit)
        current_app.logger.info(f"Embedding for table '{table}' completed")

    embedding_result = compare_all_tables(tables, chroma_client)
    current_app.logger.info(f"Embedding comparison across all tables completed")
    return embedding_result

def get_profiling_for_tables(tables, engine, limit):
    profiling_result = []
    for table in tables:
        ### Profiling
        profiling_result.append({table: generate_profiling_report(table, engine, limit)})
        current_app.logger.info(f"Profiling for table '{table}' completed")

    current_app.logger.info(f"Profiling across all tables completed")
    return profiling_result

def _summarize_column(series: pd.Series, max_values: int = 512, max_chars: int = 4000) -> str:
    # representative, bounded, deterministic
    s = series.dropna().astype(str)
    uniq = pd.unique(s.values)
    if len(uniq) > max_values:
        uniq = uniq[:max_values]
    txt = " ".join(uniq)
    return txt[:max_chars]

def _safe_table_head(engine: sqlalchemy.Engine, table_name: str, limit: int = 100, schema: str | None = None) -> pd.DataFrame:
    """
    Read first N rows *from the database* (no full scan into Pandas),
    skip huge column types, set short timeouts, and stream results.
    """
    md = MetaData()
    tbl = Table(table_name, md, schema=schema, autoload_with=engine)

    # choose columns to avoid LOB/JSON fields
    cols = list(tbl.c)

    stmt = select(*cols).limit(limit)
    with engine.connect() as conn:
        try:
            result = conn.execute(stmt)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            current_app.logger.error(f"Error fetching data from {table_name}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    return df

def embed_table_columns(table_name, chroma_client, engine, model, limit):
    current_app.logger.info(f"Creating embeddings for {table_name}")
    collection = chroma_client.get_or_create_collection(table_name)
    current_app.logger.info(f"Collection for {table_name}")

    # 🔻 THIS is the critical change
    # df = pd.read_sql_table(table_name, engine).head(limit)
    df = _safe_table_head(engine, table_name, limit=limit)
    current_app.logger.info(f"Table {table_name} read: {len(df)} rows, {len(df.columns)} cols")

    ncols = len(df.columns)
    current_app.logger.info(f"Embedding {len(df)} rows of {ncols} columns for table {table_name}")

    try:
        model.max_seq_length = 256
    except Exception:
        pass

    for i, col in enumerate(df.columns, start=1):
        current_app.logger.info(f"[{i}/{ncols}] {table_name}.{col}")
        s = df[col]
        if s.isnull().all():
            current_app.logger.info("Null")
            continue

        text = _summarize_column(s, max_values=512, max_chars=4000)
        if not text:
            current_app.logger.info("No values")
            continue
        
        try:
            emb = model.encode([text], show_progress_bar=False)[0]

            collection.add(
                documents=[text],
                embeddings=[emb.tolist()],
                metadatas=[{"column": col, "table": table_name}],
                ids=[f"{table_name}_{col}"]
            )
        except Exception as e:
            current_app.logger.error(f"Error adding embeddings to collection for {table_name}.{col}: {e}")

        # rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        # current_app.logger.info(f"DEBUG: Mem after {table_name}.{col}: {rss:.2f} MB")
        del text, emb
        gc.collect()

    current_app.logger.info(f"Column embeddings for table '{table_name}' stored in ChromaDB.")

def compare_all_tables(tables, chroma_client):
    # Compare embeddings across all tables without duplicates
    current_app.logger.info("Comparing all tables")

    result = []
    for i, table1 in enumerate(tables):
        for table2 in tables[i + 1:]:
            result.extend(compare_tables(table1, table2, chroma_client))

    return result

def compare_one_table(table_name, tables, chroma_client):
    # Compare a specific table with all others
    current_app.logger.info(f"Comparing table {table_name} with all other tables")
    for other_table in tables:
        if other_table != table_name:
            compare_tables(table_name, other_table, chroma_client)

def compare_tables(table1, table2, chroma_client):
    # Compare embeddings between two tables
    current_app.logger.info(f"Comparing {table1} with {table2}")
    try:
        collection1 = chroma_client.get_collection(table1)
        collection2 = chroma_client.get_collection(table2)
    except Exception as e:
        current_app.logger.info(f"Error getting collections: {e}")
        return []

    # Get all documents, embeddings, and metadatas from collection1
    try:
        data1 = collection1.get(include=["documents", "embeddings", "metadatas"])
    except Exception as e:
        current_app.logger.info(f"Error fetching data from collection '{table1}': {e}")
        return []

    docs1 = data1.get("documents", [])
    embeds1 = data1.get("embeddings", [])
    metas1 = data1.get("metadatas", [])

    if not embeds1.size:
        current_app.logger.info(f"No embeddings found in collection '{table1}'. Skipping comparison.")
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
            current_app.logger.info(f"Error querying collection '{table2}': {e}")
            continue

        # current_app.logger.info results if any found
        if query_result.get("ids") and query_result["ids"][0] and query_result["distances"][0][0] < 0.5:
            doc2_meta = query_result["metadatas"][0][0]
            doc2_doc = query_result["documents"][0][0]
            res_str = ''
            # res_str += f"Found similar column: {table1}.{metas1[idx]['column']} == {table2}.{doc2_meta['column']}" + "\n"
            res_str += f"{table1}.{metas1[idx]['column']} == {table2}.{doc2_meta['column']}" + "\n"
            res_str += f"Distance: {query_result['distances'][0][0]}" + "\n"
            # res_str += f"{table1}.{metas1[idx]['column']} content preview: " + docs1[idx][:100] + "\n"
            # res_str += f"{table2}.{doc2_meta['column']} content preview: " + doc2_doc[:100]
            current_app.logger.info(res_str + "\n")
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
def generate_profiling_report(table_name, engine, limit):
    result = generate_profiling_report_custom(table_name, engine, limit)
    return result

def generate_profiling_report_custom(table_name, engine, limit):
    # df = pd.read_sql_table(table_name, engine)
    # df = df[0:limit]  # Limit to specified rows for performance
    df = _safe_table_head(engine, table_name, limit=limit)
    current_app.logger.info(f"Generating profiling report for {len(df)} rows of {len(df.columns)} columns for table {table_name}")

    report = {}
    metadata = MetaData()
    table_obj = Table(table_name, metadata, autoload_with=engine)
            
    # determine if values are unique, how many values are missing, the data type of the values (numeric or text),
    # count of values, max value or length, min value or length
    for column in df.columns:
        report[column] = {}

        # Get data type
        # report[column]['data_type'] = str(df[column].dtype)
        with engine.connect() as conn:
            # Number of rows in column
            try:
                count_stmt = select(sqlalchemy.func.count(table_obj.c[column]))
                count_result = conn.execute(count_stmt)
                report[column]['count'] = count_result.scalar()
            except Exception as e:
                current_app.logger.error(f"Error fetching row count for {table_name}.{column}: {e}")
                report[column]['count'] = 0
            if report[column]['count'] == 0:
                continue  # Skip empty columns

            # Get column data type using SQL statement
            dtype_stmt = f"SELECT data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{column}'"
            try:
                result = conn.execute(text(dtype_stmt))
                col_type_row = result.fetchone()
                if col_type_row and col_type_row[0]:
                    report[column]['data_type'] = col_type_row[0]
                else:
                    # Fallback to pandas dtype
                    if pd.api.types.is_numeric_dtype(df[column]):
                        report[column]['data_type'] = 'numeric'
                    elif pd.api.types.is_string_dtype(df[column]):
                        report[column]['data_type'] = 'text'
                    elif pd.api.types.is_datetime64_any_dtype(df[column]):
                        report[column]['data_type'] = 'datetime'
                    else:
                        report[column]['data_type'] = 'other'
            except Exception as e:
                current_app.logger.error(f"Error fetching column type for {table_name}.{column}: {e}")
                # Fallback to pandas dtype
                if pd.api.types.is_numeric_dtype(df[column]):
                    report[column]['data_type'] = 'numeric'
                elif pd.api.types.is_string_dtype(df[column]):
                    report[column]['data_type'] = 'text'
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    report[column]['data_type'] = 'datetime'
                else:
                    report[column]['data_type'] = 'other'

            # Get null counts from the SQL query
            try:
                null_stmt = select(sqlalchemy.func.count()).where(table_obj.c[column].is_(None))
                null_result = conn.execute(null_stmt)
                report[column]['n_missing'] = null_result.scalar() # number of null values
            except Exception as e:
                current_app.logger.error(f"Error fetching null count for {table_name}.{column}: {e}")

            # Get number of distinct values
            try:
                distinct_stmt = select(sqlalchemy.func.count(table_obj.c[column].distinct())).where(table_obj.c[column].isnot(None))
                distinct_result = conn.execute(distinct_stmt)
                report[column]['n_distinct'] = distinct_result.scalar() # number of distinct values
            except Exception as e:
                current_app.logger.error(f"Error fetching distinct count for {table_name}.{column}: {e}")
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
        # report[column]['n_distinct'] = df[column].nunique() # number of unique values
        # report[column]['p_distinct'] = report[column]['n_distinct'] / report[column]['n']
        if "n_distinct" in report[column]:
            report[column]['is_unique'] = report[column]['n_distinct'] == report[column]['count'] # if all values are unique

    return report
