import psycopg2
import json
from datetime import datetime
from flask import current_app
import pandas as pd
from app.hitl_data_processing import compare_and_update_postgres
#Postgres
from app.postgres_db import DatabaseManager
import os
DB_CONFIG = {
    "host": "192.168.1.116",
    "database": "postgres",
    "user": "postgres",
    "password": "12345",
    "port": "25432"
}
# def create_documents_table():
#     """Creates the documents table if it does not exist."""
#     connection = None
#     cursor = None

#     try:
#         connection = psycopg2.connect(**DB_CONFIG)
#         cursor = connection.cursor()
       
#         create_table_query = """
#         CREATE TABLE IF NOT EXISTS human_altered_table (
#             id UUID PRIMARY KEY,
#             file_name TEXT NOT NULL,
#             attributes JSONB NOT NULL,
#             data_classifiers TEXT,
#             label TEXT NOT NULL, 
#             responsible_values TEXT, 
#             retention_time TEXT,
#             sensitivity INT,
#             reason_for_change TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#         """
#         cursor.execute(create_table_query)
#         connection.commit()
#         print("[PostgreSQL] Table 'human_altered_table' checked/created successfully.")

#     except Exception as e:
#         print(f"[PostgreSQL] Error creating table: {e}")

#     finally:
#         if cursor:
#             cursor.close()
#         if connection:
#             connection.close()


def clean_text(text):
    """Ensures all escape sequences are properly handled before inserting into PostgreSQL."""
    if text:
        return text.replace("\n", " \\n") \
                   .replace("\t", "\\t") \
                   .replace("\b", "")\
                   .replace("\r", "\\r") \
                   .replace("'", "''")  
    return text


def store_in_postgres(data_list):
    """Stores ChromaDB documents in PostgreSQL after a successful update."""
    connection = None
    cursor = None

    try:
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        current_app.logger.info("PostgreSQL data received: %s", data_list)

        for record in data_list:
            file_name = record['file_name']

            cursor.execute("SELECT COUNT(*) FROM human_altered_table WHERE file_name = %s;", (file_name,))
            result = cursor.fetchone()[0]
            print("filenames:", result)

            if result > 0:
          
                cursor.execute("DELETE FROM human_altered_table WHERE file_name = %s;", (file_name,))
                connection.commit()
                current_app.logger.info(f"[PostgreSQL] Old record for '{file_name}' deleted.")

        # Updated insert/upsert for human_altered_table with all columns
        insert_query = """
        INSERT INTO public.human_altered_table (
            file_id,
            file_name,
            data_category,
            sensitivity,
            data_classifiers,
            responsible_values,
            retention_time,
            word_count,
            reason_for_change,
            creation_time,
            last_modification_time,
            created_by,
            modified_by
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (file_id) DO UPDATE SET
            file_name             = EXCLUDED.file_name,
            data_category         = EXCLUDED.data_category,
            sensitivity           = EXCLUDED.sensitivity,
            data_classifiers      = EXCLUDED.data_classifiers,
            responsible_values    = EXCLUDED.responsible_values,
            retention_time        = EXCLUDED.retention_time,
            word_count            = EXCLUDED.word_count,
            reason_for_change     = EXCLUDED.reason_for_change,
            last_modification_time= EXCLUDED.last_modification_time,
            modified_by           = EXCLUDED.modified_by;
        """

        values = []
        for record in data_list:
            # Clean and coerce retention_time to None if empty or invalid
            raw_rt = record.get('retention_time')
            if raw_rt in (None, ''):
                retention_time_val = None
            else:
                try:
                    retention_time_val = float(raw_rt)
                except (ValueError, TypeError):
                    retention_time_val = None

            # word_count is integer (or None)
            word_count_val = record.get('word_count')
            if word_count_val in (None, ''):
                word_count_val = None

            values.append((
                record['id'],                                  # file_id
                clean_text(record['file_name']),
                clean_text(record.get('data_category', '')),
                record.get('sensitivity'),
                clean_text(record.get('data_classifiers', '')),
                clean_text(record.get('responsible_values', '')),
                retention_time_val,                            # now a float or None
                word_count_val,                                # integer or None
                clean_text(record.get('reason_for_change', '')),
                record.get('created_at'),
                record.get('last_modification_time'),
                clean_text(record.get('created_by', '')),
                clean_text(record.get('modified_by', ''))
            ))
       
        cursor.executemany(insert_query, values)

        connection.commit()
        current_app.logger.info("[PostgreSQL] Data inserted successfully.")
        compare_and_update_postgres()
    except Exception as e:
        print(f"[PostgreSQL] Error: {e}")
        current_app.logger.error(f"[PostgreSQL] Error: {e}")
        if connection:
            connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()