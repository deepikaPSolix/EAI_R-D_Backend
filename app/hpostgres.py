import psycopg2
import json
from datetime import datetime
from flask import current_app
import pandas as pd
from app.hitl_data_processing import compare_and_update_postgres
DB_CONFIG = {
    "host": "192.168.1.116",
    "database": "postgres",
    "user": "postgres",
    "password": "12345",
    "port": "25432"
}
def create_documents_table():
    """Creates the documents table if it does not exist."""
    connection = None
    cursor = None

    try:
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()
       
        create_table_query = """
        CREATE TABLE IF NOT EXISTS human_altered_table (
            id UUID PRIMARY KEY,
            file_name TEXT NOT NULL,
            attributes JSONB NOT NULL,
            data_classifiers TEXT,
            label TEXT NOT NULL, 
            responsible_values TEXT, 
            retention_time TEXT,
            sensitivity INT,
            reason_for_change TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print("[PostgreSQL] Table 'human_altered_table' checked/created successfully.")

    except Exception as e:
        print(f"[PostgreSQL] Error creating table: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


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

        insert_query = """
        INSERT INTO human_altered_table (id, file_name, attributes, data_classifiers, label, responsible_values, retention_time, sensitivity, reason_for_change)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = [
            (
                record['id'],
                clean_text(record['file_name']),
                json.dumps(record['attributes']),
                clean_text(record['data_classifiers']),
                clean_text(record['label']),
                clean_text(record['responsible_values']),
                clean_text(record['retention_time']),
                record['sensitivity'],
                clean_text(record['reason_for_change'])
            )
            for record in data_list
        ]

       
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