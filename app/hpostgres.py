import psycopg2
import json
from datetime import datetime
from flask import current_app
import pandas as pd
from app.hitl_data_processing import compare_and_update_postgres
#Postgres
from app.postgres_db import DatabaseManager
import os
dns_host = os.getenv("DNS_HOST")
dns_dbname = os.getenv("DNS_DBNAME")
dns_user = os.getenv("DNS_USER")
dns_password = os.getenv("DNS_PASSWORD")
dns_port = os.getenv("DNS_PORT")
dns = f"host={dns_host} dbname={dns_dbname} user={dns_user} password={dns_password} port={dns_port}"
def parse_retention(rt_str):
    """
    Convert strings like "1 year, 6 months" or "18 months" into a numeric
    value in years (float). Returns None for empty/invalid input.
    """
    if not rt_str or not rt_str.strip():
        return None
    try:
        total = 0.0
        # split on comma: ["1 year", " 6 months"]
        for part in rt_str.split(','):
            num, unit = part.strip().split()[:2]
            n = float(num)
            if unit.startswith('year'):
                total += n
            elif unit.startswith('month'):
                total += n / 12.0
            else:
                total += n  # fallback
        return total
    except Exception:
        try:
            return float(rt_str)
        except Exception:
            return None


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
        connection = psycopg2.connect(dns)
        cursor = connection.cursor()

        current_app.logger.info("PostgreSQL data received: %s", data_list)

        for record in data_list:
            file_name = record['file_name']

            cursor.execute("SELECT COUNT(*) FROM human_altered_table WHERE file_name = %s;", (file_name,))
            result = cursor.fetchone()[0]
            
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
                %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(),NOW(), %s, %s
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
        # map numeric codes → descriptive strings
        SENSITIVITY_LABELS = {
            1: "Public Data",
            2: "Internal Data",
            3: "Confidential Data",
            4: "Restricted Data",
            5: "Private Data",
            6: "Critical Data",
            7: "Regulatory Data",
        }

        for record in data_list:
            # Clean and coerce retention_time to None if empty or invalid
            raw_rt = record.get('retention_time')
            retention_time_val = parse_retention(raw_rt)                 
            word_count_val = record.get('word_count')
            if word_count_val in (None, ''):
                word_count_val = None

            values.append((
                record['id'],                                  # file_id
                clean_text(record['file_name']),
                clean_text(record.get('label', '')),
                SENSITIVITY_LABELS.get(int(record['sensitivity']), record['sensitivity']),
                clean_text(record.get('data_classifiers', '')),
                clean_text(record.get('responsible_values', '')),
                retention_time_val,                            # now a float or None
                word_count_val,                                # integer or None
                clean_text(record.get('reason_for_change', '')),
                clean_text(record.get('created_by', 'System')),
                clean_text(record.get('modified_by', 'System'))
            ))
                   
        cursor.executemany(insert_query, values)

        connection.commit()
        current_app.logger.info("[PostgreSQL] Data inserted successfully.")
        compare_and_update_postgres()
    except Exception as e:
        current_app.logger.error(f"[PostgreSQL] Error: {e}")
        if connection:
            connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
