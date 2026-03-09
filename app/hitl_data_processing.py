import psycopg2
import pandas as pd
import json
from datetime import datetime
from flask import current_app

#Postgres
from app.postgres_db import DatabaseManager
import os

dns_host = os.getenv("DNS_HOST") or os.getenv("DB_HOST")
dns_dbname = os.getenv("DNS_DBNAME") or os.getenv("DB_NAME")
dns_user = os.getenv("DNS_USER") or os.getenv("DB_USER")
dns_password = os.getenv("DNS_PASSWORD") or os.getenv("DB_PASSWORD")
dns_port = os.getenv("DNS_PORT") or os.getenv("DB_PORT", "5432")
dns = f"host={dns_host} dbname={dns_dbname} user={dns_user} password={dns_password} port={dns_port}"
def compare_and_update_postgres():
    """ Compares and updates machine-generated data with human-reviewed documents in PostgreSQL. """

    try:
        connection = psycopg2.connect(dns)
        cursor = connection.cursor()
           
        machine_df = pd.read_sql_query("SELECT * FROM public.file_metadata_classification", connection)
        human_df = pd.read_sql_query("SELECT * FROM public.human_altered_table", connection)  
        current_app.logger.info(f"machine_df:{machine_df}, human_df:{human_df}")
        
        
        if 'reason_for_change' not in machine_df.columns:
            machine_df['reason_for_change'] = "N/A"
        updated_rows = []
        for _, machine_row in machine_df.iterrows():
            human_match = human_df[human_df['file_name'] == machine_row['file_name']]
            if not human_match.empty:
                human_row = human_match.iloc[0]
                updated_row = machine_row.copy()
                for col in human_df.columns:
                    if col in machine_df.columns and pd.notna(human_row[col]) and human_row[col] != machine_row[col]:
                        updated_row[col] = human_row[col]
                updated_rows.append(updated_row)
            else:
                updated_rows.append(machine_row)
        updated_df = pd.DataFrame(updated_rows)
        current_app.logger.info(f"PostgreSQL data updated_df_data: ")

        update_query = """
            INSERT INTO public.hitl_updated_table (
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
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s , NOW(), %s, %s
            )
            ON CONFLICT (file_id) DO UPDATE SET
                file_name            = EXCLUDED.file_name,
                data_category        = EXCLUDED.data_category,
                sensitivity          = EXCLUDED.sensitivity,
                data_classifiers     = EXCLUDED.data_classifiers,
                responsible_values   = EXCLUDED.responsible_values,
                retention_time       = EXCLUDED.retention_time,
                word_count           = EXCLUDED.word_count,
                reason_for_change    = EXCLUDED.reason_for_change,
                created_by           = EXCLUDED.created_by,
                modified_by          = EXCLUDED.modified_by;
        """


        values = [
            (
                record['file_id'],                       # file_id
                record['file_name'],
                record['data_category'],
                record['sensitivity'],
                record['data_classifiers'],
                record['responsible_values'],
                record['retention_time'],
                record['word_count'],
                record['reason_for_change'],
                record['creation_time'],  
                record.get('created_by', 'System'),
                record.get('modified_by', 'System')
            )
            for _, record in updated_df.iterrows()
        ]

        current_app.logger.info(
            "Retention times (value, type): %s",
            [(rt, type(rt).__name__) for rt in updated_df['retention_time'].tolist()]
        )
        cursor.executemany(update_query, values)
        connection.commit()

        current_app.logger.info("[PostgreSQL] Data updated successfully.")

    except Exception as e:
        current_app.logger.error(f"[PostgreSQL] Error: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()