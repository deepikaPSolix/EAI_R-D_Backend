import psycopg2
import pandas as pd
import json
from datetime import datetime
from flask import current_app

DB_CONFIG = {
    "host": "192.168.1.116",
    "database": "postgres",
    "user": "postgres",
    "password": "12345",
    "port": "25432"
}


def compare_and_update_postgres():
    """ Compares and updates machine-generated data with human-reviewed documents in PostgreSQL. """

    try:
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

      
        machine_df = pd.read_sql_query("SELECT * FROM file_metadata_classifcation", connection)
        human_df = pd.read_sql_query("SELECT * FROM human_altered_table", connection)  
        
        
        if 'reason_for_change' not in machine_df.columns:
            machine_df['reason_for_change'] = "N/A"
            machine_df['created_at']="N/A"
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
        print("PostgreSQL data updated_df_data: ")

        update_query = """
       INSERT INTO hitl_updated_table (
            id, file_name, attributes, data_classifiers, label, responsible_values, 
            retention_time, sensitivity, reason_for_change, created_at
)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) 
        DO UPDATE SET 
            file_name = EXCLUDED.file_name,
            attributes = EXCLUDED.attributes,
            data_classifiers = EXCLUDED.data_classifiers,
            label = EXCLUDED.label,
            responsible_values = EXCLUDED.responsible_values,
            retention_time = EXCLUDED.retention_time,
            sensitivity = EXCLUDED.sensitivity,
            reason_for_change = EXCLUDED.reason_for_change,
            created_at = EXCLUDED.created_at;
"""

        values = [
    [
        record['id'],
        record['file_name'],
        json.dumps(record['attributes']),
        record['data_classifiers'],  
        record['label'],
        record['responsible_values'],
        record['retention_time'],
        record['sensitivity'],
        record['reason_for_change'],
        record['created_at']
    ]
    for _, record in updated_df.iterrows()
]


        cursor.executemany(update_query, values)
        connection.commit()

        print("[PostgreSQL] Data updated successfully.")

    except Exception as e:
        print(f"[PostgreSQL] Error: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
# compare_and_update_postgres()