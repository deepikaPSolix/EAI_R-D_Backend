import os, psycopg2

def open_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "vanna_db"),
        user=os.getenv("DB_USER", "vanna_user"),
        password=os.getenv("DB_PASSWORD", "vanna_password"),
    )
