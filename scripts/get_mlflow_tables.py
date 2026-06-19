import psycopg2

db_uri = "postgresql://runner:password123@172.17.0.1:5432/mlflow_db"

try:
    # 1. Connect to your database
    conn = psycopg2.connect(db_uri)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)

    tables = cursor.fetchall()

    print("Tables in database:")
    for table in tables:
        # fetchall() returns a list of tuples, so table[0] extracts the string name
        print(f"- {table[0]}")

except Exception as e:
    print(f"Database error: {e}")
finally:
    if 'conn' in locals() and conn:
        cursor.close()
        conn.close()
