import psycopg2

db_uri = "postgresql://runner:password123@172.17.0.1:5432/mlflow_db"

try:
    conn = psycopg2.connect(db_uri)
    cursor = conn.cursor()

    for table in ("tags", "experiment_tags", "runs", "experiments"):

        query = f"SELECT * FROM {table};"

        cursor.execute(query)

        #Extract the column headers from cursor.description
        # cursor.description is a tuple of tuples; the 0th element of each is the column name
        headers = [desc[0] for desc in cursor.description]
    
        # Print the headers separated by a pipe character
        print(f'\nTABLE={table}:')
        print(" | ".join(headers))
        print("-" * 100)  # Visual separator line

        rows = cursor.fetchall()
        for row in rows:
            # Map all values to strings so they can be easily joined and printed
            string_row = [str(val) if val is not None else "NULL" for val in row]
            print(" | ".join(string_row))

except Exception as e:
    print(f"Database error: {e}")
finally:
    if 'conn' in locals() and conn:
        cursor.close()
        conn.close()
