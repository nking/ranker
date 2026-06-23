
import subprocess
import logging
import psycopg2


def reset_mlflow_records(experiment_name: str, mlflow_experiment_tracking_uri:str):
    """
    Deletes an MLflow experiment and all associated runs, tags, metrics, and params.
    """
    try:
        conn = psycopg2.connect(mlflow_experiment_tracking_uri)
        logging.info(f"conn = {conn}")
        with conn:
            with conn.cursor() as cursor:
                # 1. Get the experiment_id
                cursor.execute(
                    "SELECT experiment_id FROM experiments WHERE name = %s",
                    (experiment_name,))
                result = cursor.fetchone()
                
                if not result:
                    print(f"Experiment '{experiment_name}' not found.")
                    return
                
                experiment_id = result[0]
                
                # 2. Get all run_uuids for this experiment
                cursor.execute(
                    "SELECT run_uuid FROM runs WHERE experiment_id = %s",
                    (experiment_id,))
                run_uuids = [row[0] for row in cursor.fetchall()]
                
                if run_uuids:
                    run_uuids_tuple = tuple(run_uuids)
                    
                    # 3. CRITICAL: Delete from every table that references run_uuid
                    # The order here is crucial to satisfy Foreign Key constraints.
                    cursor.execute(
                        "DELETE FROM latest_metrics WHERE run_uuid IN %s",
                        (run_uuids_tuple,))
                    cursor.execute("DELETE FROM metrics WHERE run_uuid IN %s",
                        (run_uuids_tuple,))
                    cursor.execute("DELETE FROM params WHERE run_uuid IN %s",
                        (run_uuids_tuple,))
                    cursor.execute("DELETE FROM tags WHERE run_uuid IN %s",
                        (run_uuids_tuple,))
                    
                    # 4. Now we can safely delete the runs
                    cursor.execute("DELETE FROM runs WHERE experiment_id = %s",
                        (experiment_id,))
                
                # 5. Delete experiment-level tags and the experiment itself
                cursor.execute(
                    "DELETE FROM experiment_tags WHERE experiment_id = %s",
                    (experiment_id,))
                cursor.execute(
                    "DELETE FROM experiments WHERE experiment_id = %s",
                    (experiment_id,))
                    
                logging.info(
                    f"Successfully deleted experiment '{experiment_name}' (ID: {experiment_id}) and all associated data.")
    
    except Exception as e:
        logging.error(f"Database error during deletion: {e}")
        raise
    finally:
        conn.close()

def reset_vizier_records(project_id: str, study_name: str):
    """
    Deletes Vizier records for a specific project_id and studstudy_namey_id.
    Executes within a single transaction to ensure consistency.
    """

    # We use the context manager (with conn:) to automatically handle
    # the transaction (commit on success, rollback on failure).
    python_script = f"""
import sqlite3
import sys

try:
    conn = sqlite3.connect('/app/data/vizier.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT study_id FROM studies WHERE study_name = ? AND owner_id = ?', ('{study_name}', '{project_id}'))
    result = cursor.fetchone()
    
    if not result:
        resource_name = 'owners/{project_id}/studies/{study_name}'
        cursor.execute('SELECT study_id FROM studies WHERE study_name = ? AND owner_id = ?', (resource_name, '{project_id}'))
        result = cursor.fetchone()
        if not result:
            #print("No study found, nothing to delete.")
            sys.exit(0)
    
    study_id = result[0]
    
    cursor.execute('DELETE FROM suggestion_operations WHERE study_id = ?', (study_id,))
    cursor.execute('DELETE FROM early_stopping_operations WHERE study_id = ?', (study_id,))
    cursor.execute('DELETE FROM trials WHERE study_id = ?', (study_id,))
    cursor.execute('DELETE FROM studies WHERE study_id = ?', (study_id,))
    #cursor.execute('DELETE FROM owners WHERE name = ?', ('{project_id}',))
    
    conn.commit()
    print(f"Successfully deleted study_id {{study_id}}")
    conn.close()
except Exception as e:
    sys.stderr.write(f"DB Error: {{str(e)}}")
    sys.exit(1)
    """
    
    cmd = ["docker", "exec", "vizier-server", "python3", "-c", python_script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Cleanup failed: {result.stderr}")
    logging.info(f"vizier cleanup finished: {result.stdout}")
    return result.stdout
    

def reset_checkpoint_buckets(study_name:str):
    for subdir in ("latest", "best"):
        command = [
            "docker", "exec", "gcs_emulator",
            "sh", "-c", f"rm -rf /storage/checkpoint-bucket/{subdir}/{study_name}"
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            print("empty checkpoint-bucket/* successful")
        except subprocess.CalledProcessError as e:
            print(f"Error resetting database: {e.stderr}")

def reset_hpo_results_bucket(project_id:str, study_name:str):
    command = [
        "docker", "exec", "gcs_emulator",
        "sh", "-c", f"rm -rf /storage/hpo-results-bucket/{project_id}/{study_name}"
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print("empty checkpoint-bucket/* successful")
    except subprocess.CalledProcessError as e:
        print(f"Error resetting database: {e.stderr}")
