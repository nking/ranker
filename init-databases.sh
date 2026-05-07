#!/bin/bash
set -e

# Connect to the default 'postgres' db to create the others
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec
    GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO "$POSTGRES_USER";
EOSQL
