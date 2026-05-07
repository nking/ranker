#!/bin/bash
set -e

mkdir -p bin/sqllite_data

touch /data/vizier.db

chmod 666 /data/vizier.db

echo "SQLite database initialized at bin/sqllite_data/vizier.db"
