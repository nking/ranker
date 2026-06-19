#!/bin/bash

#if the vzier service is running on docker:
docker exec -it vizier-server python3 -c "
import sqlite3
conn = sqlite3.connect('/app/data/vizier.db') 
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type=\'table\';')
print(cursor.fetchall())"

echo ""
echo "here are the columns"
docker exec -it vizier-server python3 -c "
import sqlite3
conn = sqlite3.connect('/app/data/vizier.db') 
cursor = conn.cursor()
cursor.execute('SELECT m.name AS table_name, p.name AS column_name, p.type AS data_type FROM sqlite_master m JOIN pragma_table_info(m.name) p WHERE m.type = \'table\';')
print(cursor.fetchall())"

echo ""
echo "here is the content of table owners"
docker exec -it vizier-server python3 -c "
import sqlite3
conn = sqlite3.connect('/app/data/vizier.db') 
cursor = conn.cursor()
cursor.execute('SELECT * from owners;')
print(cursor.fetchall())"

echo ""
echo "here is the content of table studies"
docker exec -it vizier-server python3 -c "
import sqlite3
conn = sqlite3.connect('/app/data/vizier.db') 
cursor = conn.cursor()
cursor.execute('SELECT * from studies;')
print(cursor.fetchall())"

echo ""
echo "here is the content of table trials"
docker exec -it vizier-server python3 -c "
import sqlite3
conn = sqlite3.connect('/app/data/vizier.db') 
cursor = conn.cursor()
cursor.execute('SELECT * from trials;')
print(cursor.fetchall())"

