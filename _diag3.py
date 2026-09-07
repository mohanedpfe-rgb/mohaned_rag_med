import sqlite3, json

conn = sqlite3.connect("data/vector_db/lexical.sqlite3")
cur = conn.cursor()
cur.execute("PRAGMA table_info(lexical_chunks)")
for r in cur.fetchall():
    print(r)
