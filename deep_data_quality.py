import sqlite3
import json

print("=== DEEP DATA QUALITY DIAGNOSTIC ===")

print("\n1. LEXICAL STORAGE")
conn = sqlite3.connect("data/vector_db/lexical.sqlite3")
cur = conn.execute("SELECT COUNT(*) FROM lexical_documents")
print("   Total chunks:", cur.fetchone()[0])
cur = conn.execute("SELECT COUNT(DISTINCT json_extract(metadata, \"$.page_numbers\")) FROM lexical_documents WHERE index_state=\"READY\"")
print("   Unique pages:", cur.fetchone()[0])
cur = conn.execute("SELECT json_extract(metadata, \"$.chunk_index\"), COUNT(*) FROM lexical_documents GROUP BY json_extract(metadata, \"$.chunk_index\") ORDER BY json_extract(metadata, \"$.chunk_index\") LIMIT 5")
print("   Sample pages:", [r for r in cur.fetchall()])
conn.close()

print("\n2. CHROMADB")
import chromadb
client = chromadb.PersistentClient("data/vector_db")
coll = client.get_collection("rag_documents")
print("   Count:", coll.count())

print("\n3. EMBEDDINGS")
if coll.count() > 0:
    result = coll.get(include=["embeddings"], limit=1)
    emb = result["embeddings"][0]
    print("   Dimension:", len(emb))
    print("   Sample:", emb[:5])

print("\n4. METADATA")
conn = sqlite3.connect("data/vector_db/lexical.sqlite3")
cur = conn.execute("SELECT metadata FROM lexical_documents WHERE index_state=\"READY\" LIMIT 1")
meta = json.loads(cur.fetchone()[0])
print("   Keys:", list(meta.keys())[:8])
conn.close()

print("\n=== END ===")

