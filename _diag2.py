import sqlite3, json

conn = sqlite3.connect("data/vector_db/lexical.sqlite3")
cur = conn.cursor()
cur.execute("SELECT DISTINCT json_extract(metadata, '$.index_state') FROM lexical_chunks LIMIT 5")
for r in cur.fetchall():
    print("index_state:", r[0])
cur.execute("SELECT COUNT(*) FROM lexical_chunks WHERE json_extract(metadata, '$.index_state') = 'READY'")
print("READY chunks:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM lexical_chunks")
print("total chunks:", cur.fetchone()[0])
# peek the full metadata and text
cur.execute("SELECT chunk_id, document_id, text, metadata FROM lexical_chunks LIMIT 2")
for row in cur.fetchall():
    cid, did, txt, meta = row
    print("\nchunk_id:", cid)
    print("  doc_id:", did)
    print("  text[:200]:", txt[:200])
    print("  metadata:", json.dumps(json.loads(meta) if meta else {}, ensure_ascii=False)[:400])
conn.close()
