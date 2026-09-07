import sqlite3, json, sys

sys.path.insert(0, ".")
conn = sqlite3.connect("data/vector_db/lexical.sqlite3")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Lexical tables:", cur.fetchall())
for table in ["chunks", "documents", "chunk_fts"]:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]} rows")
    except Exception as e:
        print(f"  {table}: no such table ({e})")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
all_tables = [r[0] for r in cur.fetchall()]
print("  all tables:", all_tables)
for t in all_tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"    {t}: {cur.fetchone()[0]}")
    except Exception as e:
        pass
try:
    for t in all_tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r for r in cur.fetchall()]
        if any("chunk" in (c[1] or "").lower() for c in cols):
            cur.execute(f"SELECT * FROM {t} LIMIT 1")
            r = cur.fetchone()
            if r:
                print(f"  sample from {t}:", [str(x)[:200] for x in r])
            break
except Exception as e:
    print("   err:", e)
conn.close()
print()
conn = sqlite3.connect("data/ingestion.sqlite3")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
itables = [r[0] for r in cur.fetchall()]
print("Ingestion tables:", itables)
for t in itables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t}: {cur.fetchone()[0]}")
    except: pass
for t in itables:
    if "doc" in t.lower():
        try:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [r[1] for r in cur.fetchall()]
            print(f"  {t} cols:", cols)
            sel = ", ".join(cols[:8])
            cur.execute(f"SELECT {sel} FROM {t} LIMIT 3")
            for r in cur.fetchall():
                print("   row:", r)
        except Exception as e:
            print("  doc err:", e)
conn.close()
