import sqlite3
import os

DB_PATH = "metadata_db/metadata.db"

def init_db():
    os.makedirs("metadata_db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for papers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            paper_name TEXT,
            authors TEXT,
            year INTEGER
        )
    """)
    
    # Table for chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT,
            section TEXT,
            page INTEGER,
            topic TEXT,
            FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
        )
    """)
    
    conn.commit()
    conn.close()

def add_paper(paper_id, paper_name, authors="Unknown", year=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO papers (paper_id, paper_name, authors, year) VALUES (?, ?, ?, ?)",
        (paper_id, paper_name, authors, year)
    )
    conn.commit()
    conn.close()

def get_all_papers():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT paper_id, paper_name FROM papers")
    rows = cursor.fetchall()
    conn.close()
    return [{"paper_id": r[0], "paper_name": r[1]} for r in rows]

def add_chunk(chunk_id, paper_id, section, page=None, topic=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO chunks (chunk_id, paper_id, section, page, topic) VALUES (?, ?, ?, ?, ?)",
        (chunk_id, paper_id, section, page, topic)
    )
    conn.commit()
    conn.close()
