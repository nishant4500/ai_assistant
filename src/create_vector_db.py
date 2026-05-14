import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.metadata_store import add_chunk

def create_or_update_vector_db(sections_dict, paper_id, paper_name, embedder, db_path="research_paper_vector_db"):
    """
    Chunks text by section, attaches metadata, and appends to or creates a FAISS index.
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    
    docs = []
    
    for section_name, content in sections_dict.items():
        chunks = splitter.split_text(content)
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "paper_name": paper_name,
                    "section": section_name
                }
            )
            docs.append(doc)
            
            # Store metadata in SQLite as well
            add_chunk(chunk_id, paper_id, section_name, None, None)
            
    if not docs:
        return None
        
    # Load existing or create new FAISS index
    if os.path.exists(db_path):
        vectordb = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
        vectordb.add_documents(docs)
    else:
        vectordb = FAISS.from_documents(docs, embedding=embedder)
        
    vectordb.save_local(db_path)
    return vectordb
