from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.load_and_extract_text import (
    extract_text_from_pdf,
    extract_pdf_sections
)

from src.detect_and_split_sections import (
    refine_sections,
    split_sections_with_content
)

from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_or_update_vector_db
from src.RAG_retrival_chain import get_qa_chain

from src.metadata_store import init_db, add_paper, get_all_papers

from dotenv import load_dotenv

import shutil
import os
import uuid
import json
import re
from collections import Counter

# ==============================
# Load Environment Variables
# ==============================

load_dotenv()

# ==============================
# Cleanup Session Data on Startup
# ==============================

def cleanup_previous_session():
    folders_to_delete = ["uploads", "research_paper_vector_db", "metadata_db"]
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            
    if os.path.exists("topics_cache.json"):
        os.remove("topics_cache.json")

# Clean up before initializing DBs
cleanup_previous_session()

# ==============================
# Initialize FastAPI App & DB
# ==============================

app = FastAPI()

init_db()

# ==============================
# Static & Templates Setup
# ==============================

templates = Jinja2Templates(directory="templates")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# ==============================
# Upload Folder Setup
# ==============================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Environment Variables
# ==============================

groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ==============================
# Global Variables & State Persistence
# ==============================

TOPICS_CACHE_FILE = "topics_cache.json"

Research_paper_topics = {}

if os.path.exists(TOPICS_CACHE_FILE):
    try:
        with open(TOPICS_CACHE_FILE, "r", encoding="utf-8") as f:
            Research_paper_topics = json.load(f)
    except Exception as e:
        print("Could not load topics cache:", e)

def save_topics_cache():
    with open(TOPICS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(Research_paper_topics, f)

# ==============================
# Initialize LLM
# ==============================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=llm_model
)

# ==============================
# Initialize Embeddings
# ==============================

embedder = HuggingFaceEmbeddings(
    model_name=embedding_model
)

# Initialize vector DB if exists
vector_db = None
if os.path.exists("research_paper_vector_db"):
    vector_db = FAISS.load_local("research_paper_vector_db", embedder, allow_dangerous_deserialization=True)


# ==============================
# Request Models
# ==============================

class SummaryRequest(BaseModel):
    topic: str
    paper_id: str = None
    summary_type: str = "full"

class ChatRequest(BaseModel):
    message: str


# ==============================
# Home Route
# ==============================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


from typing import List

# ==============================
# Upload PDF Route
# ==============================

@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):

    global Research_paper_topics
    global vector_db

    try:
        uploaded_papers = []
        all_topics = set()
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
                
            paper_id = f"P_{uuid.uuid4().hex[:8]}"
            paper_name = file.filename

            # Save uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, paper_name)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Extract text
            extracted_text = extract_text_from_pdf(file_path)

            # Extract sections
            extracted_sections = extract_pdf_sections(full_text=extracted_text)

            # Refine sections
            refined_sections = refine_sections(extracted_sections, llm)

            # Split sections
            section_with_content = split_sections_with_content(extracted_text, refined_sections)

            # Update Vector DB and SQLite
            add_paper(paper_id, paper_name)
            vector_db = create_or_update_vector_db(section_with_content, paper_id, paper_name, embedder)

            # Store in-memory and save
            Research_paper_topics[paper_id] = section_with_content
            save_topics_cache()
            
            uploaded_papers.append({"paper_id": paper_id, "paper_name": paper_name})
            all_topics.update(section_with_content.keys())

        papers = get_all_papers()

        if not uploaded_papers:
            return JSONResponse(status_code=400, content={"error": "No valid PDF files were processed."})

        # Return the last processed paper's ID to auto-select it in UI
        last_paper = uploaded_papers[-1]
        
        return JSONResponse(
            content={
                "message": f"Successfully uploaded {len(uploaded_papers)} paper(s)",
                "paper_id": last_paper["paper_id"],
                "paper_name": last_paper["paper_name"],
                "topics": list(all_topics),
                "all_papers": papers,
                "uploaded_papers": uploaded_papers
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


class CompareRequest(BaseModel):
    paper_ids: list[str]

class LitReviewRequest(BaseModel):
    topic: str

# ==============================
# Insights Route
# ==============================

@app.get("/insights/{paper_id}")
async def get_insights(paper_id: str):
    global Research_paper_topics
    if paper_id not in Research_paper_topics:
        return JSONResponse(status_code=404, content={"error": "Paper not found"})
    
    sections = Research_paper_topics[paper_id]
    
    # 1. Topic Distribution (Section lengths)
    topic_labels = list(sections.keys())
    topic_data = [len(content.split()) for content in sections.values()]
    
    # 2. Keyword Frequency
    full_text = " ".join(sections.values()).lower()
    words = re.findall(r'\b[a-z]{4,}\b', full_text)
    stop_words = {"this", "that", "with", "from", "these", "those", "have", "were", "which", "their", "there", "what", "when", "where", "also", "such", "been", "using", "used", "can", "only", "more", "other", "some", "than", "over", "into", "between"}
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words).most_common(10)
    keyword_labels = [w[0] for w in word_counts]
    keyword_data = [w[1] for w in word_counts]
    
    # 3. Citation Network
    citation_labels = topic_labels
    citation_data = [len(re.findall(r'\[\d+\]', content)) for content in sections.values()]
    
    return JSONResponse(content={
        "topic_distribution": {"labels": topic_labels, "data": topic_data},
        "keyword_frequency": {"labels": keyword_labels, "data": keyword_data},
        "citation_network": {"labels": citation_labels, "data": citation_data}
    })

# ==============================
# Summary Route
# ==============================

@app.post("/summary")
async def get_summary(data: SummaryRequest):
    global Research_paper_topics
    try:
        topic = data.topic
        paper_id = data.paper_id
        summary_type = data.summary_type
        
        # If paper_id is not provided, use the last one (fallback)
        if not paper_id and Research_paper_topics:
            paper_id = list(Research_paper_topics.keys())[-1]
            
        topic_content = "No summary available."
        
        if paper_id in Research_paper_topics:
            sections_dict = Research_paper_topics[paper_id]
            if summary_type == "section":
                topic_content = sections_dict.get(topic, "No summary available.")
            else:
                # For full or bullet summary, aggregate the whole paper text
                topic_content = "\n\n".join(sections_dict.values())

        prompt = f"Please generate a summary of the following content: {topic_content}"
        if summary_type == "full":
            prompt = f"Provide a comprehensive full paper summary of the following content focusing on objective, methodology, dataset, results, conclusion, and limitations: {topic_content[:15000]}"
        elif summary_type == "section":
            prompt = f"Provide a detailed section-wise summary of the following content: {topic_content[:15000]}"
        elif summary_type == "bullet":
            prompt = f"Provide a concise bullet point summary of the entire paper (Main contribution, Architecture used, Dataset used, Results achieved, Limitations): {topic_content[:15000]}"

        response = llm.invoke(prompt)
        summary = response.content

        return JSONResponse(
            content={"summary": summary}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==============================
# Compare Route
# ==============================

@app.post("/compare")
async def compare_papers(data: CompareRequest):
    global Research_paper_topics, vector_db
    try:
        paper_ids = data.paper_ids
        if len(paper_ids) < 2:
            return JSONResponse(status_code=400, content={"error": "Please select at least two papers to compare."})
            
        if vector_db is None:
            return JSONResponse(status_code=400, content={"error": "Vector database not initialized."})
            
        combined_content = ""
        # Retrieve the most relevant chunks for comparison using a generic comparison query
        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        query = "methodology datasets architecture performance advantages limitations"
        
        for pid in paper_ids:
            # We filter by paper_id to get only chunks from this paper
            # FAISS allows filtering if metadata is configured, but if not we can just retrieve generically
            # and format the results. Since we just need summary chunks:
            paper_name = f"Paper ID {pid}"
            if pid in Research_paper_topics:
                # Try to use abstract if it's small enough, max 1000 chars
                intro = Research_paper_topics[pid].get("Abstract", "")
                if intro:
                    combined_content += f"{paper_name}:\n{intro[:1000]}\n\n"
                else:
                    combined_content += f"{paper_name}:\n{list(Research_paper_topics[pid].values())[0][:1000]}\n\n"
                    
        # To strictly avoid the 6000 TPM limit of Groq's free tier, we must truncate combined_content
        combined_content = combined_content[:15000] # Safe limit around ~3500 tokens
                
        prompt = f"""Compare the following research papers.
Focus on: methodology, datasets, architecture, performance, advantages, limitations. Keep it concise.

Papers:
{combined_content}
"""
        response = llm.invoke(prompt)
        return JSONResponse(content={"comparison": response.content})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==============================
# Literature Review Route
# ==============================

@app.post("/literature-review")
async def generate_lit_review(data: LitReviewRequest):
    global vector_db
    try:
        topic = data.topic
        
        if vector_db is None:
            return JSONResponse(status_code=400, content={"error": "Vector database not initialized."})
            
        # Retrieve top chunks related to the topic
        retriever = vector_db.as_retriever(search_kwargs={"k": 8})
        docs = retriever.invoke(topic)
        
        combined_content = ""
        for i, doc in enumerate(docs):
            p_name = doc.metadata.get("paper_name", f"Source {i}")
            combined_content += f"Paper: {p_name}\nContent: {doc.page_content[:600]}\n\n"
            
        # Truncate strictly to avoid TPM limit
        combined_content = combined_content[:15000]
            
        prompt = f"""Generate a short academic literature review for the topic: "{topic}" using the provided retrieved paper content.
Include: research trends, methodologies, limitations, future directions, comparison between approaches.
Use ONLY the provided content.

CRITICAL INSTRUCTION: If the provided "Papers Content" does not contain any relevant information about the topic "{topic}", DO NOT generate a literature review. Instead, reply EXACTLY with: "The requested topic was not found in the uploaded papers."

Papers Content:
{combined_content}
"""
        response = llm.invoke(prompt)
        return JSONResponse(content={"literature_review": response.content})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==============================
# Chat Route
# ==============================

@app.post("/chat")
async def chat(data: ChatRequest):
    global vector_db
    try:
        if vector_db is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a PDF first."}
            )

        user_message = data.message

        # Create QA Chain
        chain = get_qa_chain(vectordb=vector_db, llm=llm)

        # Generate Response
        response = chain.invoke(user_message)
        ai_response = response["result"]

        return JSONResponse(
            content={"response": ai_response}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)