from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.load_and_extract_text import (
    extract_text_from_pdf,
    extract_pdf_sections
)

from src.detect_and_split_sections import (
    refine_sections,
    split_sections_with_content
)

from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

from dotenv import load_dotenv

import shutil
import os


# ==============================
# Load Environment Variables
# ==============================

load_dotenv()


# ==============================
# Initialize FastAPI App
# ==============================

app = FastAPI()


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

llm_model = os.getenv(
    "LLM_MODEL",
    "llama-3.1-8b-instant"
)

embedding_model = os.getenv("EMBEDDING_MODEL")


# ==============================
# Global Variables
# ==============================

full_text = ""

Research_paper_topics = None

vector_db = None


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


# ==============================
# Request Models
# ==============================

class SummaryRequest(BaseModel):
    topic: str


class ChatRequest(BaseModel):
    message: str


# ==============================
# Home Route
# ==============================

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ==============================
# Upload PDF Route
# ==============================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global full_text
    global Research_paper_topics
    global vector_db

    try:

        # Reset vector db for new file
        vector_db = None

        # Save uploaded file
        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text
        extracted_text = extract_text_from_pdf(file_path)

        full_text = extracted_text

        # Extract sections
        extracted_sections = extract_pdf_sections(
            full_text=extracted_text
        )

        # Refine sections
        refined_sections = refine_sections(
            extracted_sections,
            llm
        )

        # Split sections
        section_with_content = split_sections_with_content(
            extracted_text,
            refined_sections
        )

        Research_paper_topics = section_with_content

        return JSONResponse(
            content={
                "topics": list(
                    Research_paper_topics.keys()
                )
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# ==============================
# Summary Route
# ==============================

@app.post("/summary")
async def get_summary(data: SummaryRequest):

    global Research_paper_topics

    try:

        if Research_paper_topics is None:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Please upload a PDF first."
                }
            )

        topic = data.topic

        topic_content = Research_paper_topics.get(
            topic,
            "No summary available."
        )

        summary = generate_detailed_summary(
            topic_content,
            llm
        )

        return JSONResponse(
            content={
                "summary": summary
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# ==============================
# Chat Route
# ==============================

@app.post("/chat")
async def chat(data: ChatRequest):

    global full_text
    global vector_db

    try:

        if not full_text:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Please upload a PDF first."
                }
            )

        user_message = data.message

        # Create vector DB only once
        if vector_db is None:

            vector_db = create_vector_db(
                text=full_text,
                embedder=embedder
            )

        # Create QA Chain
        chain = get_qa_chain(
            vectordb=vector_db,
            llm=llm
        )

        # Generate Response
        ai_response = chain.invoke(
            user_message
        )["result"]

        return JSONResponse(
            content={
                "response": ai_response
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )