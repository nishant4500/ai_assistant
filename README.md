# ResearchPortal — AI Research Paper Assistant

ResearchPortal is a sophisticated AI-powered research assistant designed to streamline the process of reading and analyzing academic papers. It leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide deep insights, summaries, and interactive Q&A for multiple research documents simultaneously.

![Hero Image](static/hero.avif)

## 🚀 Features

- **Smart PDF Parsing**: Automatically detects and splits research papers into semantic sections (Abstract, Introduction, Methodology, etc.).
- **Interactive Q&A**: Chat with your papers! Ask nuanced questions and get answers with specific citations.
- **AI Summarization**: 
  - **Full Paper Summary**: Comprehensive overview of objectives, results, and limitations.
  - **Section-wise Summary**: Detailed breakdown of specific parts of the paper.
  - **Bullet Point Summary**: Quick insights into main contributions and architecture.
- **Advanced Research Tools**:
  - **Paper Comparison**: Compare methodology and results across multiple uploaded papers.
  - **Literature Review**: Automatically generate an academic review for any topic based on your paper library.
- **Dynamic Data Insights**: Visualize research data through interactive charts:
  - **Keyword Frequency**: Top themes extracted from the text.
  - **Topic Distribution**: Breakdown of section sizes and focus areas.
  - **Citation Analysis**: Insights into reference density across sections.

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM**: [Groq](https://groq.com/) (Llama 3.1 8B/70B)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: HuggingFace Sentence Transformers
- **Frontend**: Vanilla HTML5, CSS3 (Modern Glassmorphism Design), JavaScript
- **Visualizations**: [Chart.js](https://www.chartjs.org/)

## 📋 Prerequisites

- Python 3.9 or higher
- A [Groq API Key](https://console.groq.com/keys)

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nishant4500/ai_assistant.git
   cd ai_assistant
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama-3.1-8b-instant
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

## 🏃 Running the Application

### Using Python
Start the server using:
```bash
python app.py
```

### Using Docker
1. **Build the Image**
   ```bash
   docker build -t research-portal .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8000:8000 --env-file .env research-portal
   ```

The application will be available at `http://127.0.0.1:8000`.

## 📁 Project Structure

- `app.py`: Main FastAPI application and API routes.
- `src/`: Core logic for RAG, PDF extraction, and metadata management.
- `static/`: Frontend assets (images, styles).
- `templates/`: Jinja2 HTML templates.
- `research_paper_vector_db/`: Local FAISS index storage.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the extraction accuracy or add new visualization features.

---
Built with ❤️ by **Nishant & Jayanth**
