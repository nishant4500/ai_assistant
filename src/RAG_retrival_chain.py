from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_qa_chain(vectordb, llm):
    
    # Retrieve top 10 chunks initially
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    
    # Set up Semantic Reranker (ms-marco-MiniLM-L-6-v2) to get top 5
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # Custom document prompt to include metadata
    document_prompt = PromptTemplate(
        input_variables=["page_content", "paper_name", "section"],
        template="Source: {paper_name} (Section: {section})\nContent: {page_content}"
    )
    
    # Custom prompt matching PRD
    prompt_template = """
You are an AI Research Assistant.

Answer ONLY using the provided research paper context.

Do not use external knowledge.

If information is unavailable in the provided context, respond with:
"The requested information was not found in the uploaded papers."

Provide concise, accurate, and academic responses.

Always prioritize factual correctness.

Mention source paper names whenever possible.

Context: 
{context}

Question: 
{question}

Instructions:  
- Use ONLY the provided context.
- Do not hallucinate.
- Do not invent citations.
- Mention relevant paper names.
"""   

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_prompt": document_prompt
        }
    )
    
    return chain