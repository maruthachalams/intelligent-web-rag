import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our custom modules
from backend.scraper import WebScraper
from backend.rag_engine import RAGEngine

# Import LLM libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environmental configs
load_dotenv()

# Verify GROQ API Key exists
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY is missing from your .env file!")

# Initialize backend instances
app = FastAPI(title="Intelligent Web-RAG API Gateway")
scraper = WebScraper()
rag = RAGEngine()

# Allow frontend requests (CORS config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatGroq LLM model
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # Modern fast developer standard (Nov 2026)
    temperature=0.0              # Temperature 0 ensures precise, grounded, factual outputs
)

# Pydantic schemas for request validation
class IngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

# Global variable to track the currently loaded web source
current_source = {"url": "No active URL"}

@app.get("/health")
def health_check():
    """Service Health check endpoint"""
    return {"status": "operational", "loaded_url": current_source["url"]}

@app.post("/ingest")
def ingest_url(request: IngestRequest):
    try:
        rag.clear_db()
        ingested_pages = scraper.crawl_and_ingest(request.url, limit=6)
        
        if not ingested_pages:
             raise HTTPException(status_code=400, detail="The scraper failed to retrieve any content.")
             
        total_chars = 0
        pages_list = []
        for page in ingested_pages:
             rag.add_document(page["text"], page["url"])
             total_chars += len(page["text"])
             pages_list.append(page["url"])
        
        current_source["url"] = request.url
        
        return {
            "status": "success",
            "message": f"Vectorized {len(ingested_pages)} pages!",
            "character_count": total_chars,
            "pages": pages_list  # Explicitly return URLs list!
        }
    except ValueError as ve:
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
@app.post("/query")
def ask_question(request: QueryRequest):
    """
    1. Fetches semantic chunks related to query
    2. Builds custom system prompt containing the search data
    3. Prompts LLM and returns the grounded generation
    """
    try:
        # Perform retrieval
        matched_chunks = rag.retrieve(request.question, k=4)
        if not matched_chunks:
             raise HTTPException(status_code=404, detail="No relevant context found. Please ingest a URL first.")
        
        # Build injected context string
        context_string = "\n\n".join([chunk.page_content for chunk in matched_chunks])
        source_url = matched_chunks[0].metadata.get("source", current_source["url"])
        
        # Strict context grounding prompt (LLM acts purely as search synthesis)
        system_prompt = (
            "You are a highly precise, factual QA assistant analyzing the contents of a webpage.\n"
            "Answer the user's question USING ONLY the provided webpage context. If the correct "
            "answer cannot be confidently found in the provided fragments, reply with: "
            "'I'm sorry, but that exact information is not available on the provided webpage.'\n"
            "Do not fabricate or extrapolate beyond these facts.\n\n"
            "---------- WEB DATA CONTEXT ----------\n"
            "{context}\n"
            "---------------------------------------"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # Construct pipeline & run
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context_string,
            "question": request.question
        })
        
        return {
            "answer": response.content,
            "source": source_url
        }
        
    except FileNotFoundError as fe:
        raise HTTPException(status_code=400, detail="Database empty. You must ingest a URL first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)