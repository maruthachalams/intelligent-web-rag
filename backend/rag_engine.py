import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()

class RAGEngine:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        
        # 1. Load Local Semantic Embedding Model
        # (This is lightweight and runs fast on your local environment)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Configure Splitter for break-downs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150
        )
        
        # 3. Setup Persistent ChromaDB Vector Store
        self.db = Chroma(
            persist_directory=self.db_path, 
            embedding_function=self.embeddings,
            collection_name="web_rag_collection"
        )
        
        # 4. Initialize Groq API Client for Answers
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def clear_db(self):
        """Clean slate: clear data inside the collection without OS-level file system deletion (Windows-safe!)"""
        try:
            # Delete and wipe the DB securely via Chroma's internal API
            existing_data = self.db.get()
            if existing_data and 'ids' in existing_data and existing_data['ids']:
                self.db.delete(ids=existing_data['ids'])
                print(f"🧹 Cleaned {len(existing_data['ids'])} existing data chunks from ChromaDB (Windows-safe reset).")
            else:
                print("🧹 ChromaDB is already empty. No deletion required.")
        except Exception as e:
            print(f"⚠️ Safe database reset failed, trying physical drop: {e}")
            # Fallback only if the DB is uninitialized or corrupt
            if os.path.exists(self.db_path):
                try:
                    shutil.rmtree(self.db_path)
                    print(f"🧹 Force-removed DB directory to fix error.")
                    # Re-initialize DB after directory deletion
                    self.db = Chroma(
                        persist_directory=self.db_path, 
                        embedding_function=self.embeddings,
                        collection_name="web_rag_collection"
                    )
                except Exception as ex:
                    print(f"❌ Failed to drop folder: {ex}")

    def add_document(self, text: str, source_url: str):
        """Split and add parsed text to the vector database with metadata."""
        if not text or len(text.strip()) < 50:
            print(f"⚠️ Content too short, skipping vector ingestion: {source_url}")
            return
            
        chunks = self.text_splitter.split_text(text)
        metadatas = [{"source": source_url} for _ in chunks]
        
        # Ingest to ChromaDB
        self.db.add_texts(texts=chunks, metadatas=metadatas)
        print(f"✅ Chunked document into {len(chunks)} chunks and loaded into ChromaDB at '{self.db_path}'")

    def query(self, question: str) -> tuple[str, str]:
        """
        Public Search & Synthesis Route.
        Returns a tuple: (Synthesized Answer String, Citations/Sources String)
        """
        print(f"🔍 Searching ChromaDB & Querying LLM for: '{question}'")
        
        try:
            # 1. Search vector database for top 4 matches
            # Specify search type to avoid any API discrepancies
            docs_and_scores = self.db.similarity_search_with_score(question, k=4)
            
            if not docs_and_scores:
                return "I could not find any relevant information in the ingested pages.", "No Source"
                
            documents = [doc.page_content for doc, _ in docs_and_scores]
            metadatas = [doc.metadata for doc, _ in docs_and_scores]
            
            # Compile context
            context = "\n---\n".join(documents)
            
            # Collect unique sources
            sources_set = {m.get("source", "unknown") for m in metadatas if m}
            sources = ", ".join(sources_set) if sources_set else "unknown"
            
        except Exception as e:
            print(f"❌ Search Error: {e}")
            return f"Search Query process failed: {e}", "unknown"
            
        # 2. Ask the Llama 3 LLM via Groq
        try:
            prompt = f"""
            You are an expert AI Search Assistant. Answer the user's question based STRICTLY on the provided context. 
            Keep your answer factual, direct, and under 4-5 sentences.
            If the context does not contain the answer, politely say: "Based on the provided ingested pages, I cannot find the answer to that."
            
            Provided Context:
            {context}
            
            User Question:
            {question}
            
            Your grounded answer:
            """
            
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-20b", # Extremely fast, modern model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            answer = completion.choices[0].message.content.strip()
            return answer, sources
            
        except Exception as e:
            print(f"❌ LLM completion failed: {e}")
            return f"Answer synthesis error: {e}", sources