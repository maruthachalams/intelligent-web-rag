import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environmental configs
load_dotenv()

class RAGEngine:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        # Use a lightweight embedding model that runs locally on your CPU (No API fees!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        # Custom spliter to split document into 1000 char bits with overlapping context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def add_document(self, text: str, source_url: str):
        """Split, embed, and store document in the vector database"""
        chunks = self.text_splitter.split_text(text)
        # Add metadata so the LLM can reference the source website later
        metadatas = [{"source": source_url} for _ in chunks]
        
        # Connection to ChromaDB (It auto-creates the DB if it doesn't exist)
        db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        # Embed and insert the texts
        db.add_texts(texts=chunks, metadatas=metadatas)
        print(f"✅ Chunked document into {len(chunks)} chunks and loaded into ChromaDB at '{self.db_path}'")
        
    def retrieve(self, query: str, k: int = 5):
        """Perform semantic search to fetch the most relevant contextual chunks"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Chroma DB directory does not exist. Please ingest data first.")
            
        db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        # Fetch the top 'k' chunks that are mathematically similar to user query
        results = db.similarity_search(query, k=k)
        return results

    def clear_db(self):
        """Clean slate: clear data inside the collection without OS-level file system deletion (Windows-safe!)"""
        if os.path.exists(self.db_path):
            try:
                # Connect to the existing DB
                db = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                # Fetch all existing index IDs
                existing_data = db.get()
                if existing_data and 'ids' in existing_data and existing_data['ids']:
                    ids_to_delete = existing_data['ids']
                    # Clear out each chunk ID inside ChromaDB internally
                    db.delete(ids=ids_to_delete)
                    print(f"🧹 Cleaned {len(ids_to_delete)} existing data chunks from ChromaDB (Windows-safe reset).")
                else:
                    print("🧹 ChromaDB is already empty. No deletion required.")
            except Exception as e:
                # In case the collection is corrupted or uninitialized, fallback to dropping with a warning
                print(f"⚠️ Safe database reset failed, falling back to drop collection: {e}")
                try:
                    db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
                    db.delete_collection()
                    print("🗑️ ChromaDB collection dropped successfully.")
                except Exception as ex:
                     print(f"❌ OS lock prevent folder deletion fallback. This is normal on Windows. Proceeding... {ex}")