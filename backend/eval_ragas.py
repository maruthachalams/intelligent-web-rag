import chromadb
import os
import pandas as pd
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from backend.rag_engine import RAGEngine
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

class SafeChatGroq(ChatGroq):
    def _combine_llm_outputs(self, llm_outputs):
        return {}

def run_evaluation():
    print("🎯 Starting Automated Ragas Evaluation...")
    
    # Initialize your local RAG engine and Groq Judge
    rag = RAGEngine()
    rag.clear_db()
    sample_text = (
        "Nvidia formed a specialized board committee to oversee national security and export controls. "
        "Jensen Huang resigned as CEO, and Colette Kress prepares as interim CEO for Q3 earnings "
        "amid a personal affairs investigation."
    )
    rag.add_document(sample_text, "https://example.com/nvidia-eval-doc")
    
    # We must explicitly use LangChain's Groq wrapper to drive Ragas evaluation!
    # Because we're on a free/fast dev pipeline, we use llama3-70b
    groq_judge = SafeChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0.0
    )
    
    # Mock some expected Evaluation data! We can test against a known source
    # E.g., we query a general RAG topic like Nvidia updates or general loaded context
    eval_dataset = [
        {
            "question": "Why did Nvidia form a board committee?",
            "ground_truth": "To oversee national security and export controls."
        },
        {
            "question": "What happened to Jensen Huang's status at Nvidia?",
            "ground_truth": "Jensen Huang resigned as CEO, and Colette Kress prepares as interim CEO for Q3 earnings amid personal affairs investigation."
        }
    ]
    
    prepared_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": []
    }
    
    # Execute Queries and Retrieve Document Contexts using our RAG
    for data in eval_dataset:
        print(f"🕵️‍♂️ Fetching and evaluating query: '{data['question']}'")
        try:
            # Query our RAG pipeline
            answer, contexts = rag.query(data['question'])
            
            prepared_data["question"].append(data["question"])
            prepared_data["answer"].append(answer)
            prepared_data["contexts"].append([sample_text])
            prepared_data["ground_truths"].append([data["ground_truth"]])
        except Exception as e:
            print(f"⚠️ Query error during evaluation setup: {e}. Ensure database contains this content.")
            return None

    # Convert to standard format Ragas dataset expects
    # (Using pandas to verify mapping)
    df = pd.DataFrame(prepared_data)
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    
    print("📊 Evaluating performance metrics (Faithfulness & Answer Relevance)...")
    
    # Set the evaluating model with Groq
   # Initialize local HuggingFace embeddings so Ragas doesn't ask for OpenAI
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Run evaluation using Groq as the LLM judge and local HuggingFace embeddings
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=groq_judge,
        embeddings=eval_embeddings
    )
    
    print("\n" + "="*50)
    print("🏆 RAGAS EVALUATION METRICS:")
    print(f"✅ Faithfulness Score: {result['faithfulness']:.4f} (Are answers based ONLY on source data?)")
    print(f"✅ Answer Relevance Score: {result['answer_relevancy']:.4f} (Do the answers address user question?)")
    print("="*50 + "\n")
    
    return result

if __name__ == "__main__":
    run_evaluation()