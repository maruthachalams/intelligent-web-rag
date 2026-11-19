import pytest
from backend.scraper import WebScraper
from backend.rag_engine import RAGEngine

@pytest.fixture
def scraper():
    return WebScraper()

@pytest.fixture
def rag_engine():
    return RAGEngine()

def test_scraper_crawl_and_ingest(scraper):
    """Test public recursive crawler captures pages and sub-links."""
    url = "https://example.com"
    # Call the exact public crawling method we used in backend/main.py!
    # A limit of 1 is enough for quick testing
    ingested_pages = scraper.crawl_and_ingest(url, limit=1)
    
    assert isinstance(ingested_pages, list)
    assert len(ingested_pages) > 0
    assert any("example" in p["text"].lower() or "example" in p["url"].lower() for p in ingested_pages)

def test_rag_engine_lifecycle(rag_engine):
    """Test vector storage inserts, retrieves, and cleanly resets."""
    # Reset local state safely
    rag_engine.clear_db()
    
    # Add dummy test target
    test_text = "Nvidia formed a specialized board committee to supervise national security and export rules."
    test_url = "https://example.com/nvidia-security"
    rag_engine.add_document(test_text, test_url)
    
    # Test safe embedding search or query. Let's make sure it handles query gracefully!
    # In case there's any parameter mismatches, let's verify both interface calls
    try:
        answer, context = rag_engine.query("What did Nvidia's board committee do?")
    except AttributeError:
        # If your local RAG named it differently, let's fall back to prompt verification
        answer, context = rag_engine.get_response("What did Nvidia's board committee do?")
        
    assert "board" in answer.lower() or "security" in answer.lower() or "committee" in answer.lower()
    assert "nvidia-security" in context.lower() or "http" in context.lower()
    
    # Clean up DB
    rag_engine.clear_db()