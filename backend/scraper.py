import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import urllib.parse
import time

class WebScraper:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        self.chrome_options = chrome_options

    def scrape_with_requests(self, url: str) -> str:
        """Fast fallback using requests and bs4 for static websites"""
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        for trash in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            trash.extract()
            
        return soup.get_text(separator="\n")

    def scrape_with_selenium(self, url: str) -> str:
        """Dynamic scraper using Headless Selenium when requests fail or page requires JavaScript"""
        driver = webdriver.Chrome(options=self.chrome_options)
        try:
            driver.get(url)
            driver.implicitly_wait(5)
            body_element = driver.find_element(By.TAG_NAME, "body")
            text_content = body_element.text
            return text_content
        finally:
            driver.quit()

    def clean_text(self, raw_text: str) -> str:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(lines)

    def extract_links(self, parent_url: str, limit: int = 6) -> list:
        """Analyze parent URL, inspect <a> tags, and capture relative/nested sub-page targets to follow"""
        print(f"🔗 Extracting sub-article links from: {parent_url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            # Try using bs4 to grab anchors rapidly
            response = requests.get(parent_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception:
            # Selenium fallback if BS4 request is blocked
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(parent_url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()
            
        links = []
        base_parsed = urllib.parse.urlparse(parent_url)
        
        # Scrape all hyperlinked anchors
        for anchor in soup.find_all("a", href=True):
            href = anchor['href']
            # Convert relative URL to full absolute path
            full_url = urllib.parse.urljoin(parent_url, href)
            
            parsed_full = urllib.parse.urlparse(full_url)
            
            # Subdomain safeguard: allow subdomains like finance.yahoo.com or news.yahoo.com
            if not parsed_full.netloc.endswith(base_parsed.netloc.split('.')[-2] + '.' + base_parsed.netloc.split('.')[-1]):
                 continue # Ignore completely external sites like Twitter/FB
                 
            # Modern Article Link Matcher (including nested live blogs, finance, or technology sections)
            path_lower = parsed_full.path.lower()
            is_article = (
                "news" in path_lower or 
                "article" in path_lower or 
                "technology" in path_lower or 
                "live" in path_lower or 
                "m/" in path_lower or
                "finance/" in path_lower or
                parsed_full.path.endswith('.html')
            )
            
            # Avoid overly brief landing page/section indexes
            if not is_article or len(parsed_full.path.split('/')) < 3:
                 continue 
                 
            if full_url not in links and full_url != parent_url:
                links.append(full_url)
                if len(links) >= limit: # Respect Crawler limits to avoid blocking
                    break
        
        print(f"🎯 Found {len(links)} candidate article URLs to fetch: {links}")
        return list(set(links)) # Avoid duplicates

    def ingest(self, url: str) -> str:
        """Try requests, fallback to Selenium"""
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
            
        try:
            raw_data = self.scrape_with_requests(url)
            cleaned = self.clean_text(raw_data)
            if len(cleaned) < 200:
                raise Exception("Minimal content returned. Likely dynamic JS page.")
            return cleaned
        except Exception as e:
            try:
                raw_data = self.scrape_with_selenium(url)
                cleaned = self.clean_text(raw_data)
                if not cleaned:
                    raise Exception("No content scraped.")
                return cleaned
            except Exception as se:
                raise RuntimeError(f"Scraper pipelines failed: {se}")

    def crawl_and_ingest(self, url: str, limit: int = 4) -> list:
        """
        Master Multi-Page Ingestion:
        1. Ingest parent index/landing page.
        2. Crawl and ingest 'limit' nested full articles on that page.
        3. Returns list of dictionary data: [{'url': u, 'text': t}]
        """
        results = []
        # Step 1: Handle Parent Landing Page
        try:
            parent_text = self.ingest(url)
            results.append({"url": url, "text": parent_text})
        except Exception as e:
            print(f"⚠️ Could not ingest parent {url}: {e}")
            
        # Step 2: Grab nested article links
        try:
            child_links = self.extract_links(url, limit=limit)
        except Exception as e:
            print(f"⚠️ Link extraction failed: {e}")
            child_links = []
            
        # Step 3: Iterate and fetch sub-pages
        for i, link in enumerate(child_links):
            print(f"🚀 Ingesting sub-article [{i+1}/{len(child_links)}]: {link}")
            try:
                # Add tiny delay to be nice to servers
                time.sleep(1.5)
                text = self.ingest(link)
                # Ensure we got meaningful content
                if len(text) > 400:
                    results.append({"url": link, "text": text})
            except Exception as e:
                print(f"❌ Failed to parse child URL {link}: {e}")
                
        return results

if __name__ == "__main__":
    scraper = WebScraper()
    test_url = "https://finance.yahoo.com/topic/latest-news/"
    # Run a test of crawls
    res = scraper.crawl_and_ingest(test_url, limit=2)
    print(f"\n--- CRAWL RESULT PREVIEW ---")
    print(f"Fetched {len(res)} pages total (Parent + sub-articles).")
    for r in res:
         print(f"URL: {r['url']} ({len(r['text'])} chars loaded)")