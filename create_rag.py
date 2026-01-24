"""
DesignSafe & SimCenter Documentation RAG System
Scrapes documentation directly from live websites:
- DesignSafe User Guide (https://www.designsafe-ci.org/user-guide/)
- DesignSafe Training (https://www.designsafe-ci.org/user-guide/training/)
- SimCenter Tools (quoFEM, EE-UQ, Hydro, WE-UQ, PBE, R2D)
"""

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
import logging
from dotenv import load_dotenv
from collections import deque

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings

# ChromaDB imports
import chromadb
from chromadb.config import Settings as ChromaSettings

# Web scraping imports
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv()


# Documentation source configurations
DOC_SOURCES = {
    "designsafe-user-guide": {
        "name": "DesignSafe User Guide",
        "base_url": "https://www.designsafe-ci.org/user-guide/",
        "type": "mkdocs",
        "description": "Main DesignSafe documentation for data management, tools, and workflows",
    },
    "designsafe-training": {
        "name": "DesignSafe Training",
        "base_url": "https://www.designsafe-ci.org/user-guide/training/",
        "type": "mkdocs",
        "description": "Training materials and tutorials for DesignSafe",
    },
    "training-opensees": {
        "name": "OpenSees Training",
        "base_url": "https://DesignSafe-CI.github.io/training-OpenSees-on-DesignSafe/",
        "type": "jupyter-book",
        "description": "OpenSees on DesignSafe training materials",
    },
    "training-database-api": {
        "name": "Database API Training",
        "base_url": "https://DesignSafe-CI.github.io/training-database-api/",
        "type": "jupyter-book",
        "description": "Database API training for DesignSafe",
    },
    "training-accelerating-python": {
        "name": "Accelerating Python Training",
        "base_url": "https://DesignSafe-CI.github.io/training-accelerating-python/",
        "type": "jupyter-book",
        "description": "Accelerating Python training",
    },
    "training-xai": {
        "name": "Explainable AI Training",
        "base_url": "https://DesignSafe-CI.github.io/training-xai/",
        "type": "jupyter-book",
        "description": "Explainable AI training",
    },
    "training-pinn": {
        "name": "Physics-Informed Neural Networks Training",
        "base_url": "https://DesignSafe-CI.github.io/training-pinn/",
        "type": "jupyter-book",
        "description": "Physics-Informed Neural Networks training",
    },
    "training-deeponet": {
        "name": "DeepONet Training",
        "base_url": "https://DesignSafe-CI.github.io/training-deeponet/",
        "type": "jupyter-book",
        "description": "DeepONet training",
    },
    "quofem": {
        "name": "quoFEM",
        "base_url": "https://nheri-simcenter.github.io/quoFEM-Documentation/",
        "type": "sphinx",
        "description": "Quantified Uncertainty with Optimization for the FEM",
    },
    "ee-uq": {
        "name": "EE-UQ",
        "base_url": "https://nheri-simcenter.github.io/EE-UQ-Documentation/",
        "type": "sphinx",
        "description": "Earthquake Engineering with Uncertainty Quantification",
    },
    "hydro-uq": {
        "name": "Hydro-UQ",
        "base_url": "https://nheri-simcenter.github.io/Hydro-Documentation/",
        "type": "sphinx",
        "description": "Water-borne Hazards Engineering with Uncertainty Quantification",
    },
    "we-uq": {
        "name": "WE-UQ",
        "base_url": "https://nheri-simcenter.github.io/WE-UQ-Documentation/",
        "type": "sphinx",
        "description": "Wind Engineering with Uncertainty Quantification",
    },
    "pbe": {
        "name": "PBE",
        "base_url": "https://nheri-simcenter.github.io/PBE-Documentation/",
        "type": "sphinx",
        "description": "Performance-Based Engineering Application",
    },
    "r2d": {
        "name": "R2D",
        "base_url": "https://nheri-simcenter.github.io/R2D-Documentation/",
        "type": "sphinx",
        "description": "Regional Resilience Determination Tool",
    },
}


class WebScraper:
    """Web scraper for documentation sites."""

    def __init__(self, delay: float = 0.3, max_pages: int = 0):
        """
        Initialize the web scraper.

        Args:
            delay: Delay between requests in seconds (be polite to servers)
            max_pages: Maximum number of pages to scrape per source
        """
        self.delay = delay
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DesignSafe-RAG-Bot/1.0 (Educational/Research Purpose)'
        })

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a single page, following redirects."""
        try:
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Check for meta refresh redirect
            meta_refresh = soup.find('meta', attrs={'http-equiv': 'Refresh'})
            if meta_refresh and meta_refresh.get('content'):
                content = meta_refresh.get('content', '')
                if 'url=' in content.lower():
                    redirect_url = content.split('url=')[-1].strip()
                    full_redirect = urljoin(url, redirect_url)
                    logger.info(f"    Following redirect to {full_redirect}")
                    return self.fetch_page(full_redirect)

            return soup
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def get_links(self, soup: BeautifulSoup, base_url: str, current_url: str) -> Set[str]:
        """Extract all internal documentation links from a page."""
        links = set()
        parsed_base = urlparse(base_url)

        for a in soup.find_all('a', href=True):
            href = a['href']

            # Skip anchors, external links, and special links
            if href.startswith('#') or href.startswith('mailto:') or href.startswith('javascript:'):
                continue

            # Resolve relative URLs
            full_url = urljoin(current_url, href)
            parsed_url = urlparse(full_url)

            # Remove fragment
            full_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

            # Only include links within the documentation
            if parsed_url.netloc == parsed_base.netloc and full_url.startswith(base_url):
                # Skip non-HTML resources and source files
                skip_extensions = ['.pdf', '.zip', '.png', '.jpg', '.jpeg', '.gif', '.svg',
                                   '.css', '.js', '.txt', '.rst', '.md', '.csv', '.json',
                                   '.py', '.ipynb', '.xml', '.yaml', '.yml']
                skip_patterns = ['/_images/', '/_static/', '/_sources/', '/_downloads/']

                if any(full_url.lower().endswith(ext) for ext in skip_extensions):
                    continue
                if any(pat in full_url for pat in skip_patterns):
                    continue
                links.add(full_url)

        return links

    def extract_mkdocs_content(self, soup: BeautifulSoup, url: str, source_name: str) -> List[Document]:
        """Extract content from MkDocs-style pages (DesignSafe uses ReadTheDocs theme)."""
        documents = []

        # Find main content area - DesignSafe uses ReadTheDocs theme with tacc_readthedocs div
        main_content = (
            soup.find('div', id='tacc_readthedocs') or
            soup.find('div', class_='rst-content') or
            soup.find('div', class_='document') or
            soup.find('div', {'role': 'main'}) or
            soup.find('article', class_='md-content__inner') or
            soup.find('main') or
            soup.find('article')
        )

        if not main_content:
            return documents

        # Get page title
        title_tag = main_content.find('h1') or soup.find('h1')
        page_title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path.split('/')[-2]

        # Remove navigation, breadcrumbs, scripts, styles
        for elem in main_content.find_all(['nav', 'footer', 'script', 'style']):
            elem.decompose()
        for elem in main_content.find_all(class_=['headerlink', 'fa-link']):
            elem.decompose()

        # Extract by headers - works for DesignSafe's simple structure
        current_section = page_title
        current_content = []
        current_anchor = ""

        # Get all relevant elements in order
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'pre', 'table', 'dl', 'blockquote']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Save previous section
                if current_content:
                    text = '\n'.join(current_content).strip()
                    if text and len(text) > 30:
                        section_url = f"{url}#{current_anchor}" if current_anchor else url
                        documents.append(Document(
                            text=text,
                            metadata={
                                'title': page_title,
                                'section': current_section,
                                'url': section_url,
                                'source': source_name,
                            }
                        ))

                # Start new section
                current_section = element.get_text(strip=True)
                current_anchor = element.get('id', '')
                current_content = []
            else:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 10:  # Skip very short elements
                    current_content.append(text)

        # Don't forget last section
        if current_content:
            text = '\n'.join(current_content).strip()
            if text and len(text) > 30:
                section_url = f"{url}#{current_anchor}" if current_anchor else url
                documents.append(Document(
                    text=text,
                    metadata={
                        'title': page_title,
                        'section': current_section,
                        'url': section_url,
                        'source': source_name,
                    }
                ))

        return documents

    def extract_sphinx_content(self, soup: BeautifulSoup, url: str, source_name: str) -> List[Document]:
        """Extract content from Sphinx-style pages (SimCenter)."""
        documents = []

        # Find main content area
        main_content = (
            soup.find('div', class_='document') or
            soup.find('div', class_='body') or
            soup.find('main') or
            soup.find('article')
        )

        if not main_content:
            return documents

        # Get page title
        title_tag = soup.find('title')
        page_title = title_tag.get_text(strip=True) if title_tag else ""
        # Clean up title
        for sep in [' — ', ' - ', ' | ']:
            if sep in page_title:
                page_title = page_title.split(sep)[0].strip()
                break

        # Remove navigation, sidebar, footer elements
        for elem in main_content.find_all(['nav', 'footer', 'script', 'style']):
            elem.decompose()
        for elem in main_content.find_all(class_=['sidebar', 'navigation', 'sphinxsidebar', 'toctree-wrapper', 'headerlink']):
            elem.decompose()

        # Find all sections
        sections = main_content.find_all('section')

        if sections:
            for section in sections:
                section_id = section.get('id', '')
                header = section.find(['h1', 'h2', 'h3', 'h4'])
                section_title = header.get_text(strip=True) if header else page_title

                # Get section text (excluding nested sections)
                text_parts = []
                for child in section.children:
                    if hasattr(child, 'name'):
                        if child.name == 'section':
                            continue  # Skip nested sections
                        if child.name in ['p', 'ul', 'ol', 'pre', 'table', 'div', 'dl']:
                            text = child.get_text(separator=' ', strip=True)
                            if text:
                                text_parts.append(text)

                text = '\n'.join(text_parts).strip()
                if text and len(text) > 50:
                    # Use page URL without section anchor for cleaner references
                    documents.append(Document(
                        text=text,
                        metadata={
                            'title': page_title,
                            'section': section_title,
                            'url': url,  # Use base URL, not anchored
                            'source': source_name,
                        }
                    ))
        else:
            # No sections, extract all content
            text = main_content.get_text(separator='\n', strip=True)
            if text and len(text) > 50:
                documents.append(Document(
                    text=text,
                    metadata={
                        'title': page_title,
                        'section': page_title,
                        'url': url,
                        'source': source_name,
                    }
                ))

        return documents

    def extract_jupyter_book_content(self, soup: BeautifulSoup, url: str, source_name: str) -> List[Document]:
        """Extract content from Jupyter Book / Sphinx Book Theme pages."""
        documents = []

        # Find main content area - Jupyter Book / Sphinx Book Theme
        main_content = (
            soup.find('article', class_='bd-article') or
            soup.find('main', id='main-content') or
            soup.find('div', class_='bd-content') or
            soup.find('div', class_='content') or
            soup.find('main') or
            soup.find('article')
        )

        if not main_content:
            return documents

        # Get page title
        title_tag = main_content.find('h1') or soup.find('h1')
        page_title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path.split('/')[-1].replace('.html', '')

        # Remove navigation elements
        for elem in main_content.find_all(['nav', 'footer', 'script', 'style']):
            elem.decompose()
        for elem in main_content.find_all(class_=['headerlink', 'toc-entry', 'cell_input', 'cell_output']):
            elem.decompose()

        # Extract by sections
        sections = main_content.find_all('section')

        if sections:
            for section in sections:
                section_id = section.get('id', '')
                header = section.find(['h1', 'h2', 'h3', 'h4'])
                section_title = header.get_text(strip=True) if header else page_title

                # Get section text
                text_parts = []
                for child in section.children:
                    if hasattr(child, 'name'):
                        if child.name == 'section':
                            continue
                        if child.name in ['p', 'ul', 'ol', 'pre', 'div', 'dl', 'blockquote']:
                            text = child.get_text(separator=' ', strip=True)
                            if text and len(text) > 10:
                                text_parts.append(text)

                text = '\n'.join(text_parts).strip()
                if text and len(text) > 50:
                    documents.append(Document(
                        text=text,
                        metadata={
                            'title': page_title,
                            'section': section_title,
                            'url': url,
                            'source': source_name,
                        }
                    ))
        else:
            # Fallback: extract by headers
            current_section = page_title
            current_content = []

            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'pre', 'div']):
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    if current_content:
                        text = '\n'.join(current_content).strip()
                        if text and len(text) > 50:
                            documents.append(Document(
                                text=text,
                                metadata={
                                    'title': page_title,
                                    'section': current_section,
                                    'url': url,
                                    'source': source_name,
                                }
                            ))
                    current_section = element.get_text(strip=True)
                    current_content = []
                else:
                    text = element.get_text(separator=' ', strip=True)
                    if text and len(text) > 10:
                        current_content.append(text)

            if current_content:
                text = '\n'.join(current_content).strip()
                if text and len(text) > 50:
                    documents.append(Document(
                        text=text,
                        metadata={
                            'title': page_title,
                            'section': current_section,
                            'url': url,
                            'source': source_name,
                        }
                    ))

        return documents

    def normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates (index.html vs /)."""
        # Remove trailing slash
        url = url.rstrip('/')
        # Remove index.html
        if url.endswith('/index.html'):
            url = url[:-11]
        elif url.endswith('index.html'):
            url = url[:-10]
        return url

    def scrape_site(self, source_key: str) -> List[Document]:
        """Scrape an entire documentation site."""
        source = DOC_SOURCES[source_key]
        base_url = source['base_url']
        source_name = source['name']
        doc_type = source['type']

        logger.info(f"Scraping {source_name} from {base_url}")

        all_documents = []
        visited = set()
        to_visit = deque([base_url])

        # max_pages = 0 means no limit
        while to_visit and (self.max_pages == 0 or len(visited) < self.max_pages):
            url = to_visit.popleft()

            # Normalize URL to avoid duplicates
            normalized_url = self.normalize_url(url)
            if normalized_url in visited:
                continue

            visited.add(normalized_url)
            limit_str = str(self.max_pages) if self.max_pages > 0 else "unlimited"
            logger.info(f"  [{len(visited)}/{limit_str}] {url}")

            soup = self.fetch_page(url)
            if not soup:
                continue

            # Extract content based on doc type
            if doc_type == 'mkdocs':
                documents = self.extract_mkdocs_content(soup, url, source_name)
            elif doc_type == 'jupyter-book':
                documents = self.extract_jupyter_book_content(soup, url, source_name)
            else:  # sphinx
                documents = self.extract_sphinx_content(soup, url, source_name)

            all_documents.extend(documents)

            # Find more links
            links = self.get_links(soup, base_url, url)
            for link in links:
                normalized_link = self.normalize_url(link)
                if normalized_link not in visited:
                    to_visit.append(link)

            # Be polite
            time.sleep(self.delay)

        logger.info(f"  Scraped {len(visited)} pages, extracted {len(all_documents)} document chunks")
        return all_documents


class DesignSafeRAG:
    """RAG system for DesignSafe and SimCenter documentation."""

    def __init__(self,
                 chroma_path: str = "./chroma_db",
                 collection_name: str = "designsafe_docs"):
        """
        Initialize the RAG system.

        Args:
            chroma_path: Path to store ChromaDB database
            collection_name: Name of the ChromaDB collection
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name

        # Initialize OpenAI embedding model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=api_key
        )

        # Set global settings for LlamaIndex
        Settings.embed_model = self.embed_model

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
        except:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")

        # Initialize vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

        # Initialize scraper (max_pages=0 means no limit)
        self.scraper = WebScraper(delay=0.3, max_pages=0)

    def reset_collection(self):
        """Delete and recreate the collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        logger.info(f"Created new collection: {self.collection_name}")

    def scrape_source(self, source_key: str) -> List[Document]:
        """Scrape a single documentation source."""
        if source_key not in DOC_SOURCES:
            raise ValueError(f"Unknown source: {source_key}. Available: {list(DOC_SOURCES.keys())}")

        return self.scraper.scrape_site(source_key)

    def scrape_all_sources(self) -> List[Document]:
        """Scrape all configured documentation sources."""
        all_documents = []

        for source_key in DOC_SOURCES:
            try:
                documents = self.scrape_source(source_key)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error scraping source {source_key}: {e}")

        return all_documents

    def scrape_designsafe_only(self) -> List[Document]:
        """Scrape only DesignSafe sources (User Guide + Training sites)."""
        documents = []
        designsafe_sources = [
            "designsafe-user-guide",
            "designsafe-training",
            "training-opensees",
            "training-database-api",
            "training-accelerating-python",
            "training-xai",
            "training-pinn",
            "training-deeponet",
        ]
        for source_key in designsafe_sources:
            try:
                documents.extend(self.scraper.scrape_site(source_key))
            except Exception as e:
                logger.error(f"Error scraping {source_key}: {e}")
        return documents

    def scrape_simcenter_only(self) -> List[Document]:
        """Scrape only SimCenter sources."""
        documents = []
        simcenter_sources = ["quofem", "ee-uq", "hydro-uq", "we-uq", "pbe", "r2d"]
        for source_key in simcenter_sources:
            documents.extend(self.scraper.scrape_site(source_key))
        return documents

    def build_index(self, documents: List[Document]):
        """Build the vector index from documents."""
        logger.info(f"Building vector index with {len(documents)} documents...")

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        logger.info("Vector index built successfully!")
        return index

    def query(self, query_text: str, top_k: int = 5):
        """Query the vector database with detailed documentation-style responses."""
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model
        )

        # Custom prompt for documentation-style responses
        qa_prompt_template = PromptTemplate(
            """You are a technical documentation assistant for DesignSafe and SimCenter tools.
Provide accurate, complete answers based on the official documentation.

Context from documentation:
---------------------
{context_str}
---------------------

Answer the question using ALL relevant information from the context above. Do not omit options, steps, or details that are present in the documentation.

Guidelines:
- For "what is" questions: Clear explanation with key features (2-4 sentences)
- For "how to" questions: Include ALL methods/options mentioned in the docs, with specific details for each
- For lists of options/features: Include every item from the documentation
- Include specific commands, paths, or tool names when mentioned

Question: {query_str}

Answer:"""
        )

        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            text_qa_template=qa_prompt_template
        )
        response = query_engine.query(query_text)

        return response


def main():
    """Main function to create and populate the RAG system."""
    rag = DesignSafeRAG()

    print("\n" + "=" * 60)
    print("DesignSafe & SimCenter Documentation RAG System")
    print("=" * 60)

    print("\nAvailable documentation sources:")
    print("\n  DesignSafe:")
    for key in ["designsafe-user-guide", "designsafe-training"]:
        source = DOC_SOURCES[key]
        print(f"    - {source['name']}: {source['base_url']}")

    print("\n  Training Sub-sites (Jupyter Books):")
    training_keys = ["training-opensees", "training-database-api", "training-accelerating-python",
                     "training-xai", "training-pinn", "training-deeponet"]
    for key in training_keys:
        source = DOC_SOURCES[key]
        print(f"    - {source['name']}: {source['base_url']}")

    print("\n  SimCenter Tools:")
    for key in ["quofem", "ee-uq", "hydro-uq", "we-uq", "pbe", "r2d"]:
        source = DOC_SOURCES[key]
        print(f"    - {source['name']}: {source['base_url']}")

    print("\nChoose processing option:")
    print("  a. Scrape ALL sources (DesignSafe + SimCenter)")
    print("  b. Scrape DesignSafe only (User Guide + Training)")
    print("  c. Scrape SimCenter tools only (6 tools)")
    print("  d. Scrape a specific source")
    print("  r. Reset database and scrape ALL sources")
    print("  q. Query existing database")

    choice = input("\nEnter choice (a/b/c/d/r/q): ").strip().lower()

    documents = []

    if choice == 'q':
        # Query mode
        print("\nEntering query mode. Type 'exit' to quit.\n")
        while True:
            query = input("Enter your question: ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                continue

            print("\nSearching...\n")
            response = rag.query(query, top_k=5)

            print("Answer:")
            print("-" * 40)
            print(response.response)
            print()

            if hasattr(response, 'source_nodes'):
                print("Sources:")
                print("-" * 40)
                for i, node in enumerate(response.source_nodes, 1):
                    metadata = node.metadata
                    print(f"{i}. [{metadata.get('source', 'Unknown')}] {metadata.get('title', 'N/A')}")
                    print(f"   Section: {metadata.get('section', 'N/A')}")
                    print(f"   URL: {metadata.get('url', 'N/A')}")
                    print()
        return

    elif choice == 'r':
        print("\nResetting database...")
        rag.reset_collection()
        documents = rag.scrape_all_sources()

    elif choice == 'a':
        documents = rag.scrape_all_sources()

    elif choice == 'b':
        documents = rag.scrape_designsafe_only()

    elif choice == 'c':
        documents = rag.scrape_simcenter_only()

    elif choice == 'd':
        print("\nAvailable sources:")
        for i, (key, source) in enumerate(DOC_SOURCES.items(), 1):
            print(f"  {i}. {key} - {source['name']}")

        source_choice = input("\nEnter source key (e.g., 'ee-uq'): ").strip().lower()
        if source_choice in DOC_SOURCES:
            documents = rag.scrape_source(source_choice)
        else:
            print(f"Unknown source: {source_choice}")
            return

    else:
        print("Invalid choice.")
        return

    if documents:
        # Build index
        index = rag.build_index(documents)
        print(f"\nSuccessfully created RAG system with {len(documents)} document chunks!")

        # Show breakdown by source
        source_counts = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        print("\nDocument breakdown by source:")
        for source, count in sorted(source_counts.items()):
            print(f"  - {source}: {count} chunks")

        # Test query
        print("\n" + "=" * 60)
        print("Testing the RAG system...")
        test_query = "What is EE-UQ and how do I install it?"
        print(f"Query: {test_query}")
        response = rag.query(test_query)
        print(f"\nResponse: {response}")
    else:
        print("No documents were scraped. Please check the configuration.")


if __name__ == "__main__":
    main()
