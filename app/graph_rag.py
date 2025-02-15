# graph_rag.py
import asyncio
import json
import io
import os
import ssl
import time
import textwrap
from urllib.parse import urlparse, urljoin
from typing import List, Tuple
import networkx as nx
import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import concurrent.futures
from flask import current_app, copy_current_request_context
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pyvis.network import Network
from .llm_model import LLMModel  # Import your existing LLMModel

class GraphRAGProcessor:
    def __init__(self):
        self.scrape_cache = {}
        self.llm = LLMModel.from_together()  # Use your existing LLMModel setup
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.app = current_app._get_current_object()
        self.vectorstore = None
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    def _generate_summary_wrapper(self, text):
        """Wrapper to preserve app context across threads"""
        def _context_aware_generation():
            with self.app.app_context():
                return self._generate_summary(text)
        return _context_aware_generation
    def _canonicalize_url(self, url):
        """Normalize URL format with proper logging"""
        try:
            parsed = urlparse(url)
            parsed = parsed._replace(fragment="")
            path = parsed.path
            if path != "/" and path.endswith("/"):
                path = path.rstrip("/")
            parsed = parsed._replace(path=path)
            current_app.logger.debug(f"Canonicalized URL: {url} -> {parsed.geturl()}")
            return parsed.geturl()
        except Exception as e:
            current_app.logger.error(f"URL canonicalization failed: {str(e)}")
            return url

    async def _scrape_website(self, url, session, retries=3, timeout=10):
        """Cached async scraper with enhanced error handling"""
        if url in self.scrape_cache:
            current_app.logger.debug(f"Using cached content for {url}")
            return self.scrape_cache[url]

        error = None
        for attempt in range(retries):
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                async with session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    ssl=self.ssl_context
                ) as response:
                    if response.status != 200:
                        current_app.logger.warning(
                            f"HTTP {response.status} for {url} (attempt {attempt+1}/{retries})"
                        )
                        continue

                    content_type = response.headers.get('Content-Type', '')
                    if "text/html" not in content_type:
                        current_app.logger.info(f"Skipping non-HTML content at {url}")
                        result = ([], "")
                        self.scrape_cache[url] = result
                        return result

                    text = await response.text()
                    soup = BeautifulSoup(text, "html.parser")

                    # Text extraction
                    paragraphs = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
                    headers_text = " ".join(
                        h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])
                    )
                    page_text = f"{paragraphs} {headers_text}".strip()

                    # Link processing
                    raw_links = [a['href'] for a in soup.find_all("a", href=True)]
                    links = [self._canonicalize_url(urljoin(url, link)) for link in raw_links]
                    links = list(dict.fromkeys(links))  # Deduplicate

                    result = (links, page_text)
                    self.scrape_cache[url] = result
                    current_app.logger.info(f"Successfully scraped {url}")
                    return result

            except Exception as e:
                error = e
                current_app.logger.error(
                    f"\n🔴 Scrape Error ({attempt+1}/{retries}): {url}\n"
                    f"Error Type: {type(e).__name__}\n"
                    f"Details: {str(e)}\n"
                    f"{'-'*40}"
                )
                await asyncio.sleep(1)

        current_app.logger.error(f"Permanently failed to scrape {url}")
        result = ([], "")
        self.scrape_cache[url] = result
        return result

    async def build_graph(self, start_url: str, allowed_domain: str) -> Tuple[nx.DiGraph, List[tuple]]:
        """Build website graph with proper logging and error handling"""
        graph = nx.DiGraph()
        edge_urls = []

        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                current_app.logger.info(f"\n🌐 Starting graph build for {start_url}\n")

                # Initial scrape
                base_links, base_text = await self._scrape_website(start_url, session)
                root_label = self._extract_node_label(start_url, 0)
                graph.add_node(root_label, text=base_text)

                # Process product links
                product_links = self._filter_product_links(
                    self._filter_links(base_links, start_url, allowed_domain)
                )
                for link in product_links:
                    second_links, link_text = await self._scrape_website(link, session)
                    level_1_label = self._extract_node_label(link, 1)
                    
                    graph.add_node(level_1_label, text=link_text)
                    graph.add_edge(root_label, level_1_label)
                    edge_urls.append((root_label, link))

                    # Process second layer
                    second_layer_links = self._filter_product_links(
                        self._filter_links(second_links, link, allowed_domain)
                    )
                    current_app.logger.debug(f"Found {len(second_layer_links)} second-layer links")

                    for second_link in second_layer_links:
                        _, second_text = await self._scrape_website(second_link, session)
                        level_2_label = self._extract_node_label(second_link, 2)
                        
                        graph.add_node(level_2_label, text=second_text)
                        graph.add_edge(level_1_label, level_2_label)
                        edge_urls.append((level_1_label, second_link))

        except Exception as e:
            current_app.logger.error(
                f"\n🔥 Critical Graph Build Error\n"
                f"URL: {start_url}\n"
                f"Error: {str(e)}\n"
                f"{'-'*40}"
            )
            raise

        return graph
    
    def generate_visualization(self, graph: nx.DiGraph) -> str:
            try:
                net = Network(height="800px", width="100%", directed=True, bgcolor="#1E1E1E", font_color="#FFFFFF")       
                net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 25,
            "font": {
              "size": 16,
              "face": "arial",
              "strokeWidth": 2
            },
            "borderWidth": 2,
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.5)",
              "size": 10
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "roundness": 0.4
            },
            "color": {
              "inherit": "both"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.8
              }
            },
            "width": 1.5,
            "hoverWidth": 2
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "solver": "barnesHut",
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 100
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "keyboard": {
              "enabled": true,
              "speed": {
                "x": 10,
                "y": 10,
                "zoom": 0.02
              }
            }
          }
        }
        """)
                node_data = list(graph.nodes(data=True))
                texts = [data.get("text", "") for _, data in node_data]

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    # Map wrapped functions with preserved context
                    futures = [
                        executor.submit(self._generate_summary_wrapper(text))
                        for text in texts
                    ]
                    
                    summaries = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            summary = future.result()
                            summaries.append(summary)
                        except Exception as e:
                            self.app.logger.error(f"Summary failed: {str(e)}")
                            summaries.append("Summary error")

                # Add nodes with summaries
                for (node_label, _), summary in zip(node_data, summaries):
                    wrapped_text = "\n".join(textwrap.wrap(summary, width=40))
                    net.add_node(node_label, 
                                label=node_label,
                                title=wrapped_text,
                                color=self._node_color(node_label))
                    self.app.logger.debug(f"Added node: {node_label}")

            
                for edge in graph.edges():
                    net.add_edge(edge[0], edge[1])
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                output_path = os.path.join(desktop_path, "graph_visualization.html")
            
            # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                net.write_html(output_path)  # Save the visualization to the specified path

                current_app.logger.info(f"Visualization generated at {output_path}")
                return output_path

            except Exception as e:
                current_app.logger.error(f"Visualization failed: {str(e)}")
                raise

    def _generate_summary(self, text: str) -> str:
        
        current_app.logger.info('TEXT RECIEVED $$$$$$$$$$@$#@##@#@#')
        """Use LLMModel to generate summaries"""
        if not text.strip():
            current_app.logger.info("NO CONTENT AVAILABLE!@@!@!#!#")
            return "No content available"
        prompt = f"Summarize this in one line:\n{text}\nSummary:"
       
        try:
            prompt = f"Summarize this in one line:\n{text}\nSummary:"
            response = self.llm.model.invoke(prompt).content.strip()
            current_app.logger.info(f"Generated Summary: {response}")
            return response
        except Exception as e:
            current_app.logger.error(f"Summary generation failed: {str(e)}")
            return "Summary unavailable"

    def create_vector_store(self, graph: nx.DiGraph):
        """Create FAISS index from graph data"""
        try:
            docs = [
                Document(page_content=data["text"])
                for _, data in graph.nodes(data=True)
                if data.get("text")
            ]
            self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
            self.vectorstore.save_local("faiss_index")
            current_app.logger.info(f"Created vector store with {len(docs)} documents")
        except Exception as e:
            current_app.logger.error(f"Vector store creation failed: {str(e)}")
            raise
    def load_vector_store(self):
        """Load the FAISS vector store from disk."""
        try:
            self.vectorstore = FAISS.load_local("faiss_index", self.embedding_model, allow_dangerous_deserialization=True)
            current_app.logger.info("Vector store loaded successfully.")
        except Exception as e:
            current_app.logger.error(f"Failed to load vector store: {str(e)}")
            raise
   
    def query_graph(self, query: str, k: int = 5) -> str:
        try:
            
            if not self.vectorstore:
                raise ValueError("Vector store not initialized - call create_vector_store first")

            # Step 1: Retrieve relevant context from FAISS
            retrieved_texts = self._graph_rag_retrieval(query, k=k)
            
            # Step 2: Format prompt using your existing template
            prompt = self._format_prompt(query, retrieved_texts)
            
            # Step 3: Generate response using your LLMModel
            response = self._generate_llm_response(prompt,retrieved_texts)
            
            return response

        except Exception as e:
            self.app.logger.error(f"Query failed: {str(e)}")
            raise

    def _graph_rag_retrieval(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top k relevant documents from FAISS index"""
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def _format_prompt(self, query: str, retrieved_texts: List[str], max_context: int = 3) -> str:
        """Format prompt using your existing template"""
        if not retrieved_texts:
            retrieved_texts = ["No relevant context found."]

        context_text = "\n".join(retrieved_texts[:max_context])

        return f"""You are an AI assistant answering questions **strictly** based on the given context.
        Provide a **short, precise answer** with no extra details. Use bullet points or direct sentences only when necessary.
        Keep your response under 3-4 sentences. Do not give your own response and no hallucinations please.

    Context:
    {context_text}

    User Query: {query}

    Response (Answer only based on the above context in 3-4 sentences):
    """

    def _generate_llm_response(self, prompt: str, retrieved_text:str) -> str:
        """Generate response using your LLMModel"""
        try:
            # Use your existing LLM integration
            response = self.llm.model.invoke(prompt).content.strip()
            return response
        
        except Exception as e:
            self.app.logger.error(f"LLM response failed: {str(e)}")
            return "Response unavailable"

    # Helper methods
    def _extract_node_label(self, url: str, level: int) -> str:
        parsed = urlparse(url)
        if level == 0:
            return parsed.netloc
        path_parts = parsed.path.strip('/').split('/')
        return path_parts[level-1] if len(path_parts) >= level else url

    def _filter_product_links(self, links: List[str]) -> List[str]:
        return [link for link in links
                if any(kw in urlparse(link).path.lower() for kw in ["product", "products"])]

    def _filter_links(self, links: List[str], base_url: str, allowed_domain: str) -> List[str]:
        return [link for link in links if allowed_domain in urlparse(link).netloc]

    def _node_color(self, label: str) -> str:
        return f"#{hash(label) % 0xFFFFFF:06x}"
    
