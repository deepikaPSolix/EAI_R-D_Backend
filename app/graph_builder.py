import asyncio
import aiohttp
import random
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Tuple
from app.canon import canonicalize_url, is_internal_link
import logging
import json
import os

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self,data_dir: str = "data"):
        self.visited = {}
        self.edges = []
        self.data_dir = data_dir  # Initialize data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.session = None
        self._initialize_data_dir()
    def _initialize_data_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Initialized data directory at {os.path.abspath(self.data_dir)}")

    async def _scrape_website(self, url: str, base_url: str):
        if url in self.visited:
            return []

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    return []
                text_content = await response.text()
                soup = BeautifulSoup(text_content, "html.parser")
                extracted_text = soup.get_text(separator="\n", strip=True)
                raw_links = [a['href'] for a in soup.find_all("a", href=True)]
                links = list(dict.fromkeys([canonicalize_url(urljoin(url, link)) for link in raw_links]))
                internal_links = [link for link in links if is_internal_link(base_url, link)]
                self.visited[url] = extracted_text
                return internal_links
        except Exception as e:
            logger.warning(f"aiohttp failed: {e}")
            return await self._playwright_fallback(url, base_url)

    async def _playwright_fallback(self, url: str, base_url: str):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=15000)
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                extracted_text = soup.get_text(separator="\n", strip=True)
                raw_links = [a['href'] for a in soup.find_all("a", href=True)]
                links = list(dict.fromkeys([canonicalize_url(urljoin(url, link)) for link in raw_links]))
                internal_links = [link for link in links if is_internal_link(base_url, link)]
                self.visited[url] = extracted_text
                logger.info(f"Extraction of Text is completed...")
                await browser.close()
                return internal_links
        except Exception as e:
            logger.error(f"Playwright failed: {e}")
            return []

    async def crawl_website(self, start_url: str, max_depth: int = 1):
        self.visited = {}
        self.edges = []
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(connector=connector)
        
        to_crawl = [(start_url, 0)]
        while to_crawl:
            current_url, depth = to_crawl.pop(0)
            if depth > max_depth:
                continue
                
            links = await self._scrape_website(current_url, start_url)
            for link in links:
                self.edges.append((current_url, link))
                if link not in self.visited and link not in [u for u, d in to_crawl]:
                    to_crawl.append((link, depth + 1))
                logger.info('Crawling the webpages...')
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        await self.session.close()
        self._save_to_json(self.visited)
        return self.visited, self.edges
    def _save_to_json(self, visited: Dict[str, str]):
        output_file = os.path.join(self.data_dir, "output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(visited, f, indent=4)
        logger.info(f"Saved extracted data to {output_file}")
    def load_scraped_data(self):
        """Load scraped data from JSON file"""
        input_file = os.path.join(self.data_dir, "output.json")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"No scraped data found at {input_file}")
            return {}