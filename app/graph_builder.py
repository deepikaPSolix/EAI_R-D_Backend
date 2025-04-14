import asyncio
import aiohttp
import random
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from flask import current_app
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Tuple
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from app.canon import canonicalize_url, is_internal_link
import logging
import json
import os



class GraphBuilder:
    def __init__(self):
        self.visited = {}
        self.edges = []
    async def _scrape_website(self, url: str, base_url: str):
        url = url.strip()
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
            current_app.logger.warning(f"aiohttp failed: {e}")
            return await self._playwright_fallback(url, base_url)

    async def _playwright_fallback(self, url: str, base_url: str):
        url = url.strip()
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
                current_app.logger.info(f"Extraction of Text is completed...")
                await browser.close()
                return internal_links
        except Exception as e:
            current_app.logger.error(f"Playwright failed: {e}")
            return []

    async def crawl_website(self, start_url: str, max_depth: int = 1):
        self.visited = {}
        self.edges = []
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(connector=connector)
        start_url = start_url.strip()
        to_crawl = [(start_url, 0)]
        current_app.logger.info('Crawling the webpages...')
        while to_crawl:
            current_url, depth = to_crawl.pop(0)
            if int(depth) > max_depth:
                continue
                
            links = await self._scrape_website(current_url, start_url)
            
            for link in links:
                self.edges.append((current_url, link))
                if link not in self.visited and link not in [u for u, d in to_crawl]:
                    to_crawl.append((link, depth + 1))
            await asyncio.sleep(random.uniform(0.5, 1.5))
        current_app.logger.info('Crawling DONE...')
        await self.session.close()
        # self._save_to_json(self.visited)
        
        return self.visited

    def chunk_text(self, text: str) -> list[dict]:
        elements = partition_text(text=text)
        chunks = chunk_by_title(
            elements,
            multipage_sections=True,
            combine_text_under_n_chars=200,
            new_after_n_chars=1000,
        )
        return [{"text": chunk.text.strip()} for chunk in chunks if chunk.text.strip()]

    def chunk_visited_pages(self, visited_pages: dict) -> list[dict]:
        all_chunks = []
        for url, content in visited_pages.items():
            chunks = self.chunk_text(content)
            for chunk in chunks:
                chunk["url"] = url  # attach source
                all_chunks.append(chunk)
        return all_chunks