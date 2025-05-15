# app/graphbuilder.py

import asyncio
import aiohttp
import random
import logging
from flask import current_app
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urljoin
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from app.canon import canonicalize_url, is_internal_link

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        self.visited = {}  # maps url -> {"text":..., "anchors":[{"text","url"},...]}
        self.edges   = []

    async def _scrape_website(self, url: str, base_url: str):
        url = url.strip()
        if url in self.visited:
            return []

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with self.session.get(url, headers=headers, timeout=10) as response:
                status = response.status
                current_app.logger.info(f"Fetched {url} → HTTP {status}")
                if status != 200:
                    current_app.logger.warning(f"Non-200 status for {url}, using Playwright fallback")
                    return await self._playwright_fallback(url, base_url)

                html = await response.read()
                html = html.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html, "html.parser")
                logger.info(f"Scrappinggggggg")
                # 1) Full page text
                page_text = soup.get_text(separator="\n", strip=True)

                # 2) Extract all <a> anchors as mini-chunks
                anchors = []
                for a in soup.find_all("a", href=True):
                    href = canonicalize_url(urljoin(url, a["href"]))
                    txt  = a.get_text(separator=" ", strip=True)
                    if txt:
                        anchors.append({"text": txt, "url": href})

                # 3) Internal links for crawl graph
                all_links = [a["url"] for a in anchors]
                internal_links = [
                    l for l in dict.fromkeys(all_links)
                    if is_internal_link(base_url, l)
                ]

                # 4) Store into visited
                self.visited[url] = {
                    "text":    page_text,
                    "anchors": anchors
                }
                logger.info(f"Scraped {url}: {len(page_text)} chars, {len(anchors)} anchors")
                return internal_links

        except Exception as e:
            logger.warning(f"aiohttp scrape failed for {url}: {e}")
            return await self._playwright_fallback(url, base_url)

    async def _playwright_fallback(self, url: str, base_url: str):
        url = url.strip()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page    = await browser.new_page(ignore_https_errors=True)
                await page.goto(url, timeout=20000)
                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")

                # replicate same logic
                page_text = soup.get_text(separator="\n", strip=True)
                anchors   = []
                for a in soup.find_all("a", href=True):
                    href = canonicalize_url(urljoin(url, a["href"]))
                    txt  = a.get_text(separator=" ", strip=True)
                    if txt:
                        anchors.append({"text":txt, "url":href})

                all_links = [a["url"] for a in anchors]
                internal_links = [
                    l for l in dict.fromkeys(all_links)
                    if is_internal_link(base_url, l)
                ]

                self.visited[url] = {"text":page_text, "anchors":anchors}
                logger.info(f"Playwright fallback scraped {url}: {len(page_text)} chars, {len(anchors)} anchors")
                await browser.close()
                return internal_links

        except Exception as e:
            logger.error(f"Playwright failed for {url}: {e}")
            return []

    async def crawl_website(self, start_url: str, max_depth: int = 1):
        self.visited = {}
        self.edges   = []
        connector    = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(connector=connector)

        start_url = start_url.strip()
        to_crawl   = [(start_url, 0)]
        logger.info(f"Starting crawl at {start_url} to depth {max_depth}")

        while to_crawl:
            current_url, depth = to_crawl.pop(0)
            if depth > max_depth:
                continue

            internal = await self._scrape_website(current_url, start_url)
            for link in internal:
                self.edges.append((current_url, link))
                if link not in self.visited and all(link != u for u, _ in to_crawl):
                    to_crawl.append((link, depth+1))
            await asyncio.sleep(random.uniform(0.5, 1.5))

        await self.session.close()
        logger.info(f"Crawling done: visited {len(self.visited)} pages, {len(self.edges)} edges")
        return self.visited

    def chunk_text(self, text: str):
        elements = partition_text(text=text)
        total    = sum(len(el.text or "") for el in elements)

        # dynamic chunk sizing
        if total <= 50_000:
            max_chars = 600
        elif total <= 200_000:
            max_chars = 1000
        else:
            max_chars = 1500

        chunks = chunk_by_title(
            elements,
            multipage_sections=True,
            combine_text_under_n_chars=50,
            new_after_n_chars=max_chars
        )

        clean = [
            {"text": c.text.strip()}
            for c in chunks
            if c.text and len(c.text.strip()) > 50
        ]
        logger.debug(f"Chunked text into {len(clean)} pieces")
        return clean

    def chunk_visited_pages(self, visited_pages: dict):
        all_chunks = []
        for url, payload in visited_pages.items():
            # 1) large text chunks (carry page URL)
            text_chunks = self.chunk_text(payload["text"])
            for c in text_chunks:
                c["url"]       = url
                c["file_name"] = ""
                all_chunks.append(c)

            # 2) anchor chunks (carry their own href)
            for anc in payload.get("anchors", []):
                all_chunks.append({
                    "text":      anc["text"].strip(),
                    "url":       anc["url"],
                    "file_name": ""
                })

        logger.info(f"Generated total {len(all_chunks)} chunks from pages+anchors")
        return all_chunks
