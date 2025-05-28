# app/graphbuilder.py
import os
import asyncio
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_dispatcher import SemaphoreDispatcher
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, concurrency: int = 10):
        self.visited = {}  
        self.edges   = []
        self.concurrency = concurrency

    async def crawl_website(self, start_url: str, max_depth: int = 1):
        concurrency: int = 20
        strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth,            
        include_external=False          
    )

        run_cfg = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=False,
            semaphore_count=concurrency     
        )

        dispatcher = SemaphoreDispatcher(concurrency)

        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun(
                url=start_url,
                config=run_cfg,
                dispatcher=dispatcher
            )

        output = []
        for result in results:
            if not result.success:
                continue
            html = result.cleaned_html or result.html
            text = BeautifulSoup(html, "html.parser") \
                .get_text(separator=" ", strip=True)
            output.append({"url": result.url, "text": text})

        return output
    def chunk_text(self, text: str) -> list[dict]:
        elements = partition_text(text=text)
        total    = sum(len(el.text or "") for el in elements)

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
        return [
            {"text": c.text.strip(), "url": ""} 
            for c in chunks
            if c.text and len(c.text.strip()) > 50
        ]

    def chunk_visited_pages(self, visited_pages: list[dict]) -> list[dict]:
        all_chunks = []
        for page in visited_pages:
            url, text = page["url"], page["text"]
            for c in self.chunk_text(text):
                c["url"] = url
                all_chunks.append(c)
            # 2) if you had anchors, add them here …
        # logger.info(f"Generated total {len(all_chunks)} chunks")
        return all_chunks
