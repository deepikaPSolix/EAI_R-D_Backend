import logging
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

def canonicalize_url(url: str) -> str:
    try:
       
        parsed = urlparse(url.strip()) 
        parsed = parsed._replace(fragment="")
        path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
        parsed = parsed._replace(path=path)
        return parsed.geturl()
    except Exception as e:
        logger.error(f"URL canonicalization failed: {str(e)}")
        return url

def is_internal_link(base_url: str, link: str) -> bool:
    return urlparse(link).netloc == urlparse(base_url).netloc