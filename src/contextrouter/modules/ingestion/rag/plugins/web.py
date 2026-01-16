"""Web content ingestion plugin."""

from __future__ import annotations

import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from contextrouter.core import Config
from contextrouter.core.types import StructData

from ..core.plugins import IngestionPlugin
from ..core.prompts import web_summary_prompt
from ..core.registry import register_plugin
from ..core.types import (
    GraphEnrichmentResult,
    IngestionMetadata,
    RawData,
    ShadowRecord,
    WebStructData,
)
from ..core.utils import (
    get_graph_enrichment,
    load_taxonomy_safe,
    normalize_clean_text,
)
from ..settings import RagIngestionConfig
from ..utils.llm import llm_generate
from ..utils.records import generate_id

logger = logging.getLogger(__name__)


def _normalize_text(s: str) -> str:
    # Remove Unicode line/paragraph separators that break some editors/JSONL viewers.
    return (s or "").replace("\u2028", "\n").replace("\u2029", "\n")


@register_plugin("web")
class WebPlugin(IngestionPlugin):
    """Plugin for processing web content."""

    @property
    def source_type(self) -> str:
        return "web"

    def load(self, assets_path: str, config: RagIngestionConfig | None = None) -> list[RawData]:
        """Load web content from multiple supported formats.

        Preferred format: assets/ingestion/source/web/url.toml
        Structure: [{"url": "...", "tags": ["..."], "force_reindex": bool}]

        Also supported: assets/ingestion/source/web/sources.json
        Structure:
        - [{"url": "...", "tags": ["..."], "force_reindex": false}]
        - or {"sources": [...]}

        Fallback: urls.txt (one URL per line) or *.url files
        """
        source_dir = Path(assets_path)
        if not source_dir.exists():
            logger.warning("Web source directory does not exist: %s", assets_path)
            return []

        raw_data: list[RawData] = []
        sources: list[dict[str, Any]] = []

        # Config to get web settings (prefer caller-provided config from preprocess stage)
        if config is None:
            from ..config import load_config

            config = load_config()

        url_file_name = config.web.url_file
        user_agent = self._get_user_agent(config.web)
        crawl_enabled = config.web.crawl_enabled
        crawl_max_pages = config.web.crawl_max_pages
        crawl_max_depth = config.web.crawl_max_depth
        crawl_include_subdomains = config.web.crawl_include_subdomains
        crawl_concurrency = config.web.crawl_concurrency
        timeout_s = config.web.timeout_s
        skip_url_substrings = config.web.crawl_skip_url_substrings

        # Preferred: load from url.toml
        url_file = source_dir / url_file_name
        if url_file.exists():
            try:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[import-not-found]

                with open(url_file, "rb") as f:
                    toml_data = tomllib.load(f)
                    # TOML structure: sources = [{"url": "...", "tags": [...], "force_reindex": bool}]
                    # Or: [[sources]] format
                    if isinstance(toml_data, dict):
                        sources = toml_data.get("sources", [])
                    elif isinstance(toml_data, list):
                        sources = toml_data
                    else:
                        sources = []
                logger.info("Loaded %d web sources from %s", len(sources), url_file)
            except Exception as e:
                logger.warning("Failed to load %s: %s", url_file, e)
                sources = []

        # Also supported: sources.json (explicitly documented)
        if not sources:
            sources_json = source_dir / "sources.json"
            if sources_json.exists():
                try:
                    payload = json.loads(sources_json.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        sources = payload.get("sources", []) or []
                    elif isinstance(payload, list):
                        sources = payload
                    else:
                        sources = []
                    logger.info("Loaded %d web sources from %s", len(sources), sources_json)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", sources_json, e)
                    sources = []

        # Fallback: URL files
        if not sources:
            url_file = source_dir / "urls.txt"
            if url_file.exists():
                urls = [
                    line.strip()
                    for line in url_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                sources = [{"url": url, "tags": [], "force_reindex": False} for url in urls]

            # Also check for individual URL files
            for url_f in source_dir.glob("*.url"):
                url = url_f.read_text().strip()
                if url:
                    sources.append({"url": url, "tags": [], "force_reindex": False})

        # Filter out example/placeholder entries
        sources = [s for s in sources if s.get("url") and "example.com" not in s.get("url", "")]

        for source in sources:
            url = source.get("url", "")
            tags = source.get("tags", [])
            # force_reindex = source.get("force_reindex", False)  # For future caching

            try:
                urls_to_fetch = [url]
                if crawl_enabled:
                    urls_to_fetch = self._crawl_site(
                        seed_url=url,
                        user_agent=user_agent,
                        max_pages=crawl_max_pages,
                        max_depth=crawl_max_depth,
                        include_subdomains=crawl_include_subdomains,
                        concurrency=crawl_concurrency,
                        timeout_s=timeout_s,
                        skip_url_substrings=skip_url_substrings,
                    )

                # Track seen URLs (after redirects) to avoid duplicates
                seen_urls: set[str] = set()

                for page_url in urls_to_fetch:
                    content, title, summary, final_url = self._fetch_web_content(
                        page_url, user_agent=user_agent, timeout_s=timeout_s
                    )
                    if not content:
                        continue

                    # Normalize final URL for deduplication
                    normalized_final_url = self._normalize_url(final_url)
                    if normalized_final_url in seen_urls:
                        logger.debug(
                            "Skipping duplicate URL after redirect: %s -> %s",
                            page_url,
                            normalized_final_url,
                        )
                        continue
                    seen_urls.add(normalized_final_url)

                    # Summary should be generated during preprocess stage (LLM) if missing;
                    # here we only keep extracted meta/OG summary if available.

                    # Normalize ambiguous unicode during preprocess
                    from ..core.utils import normalize_ambiguous_unicode

                    normalized_content = normalize_ambiguous_unicode(content)
                    normalized_title = normalize_ambiguous_unicode(title or final_url)

                    metadata: IngestionMetadata = {
                        "url": normalized_final_url,  # Use normalized final URL (after redirects)
                        "title": normalized_title,
                        "summary": summary or "",
                        "keywords": tags,  # Pass tags as initial keywords
                        "crawl_seed_url": url,
                    }

                    raw_data.append(
                        RawData(
                            content=normalized_content,
                            source_type="web",
                            metadata=metadata,
                        )
                    )

            except Exception as e:
                logger.error("Failed to fetch web content from %s: %s", url, e)
                continue

        return raw_data

    def _crawl_site(
        self,
        *,
        seed_url: str,
        user_agent: str,
        max_pages: int,
        max_depth: int,
        include_subdomains: bool,
        concurrency: int,
        timeout_s: float,
        skip_url_substrings: list[str],
    ) -> list[str]:
        """Bounded crawl starting from a seed URL, returning a list of page URLs.

        - Same-domain only by default
        - Skips obvious non-HTML assets by extension
        """
        seed = self._normalize_url(seed_url)
        if not seed:
            return []

        parsed_seed = urlparse(seed)
        seed_host = parsed_seed.netloc.lower()
        if not seed_host:
            return [seed]

        def host_allowed(url: str) -> bool:
            h = urlparse(url).netloc.lower()
            if not h:
                return False
            if include_subdomains:
                return h == seed_host or h.endswith("." + seed_host)
            return h == seed_host

        visited: set[str] = set()
        out: list[str] = []
        q: deque[tuple[str, int]] = deque([(seed, 0)])

        logger.info(
            "web crawl: seed=%s depth<=%d max_pages=%d concurrency=%d",
            seed_host,
            max_depth,
            max_pages,
            concurrency,
        )

        # Breadth-first crawl by depth; fetch pages in small concurrent batches.
        while q and len(out) < max_pages:
            # Collect a batch of URLs at the same depth.
            batch: list[tuple[str, int]] = []
            while q and len(batch) < max(1, concurrency * 2) and len(out) + len(batch) < max_pages:
                u, d = q.popleft()
                if u in visited:
                    continue
                if d > max_depth:
                    continue
                if not host_allowed(u):
                    continue
                if self._should_skip_url(u, skip_url_substrings=skip_url_substrings):
                    continue
                visited.add(u)
                batch.append((u, d))

            if not batch:
                continue

            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
                futs = {
                    ex.submit(
                        self._download_html,
                        u,
                        user_agent=user_agent,
                        timeout_s=timeout_s,
                    ): (
                        u,
                        d,
                    )
                    for u, d in batch
                }
                for fut in as_completed(futs):
                    u, d = futs[fut]
                    try:
                        html, final_url = fut.result()
                    except Exception as e:
                        # Skip this page; continue crawling.
                        logger.debug("Failed to download %s: %s", u, e)
                        continue
                    if not html.strip():
                        continue
                    # Always add successfully downloaded page to output
                    # Track final URL (after redirects) in visited for deduplication
                    if final_url not in visited:
                        visited.add(final_url)
                    # Only add to output if not already there (handles original URL == final URL case)
                    if final_url not in out:
                        out.append(final_url)
                    if d >= max_depth:
                        continue
                    # Extract links using final URL as base
                    for link in self._extract_links(html, base_url=final_url):
                        link_n = self._normalize_url(link)
                        if not link_n or link_n in visited:
                            continue
                        if not host_allowed(link_n):
                            continue
                        if self._should_skip_url(link_n, skip_url_substrings=skip_url_substrings):
                            continue
                        q.append((link_n, d + 1))

        return out

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for deduplication.

        - Removes fragments (#anchor)
        - Removes trailing slashes (except for root)
        - Converts to lowercase for scheme and host
        - Removes default ports (80 for http, 443 for https)
        """
        url = (url or "").strip()
        if not url:
            return ""

        # Remove fragment
        u, _frag = urldefrag(url)
        if not u:
            return ""

        # Parse and normalize components
        parsed = urlparse(u)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if ":" in netloc:
            host, port = netloc.rsplit(":", 1)
            if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                netloc = host

        # Normalize path: remove trailing slash (except root)
        path = parsed.path.rstrip("/") or "/"

        # Reconstruct URL
        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if parsed.params:
            normalized += f";{parsed.params}"

        return normalized.strip()

    @staticmethod
    def _should_skip_url(url: str, *, skip_url_substrings: list[str]) -> bool:
        path = urlparse(url).path.lower()
        url_l = url.lower()
        for s in skip_url_substrings:
            if isinstance(s, str) and s and s.lower() in url_l:
                return True
        # Skip obvious non-HTML assets
        for ext in (
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".css",
            ".js",
            ".ico",
            ".zip",
            ".mp4",
            ".mp3",
        ):
            if path.endswith(ext):
                return True
        return False

    @staticmethod
    def _extract_links(html: str, *, base_url: str) -> list[str]:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]

            soup = BeautifulSoup(html, "html.parser")
            links: list[str] = []
            for a in soup.find_all("a"):
                href = a.get("href")
                if not isinstance(href, str) or not href.strip():
                    continue
                if href.startswith(("mailto:", "tel:", "javascript:")):
                    continue
                abs_url = urljoin(base_url, href)
                links.append(abs_url)
            return links
        except Exception:
            return []

    @staticmethod
    def _get_user_agent(web_config: Any) -> str:
        """Extract user agent from WebSection config."""
        ua = web_config.user_agent.strip()
        if ua:
            return ua
        # Non-empty default: prevents basic 403/robot blocks for many sites
        return "ContextrouterIngestionBot/1.0 (+https://example.com/bot)"

    @staticmethod
    def _download_html(url: str, *, user_agent: str, timeout_s: float = 20.0) -> tuple[str, str]:
        """Download HTML and return (html_content, final_url_after_redirects).

        Returns:
            Tuple of (html_content, final_url). final_url is the normalized URL after redirects.
        """
        # Validate URL scheme for security - only allow HTTP/HTTPS
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(
                f"Unsupported URL scheme '{parsed_url.scheme}'. Only HTTP and HTTPS are allowed."
            )

        req = Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        final_url = url
        with urlopen(req, timeout=timeout_s) as resp:
            # Get final URL after redirects
            final_url = resp.geturl()
            raw = resp.read()
            try:
                charset = resp.headers.get_content_charset() or "utf-8"
            except Exception:
                charset = "utf-8"
        html = raw.decode(charset, errors="ignore")
        # Normalize final URL for deduplication
        normalized_final_url = WebPlugin._normalize_url(final_url)
        return html, normalized_final_url

    @staticmethod
    def _extract_basic_metadata(html: str, url: str) -> tuple[str, str]:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]

            soup = BeautifulSoup(html, "html.parser")
            title = ""
            if soup.title and soup.title.string:
                title = str(soup.title.string).strip()

            summary = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                summary = str(meta_desc.get("content")).strip()
            if not summary:
                meta_og = soup.find("meta", attrs={"property": "og:description"})
                if meta_og and meta_og.get("content"):
                    summary = str(meta_og.get("content")).strip()

            if not title:
                title = urlparse(url).path or url
            return title, summary
        except Exception:
            return urlparse(url).path or url, ""

    def _fetch_web_content(
        self, url: str, *, user_agent: str, timeout_s: float
    ) -> tuple[str, str, str, str]:
        """Fetch and clean web content, always sending a User-Agent.

        Returns:
            Tuple of (content, title, summary, final_url_after_redirects)
        """
        try:
            html, final_url = self._download_html(url, user_agent=user_agent, timeout_s=timeout_s)
        except Exception as e:
            # Treat as a normal skip (timeouts/blocked pages are common during crawling).
            logger.warning("HTTP fetch skipped for %s: %s", url, e)
            return "", "", "", url

        title, summary = self._extract_basic_metadata(html, final_url)

        # Prefer trafilatura for boilerplate removal when available
        try:
            import trafilatura  # type: ignore[import-not-found]

            content = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
            if content.strip():
                return (
                    _normalize_text(content),
                    _normalize_text(title),
                    _normalize_text(summary),
                    final_url,
                )
        except Exception as e:
            logger.warning("trafilatura extract failed for %s: %s", final_url, e)

        # Fallback: BeautifulSoup text extraction
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            content = soup.get_text(" ", strip=True)
            return (
                _normalize_text(content),
                _normalize_text(title),
                _normalize_text(summary),
                final_url,
            )
        except Exception as e:
            logger.error("HTML text extraction failed for %s: %s", final_url, e)
            return "", "", "", final_url

    def transform(
        self,
        data: list[RawData],
        enrichment_func: Callable[[str], GraphEnrichmentResult],
        taxonomy_path: Path | None = None,
        config: RagIngestionConfig | None = None,
        core_cfg: Config | None = None,
        **kwargs: object,
    ) -> list[ShadowRecord]:
        """Transform web data using standard text splitting with taxonomy keywords.

        Process:
        1. Chunk by paragraphs (~1000 chars)
        2. Identify 3 main taxonomy keywords per chunk
        3. Generate taxonomy-aligned summary if missing
        4. Create ShadowRecords with enriched input_text
        """
        shadow_records: list[ShadowRecord] = []

        _ = kwargs

        # Load taxonomy for keyword identification
        taxonomy = load_taxonomy_safe(taxonomy_path)

        for raw in data:
            url = raw.metadata.get("url", "")
            title = raw.metadata.get("title", "Web Content")
            base_summary = raw.metadata.get("summary", "")
            initial_keywords = raw.metadata.get("keywords", [])

            # Chunk by paragraphs (~1000 chars)
            paragraphs = [p.strip() for p in raw.content.split("\n\n") if p.strip()]
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > 1000 and current_chunk:
                    record = self._create_web_record(
                        current_chunk,
                        url,
                        title,
                        base_summary,
                        initial_keywords,
                        enrichment_func,
                        taxonomy,
                        config=config,
                        core_cfg=core_cfg,
                    )
                    shadow_records.append(record)
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            # Handle remaining chunk
            if current_chunk:
                record = self._create_web_record(
                    current_chunk,
                    url,
                    title,
                    base_summary,
                    initial_keywords,
                    enrichment_func,
                    taxonomy,
                    config=config,
                    core_cfg=core_cfg,
                )
                shadow_records.append(record)

        return shadow_records

    def _create_web_record(
        self,
        chunk: str,
        url: str,
        title: str,
        base_summary: str,
        initial_keywords: list[str],
        enrichment_func: Callable[[str], GraphEnrichmentResult],
        taxonomy: StructData | None,
        *,
        config: RagIngestionConfig | None,
        core_cfg: Config | None,
    ) -> ShadowRecord:
        """Create a ShadowRecord for a web content chunk."""
        # Graph enrichment
        graph_keywords, enrichment_summary, parent_categories = get_graph_enrichment(
            text=chunk, enrichment_func=enrichment_func
        )

        # Identify taxonomy keywords
        taxonomy_keywords = self._identify_taxonomy_keywords(chunk, taxonomy)

        # Combine all keywords (initial + graph + taxonomy), deduplicated
        all_keywords = list(dict.fromkeys(initial_keywords + taxonomy_keywords + graph_keywords))[
            :10
        ]

        # Generate summary if missing
        summary = base_summary
        if not summary and len(chunk) > 200:
            summary = self._generate_summary(chunk, taxonomy, core_cfg=core_cfg)

        # Build enriched input_text with QA-style explicit format
        input_text = self._build_input_text(
            content=chunk,
            keywords=all_keywords,
            summary=enrichment_summary,
            parent_categories=parent_categories,
        )

        # Clean quote text
        clean_chunk = normalize_clean_text(chunk)

        record_id = generate_id(url, chunk[:50])

        struct_data: WebStructData = {
            "source_type": "web",
            "title": title,
            "url": url,
            "summary": summary,
            "quote": clean_chunk,
        }

        return ShadowRecord(
            id=record_id,
            input_text=input_text,
            struct_data=struct_data,
            title=title,
            source_type="web",
        )

    def _identify_taxonomy_keywords(
        self,
        content: str,
        taxonomy: StructData | None,
        max_keywords: int = 3,
    ) -> list[str]:
        """Identify top taxonomy keywords from content.

        Uses simple keyword matching for efficiency.
        Falls back to LLM only if no matches found.
        """
        if not taxonomy:
            return []

        all_keywords = taxonomy.get("all_keywords", [])
        if not all_keywords:
            return []

        # Simple case-insensitive matching
        content_lower = content.lower()
        matches: list[tuple[str, int]] = []

        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            if count > 0:
                matches.append((keyword, count))

        # Sort by frequency, return top N
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:max_keywords]]

    def _generate_summary(
        self,
        content: str,
        taxonomy: dict[str, Any] | None,
        *,
        core_cfg: Config | None,
    ) -> str:
        """Generate taxonomy-aligned summary using LLM."""
        if len(content) < 200:
            return ""
        if core_cfg is None:
            raise ValueError(
                "WebPlugin summary generation requires core_cfg (contextrouter.core.config.Config)"
            )

        try:
            categories = None
            if taxonomy and isinstance(taxonomy.get("categories"), dict):
                categories = [
                    str(c).replace("_", " ").title()
                    for c in list(taxonomy.get("categories", {}).keys())[:10]
                ]
            prompt = web_summary_prompt(content=content[:3000], categories=categories)

            result = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=core_cfg.models.ingestion.preprocess.model,
                max_tokens=256,
                temperature=0.3,
                parse_json=False,
            )

            if isinstance(result, str):
                return result.strip()[:500]
            return ""

        except Exception as e:
            logger.debug("Summary generation failed: %s", e)
            return ""

    def _build_input_text(
        self,
        content: str,
        keywords: list[str],
        summary: str | None = None,
        parent_categories: list[str] | None = None,
    ) -> str:
        """Build shadow context input_text with QA-style explicit enrichment format.

        Args:
            content: Web chunk content
            keywords: Combined keywords (initial + graph + taxonomy)
            summary: Enrichment summary (graph relations or generated)
            parent_categories: Taxonomy categories from graph enrichment

        Returns:
            Formatted input_text string with explicit Categories: and Additional Knowledge: headers
        """
        parts = [content]

        # Add taxonomy categories from graph enrichment (QA-style)
        if parent_categories:
            cats = [c for c in parent_categories if isinstance(c, str) and c.strip()]
            if cats:
                cat_str = ", ".join(cats[:5])
                parts.append(f"Categories: {cat_str}")

        # Add natural language enrichment for keywords (QA-style)
        if keywords:
            top_keywords = keywords[:10]
            if len(top_keywords) == 1:
                parts.append(f"Additional Knowledge: This text is related to {top_keywords[0]}.")
            elif len(top_keywords) <= 3:
                keywords_str = ", ".join(top_keywords[:-1])
                parts.append(
                    f"Additional Knowledge: This text is related to {keywords_str} and {top_keywords[-1]}."
                )
            else:
                keywords_str = ", ".join(top_keywords[:5])
                parts.append(
                    f"Additional Knowledge: This text is related to {keywords_str}, and other concepts."
                )

        # Add summary if available (graph relations or generated)
        if isinstance(summary, str) and summary.strip():
            parts.append(f"Additional Knowledge: {summary.strip()}")

        return "\n".join(parts)
