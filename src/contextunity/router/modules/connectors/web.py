"""Web connectors (raw sources).
Per `.cursorrules` ALL web-fetching/searching code belongs under connectors.
- `WebSearchConnector` (key: "web"): Google CSE site-limited search → RetrievedDoc units
- `WebScraperConnector` (key: "web_scraper"): stub for full-page scraping
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import AsyncIterator
from typing import Protocol, override, runtime_checkable

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list

from contextunity.router.core import (
    BaseConnector,
    get_bool_env,
    get_core_config,
)
from contextunity.router.modules.observability import retrieval_span
from contextunity.router.modules.retrieval.rag.models import RetrievedDoc

logger = get_contextunit_logger(__name__)

# Suppress googleapiclient.discovery_cache warnings about oauth2client
get_contextunit_logger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


@runtime_checkable
class _GoogleSearchWrapper(Protocol):
    """Runtime surface used from the Google CSE search wrapper."""

    def results(
        self,
        *,
        query: str,
        num_results: int,
        search_params: dict[str, str] | None = None,
    ) -> object:
        """Execute a scoped Google CSE query."""
        ...


@runtime_checkable
class _GoogleSearchWrapperFactory(Protocol):
    """Constructor surface for the Google CSE search wrapper."""

    def __call__(self, *, google_api_key: str, google_cse_id: str) -> _GoogleSearchWrapper:
        """Build a configured Google CSE wrapper."""
        ...


def _async_iterator_marker() -> bool:
    """Return ``False`` while preserving async-generator typing for stubs."""
    return False


def _cse_result_rows(raw: object) -> list[dict[str, object]]:
    """Narrow untyped CSE wrapper output to object dict rows."""
    if not is_object_list(raw):
        return []
    rows: list[dict[str, object]] = []
    for item in raw:
        if is_object_dict(item):
            rows.append(item)
    return rows


def _safe_preview(val: object, limit: int = 240) -> str:
    """Collapse whitespace and truncate *val* to *limit* chars, appending ‘…’ if clipped."""
    if val is None:
        return ""
    s = val if isinstance(val, str) else str(val)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def _host_for_url(url: str) -> str:
    """Extract the lowercase hostname from *url* (trailing dot stripped)."""
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").strip().lower().rstrip(".")


def _is_allowed_domain(host: str, allowed_domains: list[str]) -> bool:
    """Return ``True`` if *host* exactly matches or is a subdomain of any entry in *allowed_domains*."""
    if not host:
        return False
    for d in allowed_domains:
        dd = d.strip().lower().rstrip(".")
        if not dd:
            continue
        if host == dd or host.endswith("." + dd):
            return True
    return False


def _normalize_http_url(link: object, alt: object) -> str | None:
    """Return the first of *link* / *alt* that begins with ``http``, or ``None``."""
    if isinstance(link, str) and link.startswith("http"):
        return link
    if isinstance(alt, str) and alt.startswith("http"):
        return alt
    return None


class WebSearchConnector(BaseConnector):
    """Google CSE connector (site-limited) that yields RetrievedDoc units."""

    def __init__(
        self,
        *,
        query: str,
        allowed_domains: list[str],
        max_results_per_domain: int = 10,
        retrieval_queries: list[str] | None = None,
    ) -> None:
        """Store *query*, domain whitelist, per-domain result cap, and optional English retrieval queries."""
        self._query: str = query
        self._allowed_domains: list[str] = [d.strip() for d in allowed_domains if d.strip()]
        self._max_results: int = int(max_results_per_domain)
        self._retrieval_queries: list[str] = [
            q.strip() for q in (retrieval_queries or []) if q.strip()
        ]

    @override
    def connect(self) -> AsyncIterator[ContextUnit]:
        return self._connect()

    async def _connect(self) -> AsyncIterator[ContextUnit]:
        """Run Google CSE searches across allowed domains and yield results as ``ContextUnit`` objects."""
        if not self._query.strip():
            return
        if not self._allowed_domains:
            return

        cfg = get_core_config()
        if not cfg.google_cse.enabled:
            return

        api_key = cfg.google_cse.api_key
        cx = cfg.google_cse.cx
        if not api_key or not cx:
            return

        def _run_for_domain(domain: str, q: str, *, english_hint: bool) -> list[dict[str, object]]:
            """Execute a Google CSE query scoped to a single *domain* via ``GoogleSearchAPIWrapper``."""
            search_mod = importlib.import_module("langchain_google_community.search")
            wrapper_factory = getattr(search_mod, "GoogleSearchAPIWrapper", None)
            if not isinstance(wrapper_factory, _GoogleSearchWrapperFactory):
                return []

            wrapper = wrapper_factory(google_api_key=api_key, google_cse_id=cx)
            params: dict[str, str] = {"siteSearch": domain, "siteSearchFilter": "i"}
            if english_hint:
                params.update({"hl": "en", "lr": "lang_en"})
            return _cse_result_rows(
                wrapper.results(
                    query=f"site:{domain} {q}",
                    num_results=self._max_results,
                    search_params=params,
                )
            )

        def _run_all_domains(q: str, *, english_hint: bool) -> list[dict[str, object]]:
            """Fan out ``_run_for_domain`` across allowed domains using a thread pool (max 6 workers)."""
            import concurrent.futures

            if len(self._allowed_domains) == 1:
                return _run_for_domain(self._allowed_domains[0], q, english_hint=english_hint)

            max_workers = min(len(self._allowed_domains), 6)
            out: list[dict[str, object]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(_run_for_domain, d, q, english_hint=english_hint)
                    for d in self._allowed_domains
                ]
                for fut in futs:
                    try:
                        out.extend(fut.result())
                    except Exception as e:
                        logger.warning("CSE query failed for one domain: %s", e)
            return out

        with retrieval_span(
            name="cse_search",
            input_data={"query": self._query, "domains": self._allowed_domains},
        ) as _span:
            t0 = time.perf_counter()
            raw = _run_all_domains(self._query, english_hint=False)
            debug_web = bool(get_bool_env("DEBUG_WEB_SEARCH"))
            if debug_web and raw:
                preview: list[dict[str, str]] = []
                for r in raw[: min(5, len(raw))]:
                    link = _normalize_http_url(r.get("link"), r.get("url") or r.get("formattedUrl"))
                    preview.append(
                        {
                            "title": _safe_preview(r.get("title"), 120),
                            "url": _safe_preview(link, 160),
                            "snippet": _safe_preview(r.get("snippet") or r.get("content"), 120),
                        }
                    )
                logger.info("DEBUG_WEB_SEARCH: raw CSE preview=%s", preview)

            def _extract_docs(raw_results: list[dict[str, object]]) -> list[RetrievedDoc]:
                """Validate raw CSE results: normalise URLs, enforce domain whitelist, build ``RetrievedDoc`` list."""
                docs: list[RetrievedDoc] = []
                invalid_links: list[object] = []
                rejected: list[tuple[str, str]] = []
                for r in raw_results or []:
                    if "Result" in r and len(r.keys()) == 1:
                        continue
                    link = _normalize_http_url(r.get("link"), r.get("url") or r.get("formattedUrl"))
                    if not link:
                        invalid_links.append(r.get("link") or r.get("url") or r.get("formattedUrl"))
                        continue
                    host = _host_for_url(link)
                    if not _is_allowed_domain(host, self._allowed_domains):
                        rejected.append((host, link))
                        continue
                    title = r.get("title")
                    snippet = r.get("snippet") or r.get("content") or ""
                    docs.append(
                        RetrievedDoc(
                            source_type="web",
                            title=title if isinstance(title, str) else link,
                            url=link,
                            content=str(snippet),
                        )
                    )
                # Filter-to-zero diagnostics (always on).
                if raw_results and not docs:
                    logger.warning(
                        (
                            "CSE returned results but all were filtered out "
                            "(invalid_links=%d rejected_domains=%d). "
                            "sample_invalid=%s sample_rejected=%s"
                        ),
                        len(invalid_links),
                        len(rejected),
                        [_safe_preview(x, 140) for x in invalid_links[:3]],
                        [{"host": h, "url": u} for (h, u) in rejected[:3]],
                    )
                return docs

            docs = _extract_docs(raw)
            if (
                not docs
                and self._retrieval_queries
                and self._retrieval_queries[0].lower() != self._query.lower()
            ):
                english_q = self._retrieval_queries[0]
                raw = _run_all_domains(english_q, english_hint=True)
                docs = _extract_docs(raw)

            logger.info(
                "CSE web connector completed (docs=%d elapsed=%.1fms)",
                len(docs),
                (time.perf_counter() - t0) * 1000,
            )
            for d in docs:
                unit = ContextUnit(
                    payload={"content": d, "url": d.url},
                    provenance=["connector:web"],
                    modality="text",
                )
                yield unit


class WebScraperConnector(BaseConnector):
    """Stub connector for full-page scraping — subclass and implement ``connect()``."""

    def __init__(self, *, url: str) -> None:
        """Store the target *url* to scrape."""
        self._url: str = url

    @override
    def connect(self) -> AsyncIterator[ContextUnit]:
        return self._connect()

    async def _connect(self) -> AsyncIterator[ContextUnit]:
        """Not implemented — subclasses must provide scraping logic (e.g. trafilatura)."""
        if _async_iterator_marker():
            yield ContextUnit(payload={}, provenance=["connector:web_scraper"], modality="text")
        raise NotImplementedError(
            "WebScraperConnector is a stub. Implement scraping (e.g. trafilatura)."
        )


__all__ = ["WebSearchConnector", "WebScraperConnector"]
