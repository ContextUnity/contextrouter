"""Web connectors (raw sources).

Per `.cursorrules` ALL web-fetching/searching code belongs under connectors.

- `WebSearchConnector` (key: "web"): Google CSE site-limited search → RetrievedDoc envelopes
- `WebScraperConnector` (key: "web_scraper"): stub for full-page scraping
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from contextrouter.core import (
    BaseConnector,
    BisquitEnvelope,
    get_bool_env,
    get_core_config,
)
from contextrouter.modules.observability import retrieval_span
from contextrouter.modules.retrieval.rag.models import RetrievedDoc

logger = logging.getLogger(__name__)

# Suppress googleapiclient.discovery_cache warnings about oauth2client
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


def _safe_preview(val: object, limit: int = 240) -> str:
    if val is None:
        return ""
    s = val if isinstance(val, str) else str(val)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def _host_for_url(url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").strip().lower().rstrip(".")


def _is_allowed_domain(host: str, allowed_domains: list[str]) -> bool:
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
    if isinstance(link, str) and link.startswith("http"):
        return link
    if isinstance(alt, str) and alt.startswith("http"):
        return alt
    return None


class WebSearchConnector(BaseConnector):
    """Google CSE connector (site-limited) that yields RetrievedDoc envelopes."""

    def __init__(
        self,
        *,
        query: str,
        allowed_domains: list[str],
        max_results_per_domain: int = 10,
        retrieval_queries: list[str] | None = None,
    ) -> None:
        self._query = query
        self._allowed_domains = [
            d.strip() for d in allowed_domains if isinstance(d, str) and d.strip()
        ]
        self._max_results = int(max_results_per_domain)
        self._retrieval_queries = [
            q.strip() for q in (retrieval_queries or []) if isinstance(q, str) and q.strip()
        ]

    async def connect(self) -> AsyncIterator[BisquitEnvelope]:
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
            from langchain_google_community.search import (  # type: ignore[import-not-found]
                GoogleSearchAPIWrapper,
            )

            wrapper = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cx)
            params: dict[str, str] = {"siteSearch": domain, "siteSearchFilter": "i"}
            if english_hint:
                params.update({"hl": "en", "lr": "lang_en"})
            results = wrapper.results(
                query=f"site:{domain} {q}",
                num_results=self._max_results,
                search_params=params,
            )
            return results if isinstance(results, list) else []

        def _run_all_domains(q: str, *, english_hint: bool) -> list[dict[str, object]]:
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
                        res = fut.result()
                        if isinstance(res, list):
                            out.extend(res)
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
                preview = []
                for r in raw[: min(5, len(raw))]:
                    if not isinstance(r, dict):
                        continue
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
                docs: list[RetrievedDoc] = []
                invalid_links: list[object] = []
                rejected: list[tuple[str, str]] = []
                for r in raw_results or []:
                    if not isinstance(r, dict):
                        continue
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
                        "CSE returned results but all were filtered out (invalid_links=%d rejected_domains=%d). "
                        "sample_invalid=%s sample_rejected=%s",
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
                env = BisquitEnvelope(
                    content=d, provenance=[], metadata={"source": "web", "url": d.url}
                )
                env.add_trace("connector:web")
                yield env


class WebScraperConnector(BaseConnector):
    def __init__(self, *, url: str) -> None:
        self._url = url

    async def connect(self) -> AsyncIterator[BisquitEnvelope]:
        raise NotImplementedError(
            "WebScraperConnector is a stub. Implement scraping (e.g. trafilatura)."
        )


__all__ = ["WebSearchConnector", "WebScraperConnector"]
