from __future__ import annotations

from contextrouter.modules.ingestion.rag import normalize_clean_text


def test_normalize_clean_text_unescapes_html_entities() -> None:
    s = "it&#39;s filled\nwith &quot;stuff&quot;  "
    assert normalize_clean_text(s) == 'it\'s filled with "stuff"'
