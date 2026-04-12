import pytest


@pytest.fixture(autouse=True)
def _disable_langfuse_telemetry(monkeypatch):
    """Globally disable Langfuse tracing during tests to avoid accidental reporting."""
    import contextunity.router.modules.observability.langfuse as lf

    monkeypatch.setattr(lf, "_enabled", lambda *args, **kwargs: False)
