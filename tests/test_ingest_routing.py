"""Which parser runs, and who records the outcome.

There used to be one path -- a subprocess against ingest/.venv -- and it only ever worked on a
dev machine. The image never contained ingest/, and ingest/.venv is gitignored so it could not
have been copied in, so every upload to the deployed service failed on a missing interpreter.
That went unnoticed for months because local and deployed share one database: documents parsed
locally appeared in the hosted demo's list and answered questions there, which is exactly what a
working upload looks like from the outside.

So the two environments now have two explicit paths, and INGEST_URL is what chooses. These pin
the choice, and the division of labour that comes with it: when a remote parser runs, the status
row is ITS to write, because /upload has already returned and this instance may be recycled long
before the parse ends.
"""
import pytest

pytest.importorskip("fastapi")
import service.app as app                                       # noqa: E402

USER, DOC, NAME = "owner@example.com", "a1b2c3d4e5f6", "filing.pdf"


@pytest.fixture
def status_writes(monkeypatch):
    seen = []
    monkeypatch.setattr(app, "_doc_set_status",
                        lambda u, d, s, **kw: seen.append((u, d, s, kw)))
    monkeypatch.setattr(app, "_user_agents", {(USER, None): object()})
    return seen


def test_with_a_parser_service_the_document_is_sent_to_it(monkeypatch, status_writes):
    calls = []
    monkeypatch.setattr(app, "_INGEST_URL", "https://parser.example")
    monkeypatch.setattr(app, "_call_parser", lambda *a: calls.append(a))
    app._ingest_async(USER, DOC, NAME, "gs://bucket/owner@example.com/a1b2c3d4e5f6.pdf")
    assert calls == [(USER, DOC, NAME, "gs://bucket/owner@example.com/a1b2c3d4e5f6.pdf")]


def test_the_remote_parser_owns_the_status_row(monkeypatch, status_writes):
    """Nothing is written here on success. /upload has returned, so this instance may be gone
    before the parse ends -- a row this side promised to update would stay 'processing' forever."""
    monkeypatch.setattr(app, "_INGEST_URL", "https://parser.example")
    monkeypatch.setattr(app, "_call_parser", lambda *a: None)
    app._ingest_async(USER, DOC, NAME, "gs://bucket/x.pdf")
    assert status_writes == []


def test_a_parser_that_cannot_be_reached_is_recorded_as_failed(monkeypatch, status_writes):
    """The one case this side still owns: if the call never landed, nobody else will write it."""
    monkeypatch.setattr(app, "_INGEST_URL", "https://parser.example")
    def boom(*a): raise ConnectionError("refused")
    monkeypatch.setattr(app, "_call_parser", boom)
    app._ingest_async(USER, DOC, NAME, "gs://bucket/x.pdf")
    assert [(s, k) for _, _, s, k in status_writes] == [("failed", {"error": "ConnectionError: refused"})]


def test_a_successful_remote_parse_drops_this_instances_agent_cache(monkeypatch, status_writes):
    """The cached agent was built without the new document; it would answer as if nothing was
    uploaded, which reads as the upload having silently done nothing."""
    monkeypatch.setattr(app, "_INGEST_URL", "https://parser.example")
    monkeypatch.setattr(app, "_call_parser", lambda *a: None)
    app._ingest_async(USER, DOC, NAME, "gs://bucket/x.pdf")
    assert app._user_agents == {}


def test_without_a_parser_service_the_local_venv_runs(monkeypatch, status_writes):
    """A dev machine has ingest/.venv and no parser service, and must keep working unchanged."""
    monkeypatch.setattr(app, "_INGEST_URL", None)
    ran = {}
    class Done:
        returncode, stdout, stderr = 0, 'PARSE_QUALITY={"status":"ok"}\n', ""
    def fake_run(argv, **kw):
        ran["argv"] = argv
        return Done()
    monkeypatch.setattr(app.subprocess, "run", fake_run)
    app._ingest_async(USER, DOC, NAME, "/tmp/uploads/owner/a1b2c3d4e5f6.pdf")
    assert ran["argv"][0] == app._INGEST_PY
    assert "--file" in ran["argv"]
    assert ran["argv"][ran["argv"].index("--file") + 1] == "/tmp/uploads/owner/a1b2c3d4e5f6.pdf"
    assert [(s, k) for _, _, s, k in status_writes] == [("ready", {"parse_quality": '{"status":"ok"}'})]


def test_a_failing_local_parse_is_recorded_with_its_stderr(monkeypatch, status_writes):
    monkeypatch.setattr(app, "_INGEST_URL", None)
    class Failed:
        returncode, stdout, stderr = 1, "", "docling exploded"
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: Failed())
    app._ingest_async(USER, DOC, NAME, "/tmp/x.pdf")
    assert [(s, k) for _, _, s, k in status_writes] == [("failed", {"error": "docling exploded"})]
