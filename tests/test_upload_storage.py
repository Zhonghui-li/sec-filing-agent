"""Where an uploaded original is kept, and whether it survives.

On Cloud Run the container filesystem is not storage: it dies with the instance, and this service
scales to zero after about thirteen idle minutes. So an upload written to local disk lasted until
the next recycle, after which the document still ANSWERED questions -- its chunks are in Postgres
-- while the citation link behind the answer 404'd, because the file it pointed at was gone. That
failure is invisible from the outside: the answer looks right, only the evidence is missing.

GCS_BUCKET selects object storage; unset keeps the local folder, which is what a dev run and
every other test want. Both branches are pinned here because the deployed one is the branch no
local run ever executes.
"""
import pytest

pytest.importorskip("fastapi")
import service.app as app                                     # noqa: E402

USER, DOC = "owner@example.com", "a1b2c3d4e5f6"


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_GCS_BUCKET", None)
    monkeypatch.setattr(app, "_UPLOAD_DIR", str(tmp_path))
    return tmp_path


class FakeBlob:
    def __init__(self, store, name): self.store, self.name = store, name
    def upload_from_string(self, data, content_type=None):
        self.store[self.name] = (data, content_type)
    def download_as_bytes(self): return self.store[self.name][0]
    def delete(self): self.store.pop(self.name, None)


class FakeBucket:
    """Only the three calls the storage layer makes. A real client is not needed to pin which
    key a file lands under, and the key is the part that decides whether one user can reach
    another's document."""
    def __init__(self): self.store = {}
    def blob(self, name): return FakeBlob(self.store, name)
    def list_blobs(self, prefix): return [FakeBlob(self.store, n)
                                          for n in sorted(self.store) if n.startswith(prefix)]


@pytest.fixture
def gcs(monkeypatch):
    b = FakeBucket()
    monkeypatch.setattr(app, "_GCS_BUCKET", "a-bucket")
    monkeypatch.setattr(app, "_bucket", lambda: b)
    return b


# --- the local branch ------------------------------------------------------------------

def test_a_local_file_comes_back_byte_identical(local):
    loc = app._store_original(USER, DOC, ".pdf", b"%PDF-1.7 hello")
    assert loc.startswith(str(local))
    assert app._read_original(USER, DOC) == (b"%PDF-1.7 hello", ".pdf")


def test_local_delete_removes_it(local):
    app._store_original(USER, DOC, ".pdf", b"x")
    app._delete_original(USER, DOC)
    assert app._read_original(USER, DOC) is None


def test_reading_something_never_stored_is_none_not_an_error(local):
    assert app._read_original(USER, DOC) is None


# --- the deployed branch ---------------------------------------------------------------

def test_object_storage_keys_the_file_under_its_owner(gcs):
    """The key carries the user id, and /file only ever looks under the CALLER's id -- that is
    what keeps one account's uploads out of another's reach."""
    loc = app._store_original(USER, DOC, ".pdf", b"%PDF-1.7 hello")
    assert loc == f"gs://a-bucket/{USER}/{DOC}.pdf"
    assert list(gcs.store) == [f"{USER}/{DOC}.pdf"]


def test_object_storage_round_trips(gcs):
    app._store_original(USER, DOC, ".pdf", b"%PDF-1.7 hello")
    assert app._read_original(USER, DOC) == (b"%PDF-1.7 hello", ".pdf")


def test_object_storage_delete_removes_it(gcs):
    app._store_original(USER, DOC, ".pdf", b"x")
    app._delete_original(USER, DOC)
    assert app._read_original(USER, DOC) is None
    assert gcs.store == {}


def test_one_users_prefix_cannot_reach_anothers_document(gcs):
    app._store_original("someone@example.com", DOC, ".pdf", b"theirs")
    assert app._read_original(USER, DOC) is None


def test_the_stored_content_type_is_the_real_one(gcs):
    """An .xlsx stored as application/pdf is what the browser is told to render, and it cannot."""
    app._store_original(USER, DOC, ".xlsx", b"PK\x03\x04")
    assert gcs.store[f"{USER}/{DOC}.xlsx"][1].endswith("spreadsheetml.sheet")


# --- what /file sends back -------------------------------------------------------------

@pytest.mark.parametrize("suffix,expected", [
    (".pdf", "application/pdf"),
    (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (".bin", "application/octet-stream"),
])
def test_media_type_follows_the_suffix(suffix, expected):
    assert app._MEDIA.get(suffix, "application/octet-stream") == expected


# --- retention -------------------------------------------------------------------------

def test_retention_can_be_turned_off_without_touching_the_database(monkeypatch):
    """A dev machine and the test suite must not need a reachable database just to boot."""
    monkeypatch.setattr(app, "_RETENTION_DAYS", 0)
    monkeypatch.setattr(app.psycopg, "connect",
                        lambda *a, **k: pytest.fail("retention queried with the window disabled"))
    app._expire_old_documents()


def test_a_failing_retention_sweep_does_not_stop_the_service_booting(monkeypatch, capsys):
    """It runs inside the lifespan. Housekeeping that can refuse to start the service is worse
    than housekeeping that silently skips a night."""
    monkeypatch.setattr(app, "_RETENTION_DAYS", 30)
    def boom(*a, **k): raise OSError("no route to host")
    monkeypatch.setattr(app.psycopg, "connect", boom)
    app._expire_old_documents()
    assert "retention" in capsys.readouterr().out
