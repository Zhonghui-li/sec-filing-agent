"""The identity contract, now carried by a cookie instead of a bearer token in every request body.

Why the move: the citation chip for an uploaded document is an <a href> the browser NAVIGATES to,
and a browser navigation cannot carry a custom header, so the credential had to travel in the
query string and therefore into request logs. Cookies are sent on navigation, so the credential
leaves the URL entirely. The cost of the move is CSRF: a cookie is attached automatically, so
another origin can make the browser issue a request that looks authenticated. SameSite=Lax is
what pays that back, and it is pinned here because nothing else in the system would notice if it
silently stopped being set.

One cookie, two shapes, which is what keeps the id space partitioned:

    an opaque 43-char token   a row in `sessions` maps it to a verified email
    anon_<22 chars>           no server state at all; anonymous visitors own no documents

Anything else carries no identity. That is the check that stops `vsid=someone@example.com` from
scoping a question to that account's private documents.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import HTTPException                              # noqa: E402
from fastapi.testclient import TestClient                       # noqa: E402

import service.app as app                                       # noqa: E402

ANON = "anon_" + "a" * 22
SID = "s" * 43                 # the shape secrets.token_urlsafe(32) produces
EMAIL = "owner@example.com"


class Req:
    """What the identity path reads off a request. A real Request needs an ASGI scope, so the
    cookie-shape rules are exercised against this instead; anything that depends on headers or
    on the proxy in front goes through TestClient below, where a stub cannot quietly diverge."""
    def __init__(self, cookies=None, scheme="https", headers=None):
        self.cookies = cookies or {}
        self.headers = headers or {}
        self.url = type("U", (), {"scheme": scheme})()


@pytest.fixture
def signed_in_world(monkeypatch):
    """Sign-in configured, and exactly one live session. The DB is never touched."""
    monkeypatch.setattr(app, "_GOOGLE_CLIENT_ID", "client-id.apps.googleusercontent.com")
    monkeypatch.setattr(app, "_session_user", lambda sid: EMAIL if sid == SID else None)
    monkeypatch.setattr(app, "_session_create", lambda email: SID)
    monkeypatch.setattr(app, "_session_destroy", lambda sid: None)
    monkeypatch.setattr(app, "_verify_google",
                        lambda tok: EMAIL if tok == "good-token" else None)


# --- what the id may be -----------------------------------------------------------------

def test_a_live_session_cookie_resolves_to_the_verified_email(signed_in_world):
    assert app._user_id(Req({"vsid": SID})) == EMAIL


def test_an_anonymous_cookie_can_ask_but_not_reach_private_data(signed_in_world):
    assert app._user_id(Req({"vsid": ANON}), require=False) == ANON
    with pytest.raises(HTTPException) as e:
        app._user_id(Req({"vsid": ANON}))
    assert e.value.status_code == 401


@pytest.mark.parametrize("claimed", [
    EMAIL,                        # the bypass: name an account and inherit its documents
    "anon_short",
    "anon_" + "a" * 21,           # one char under
    "anon_" + "a" * 23,           # one char over
    "anon_" + "!" * 22,           # outside the character class
    "xanon_" + "a" * 22,          # fullmatch, not search
    ANON + " ",
    "",
])
def test_a_cookie_of_any_other_shape_carries_no_identity(signed_in_world, claimed):
    """It is not an error -- an unrecognised cookie is simply not an identity, so the caller is
    treated as a brand-new anonymous visitor and gets a fresh id the SERVER chose."""
    got = app._user_id(Req({"vsid": claimed}), require=False)
    assert got != claimed
    assert app._ANON_ID.fullmatch(got)


def test_a_missing_cookie_mints_a_new_anonymous_identity(signed_in_world):
    got = app._user_id(Req(), require=False)
    assert app._ANON_ID.fullmatch(got)
    assert app._user_id(Req(), require=False) != got        # a fresh one each time


def test_minting_sets_the_cookie_so_the_next_request_keeps_the_same_id(signed_in_world):
    class Resp:
        def __init__(self): self.kw = None
        def set_cookie(self, **kw): self.kw = kw
    r = Resp()
    got = app._user_id(Req(), response=r, require=False)
    assert r.kw["key"] == "vsid" and r.kw["value"] == got


def test_private_endpoints_demand_a_session_by_default(signed_in_world):
    """require=True is the default, so a newly added endpoint is private unless it opts out."""
    with pytest.raises(HTTPException) as e:
        app._user_id(Req())
    assert e.value.status_code == 401


def test_a_destroyed_session_stops_working(signed_in_world, monkeypatch):
    """What the bearer token could not do: revoke. Signing out deletes the row, so the cookie is
    dead on the next request instead of living until its own expiry."""
    monkeypatch.setattr(app, "_session_user", lambda sid: None)
    with pytest.raises(HTTPException):
        app._user_id(Req({"vsid": SID}))


def test_with_sign_in_unconfigured_anonymous_is_a_full_identity(monkeypatch):
    """Local dev and CI run with no client id. That path must not start demanding sessions."""
    monkeypatch.setattr(app, "_GOOGLE_CLIENT_ID", "")
    monkeypatch.setattr(app, "_session_user", lambda sid: None)
    assert app._user_id(Req({"vsid": ANON})) == ANON


# --- the cookie's own attributes, over real HTTP ----------------------------------------

@pytest.fixture
def client(signed_in_world):
    return TestClient(app.app)                 # no `with`: lifespan (agent + DB) must not run


def test_login_sets_an_httponly_samesite_lax_cookie(client):
    # https explicitly: TestClient's base_url is http://testserver, and `secure` is derived from
    # the scheme, so asking for these three attributes over http would assert the wrong thing.
    r = client.post("https://testserver/auth/login", json={"token": "good-token"})
    assert r.status_code == 200 and r.json()["email"] == EMAIL
    sc = r.headers["set-cookie"].lower()
    # HttpOnly: JS cannot read it, so an XSS cannot exfiltrate the session the way it could read
    # localStorage. SameSite=Lax: not attached to a cross-site POST, which is the CSRF that
    # cookies introduce and bearer tokens did not have -- but still attached on a top-level GET
    # navigation, which is exactly what makes the /file citation link work without a credential
    # in the URL. Secure: HTTPS only.
    assert "httponly" in sc
    assert "samesite=lax" in sc
    assert "secure" in sc


def test_login_rejects_an_unverifiable_token(client):
    assert client.post("/auth/login", json={"token": "forged"}).status_code == 401


def test_logout_clears_the_cookie(client):
    r = client.post("/auth/logout", cookies={"vsid": SID})
    assert r.status_code == 200
    assert "vsid=" in r.headers.get("set-cookie", "")


def test_over_plain_http_the_cookie_is_not_marked_secure(client):
    """Secure is derived from the request, not hardcoded -- otherwise the cookie is silently
    dropped at http://localhost and local dev cannot sign in at all."""
    r = client.post("http://testserver/auth/login", json={"token": "good-token"})
    assert "secure" not in r.headers["set-cookie"].lower()


def test_a_proxy_terminating_tls_still_yields_a_secure_cookie(client):
    """The bug the first deploy actually shipped. Cloud Run terminates TLS at its front end and
    forwards over plain HTTP, so the scheme inside the container is "http" and Secure went
    missing in production while every test passed -- a test sets the scheme directly and so
    cannot see a proxy in front of it. X-Forwarded-Proto is the only evidence that survives."""
    r = client.post("http://testserver/auth/login", json={"token": "good-token"},
                    headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_the_first_hop_decides_when_several_proxies_appended_themselves(client):
    """X-Forwarded-Proto accumulates left to right, so the browser's own protocol is the FIRST
    entry. Reading the last one would report the hop nearest us, which is the plaintext one."""
    r = client.post("http://testserver/auth/login", json={"token": "good-token"},
                    headers={"X-Forwarded-Proto": "https, http"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_signing_out_replaces_the_cookie_with_matching_attributes(client):
    """A browser matches attributes when deciding which cookie a Set-Cookie replaces, so a delete
    that drops Secure/SameSite can leave the original in place and sign-out silently fails."""
    r = client.post("http://testserver/auth/logout", cookies={"vsid": SID},
                    headers={"X-Forwarded-Proto": "https"})
    sc = r.headers["set-cookie"].lower()
    assert "secure" in sc and "samesite=lax" in sc and "httponly" in sc


def test_mydocs_without_a_cookie_is_401_and_never_reaches_the_database(client):
    """_user_id runs before any query, so this is a safe test to run with no DB present."""
    assert client.post("/mydocs", json={}).status_code == 401


def test_serving_a_file_needs_a_session_not_a_url_parameter(client):
    """The whole point of the move: /file takes no identity in the URL any more. A query string
    naming the owner -- which used to return the file -- is now just an ignored parameter."""
    r = client.get("/file/a1b2c3d4e5f6", params={"session_id": EMAIL, "token": "good-token"})
    assert r.status_code == 401


def test_a_malformed_doc_id_is_rejected_before_identity_is_considered(client):
    r = client.get("/file/..%2F..%2Fetc%2Fpasswd", cookies={"vsid": SID})
    assert r.status_code in (400, 404)

def test_auth_config_reports_an_existing_session(client):
    """The page cannot read an HttpOnly cookie, so this is the only way it learns it is signed in.
    If this silently returned nothing, a returning user would land on the login screen holding a
    perfectly good session."""
    assert client.get("/auth/config", cookies={"vsid": SID}).json()["email"] == EMAIL
    assert client.get("/auth/config").json()["email"] is None
    assert client.get("/auth/config", cookies={"vsid": ANON}).json()["email"] is None


def test_the_common_path_never_touches_the_database(monkeypatch):
    """An anonymous visitor and a first-time one are the usual traffic once sign-in is optional,
    and neither can have a session row. Querying for one anyway cost ~700ms of Neon round trip
    per request -- a cost the identity check did not have before sessions existed. Any connection
    attempt here fails the test rather than being merely slow."""
    monkeypatch.setattr(app, "_GOOGLE_CLIENT_ID", "client-id.apps.googleusercontent.com")
    def forbidden(*a, **k):
        raise AssertionError("the identity check reached the database on the anonymous path")
    monkeypatch.setattr(app.psycopg, "connect", forbidden)

    assert app._user_id(Req({"vsid": ANON}), require=False) == ANON
    assert app._ANON_ID.fullmatch(app._user_id(Req(), require=False))     # no cookie at all
    assert app.auth_config(Req({"vsid": ANON}))["email"] is None          # and on page load
    assert app.auth_config(Req())["email"] is None
