"""The identity contract, now that signing in is optional.

Opening the demo to real traffic split the endpoints in two: asking a question needs no account,
while anything that reaches a user's own uploaded documents still needs a verified Google identity.
That split is only safe because of one thing — the id an unverified caller may claim is restricted
to the `sess_` shape. Document scope is keyed on the same id, so an unverified caller allowed to
name an email would be asking questions inside that account. These pin both halves.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException                       # noqa: E402

import service.app as app                               # noqa: E402

ANON = "sess_" + "a1b2c3d4e5f6"


@pytest.fixture
def signed_in_world(monkeypatch):
    """Sign-in configured, and one token that verifies. Anything else is an unverified caller."""
    monkeypatch.setattr(app, "_GOOGLE_CLIENT_ID", "client-id.apps.googleusercontent.com")
    monkeypatch.setattr(app, "_verify_google",
                        lambda tok: "owner@example.com" if tok == "good-token" else None)


def test_asking_works_with_no_token_at_all(signed_in_world):
    assert app._user_id(None, ANON, require=False) == ANON


def test_a_verified_caller_keeps_their_email_even_where_it_is_optional(signed_in_world):
    """Otherwise a signed-in visitor asking a question would be scoped to their browser id and
    their own uploaded documents would drop out of scope."""
    assert app._user_id("good-token", ANON, require=False) == "owner@example.com"


def test_an_unverified_caller_cannot_claim_an_email(signed_in_world):
    """The whole reason the anonymous path is safe. Without this, `session_id=owner@example.com`
    reads that account's private documents through /ask."""
    with pytest.raises(HTTPException) as e:
        app._user_id(None, "owner@example.com", require=False)
    assert e.value.status_code == 401


@pytest.mark.parametrize("claimed", [
    "owner@example.com",
    "sess_short",                 # under the length floor
    "sess_has-a-dash",            # not in the character class
    "xsess_" + "a1b2c3d4e5f6",    # fullmatch, not search
    "sess_" + "a1b2c3d4e5f6" + " ",
    "",
    None,
])
def test_only_the_sess_shape_is_accepted_unverified(signed_in_world, claimed):
    with pytest.raises(HTTPException):
        app._user_id(None, claimed, require=False)


def test_private_endpoints_still_demand_a_token(signed_in_world):
    """require=True is the default, so a new endpoint is private unless it opts out."""
    with pytest.raises(HTTPException) as e:
        app._user_id(None, ANON)
    assert e.value.status_code == 401
    assert app._user_id("good-token", ANON) == "owner@example.com"


def test_with_sign_in_unconfigured_the_session_id_is_the_identity(monkeypatch):
    """Local dev and the test suite run without a client id; that path must not start demanding
    tokens, and must not start enforcing the `sess_` shape on ids it never issued."""
    monkeypatch.setattr(app, "_GOOGLE_CLIENT_ID", "")
    assert app._user_id(None, "anything-at-all") == "anything-at-all"
    assert app._user_id(None, "anything-at-all", require=False) == "anything-at-all"


def test_serving_an_uploaded_file_verifies_rather_than_trusting_the_url(signed_in_world):
    """/file used to take the identifier straight off the query string. The endpoint now routes it
    through _user_id, so an unverified request for someone's file is rejected before any query."""
    with pytest.raises(HTTPException) as e:
        app.get_file("a1b2c3d4e5f6", session_id="owner@example.com")
    assert e.value.status_code == 401


def test_a_malformed_doc_id_is_rejected_before_identity_is_even_considered(signed_in_world):
    with pytest.raises(HTTPException) as e:
        app.get_file("../../etc/passwd", session_id="owner@example.com", token="good-token")
    assert e.value.status_code == 400
