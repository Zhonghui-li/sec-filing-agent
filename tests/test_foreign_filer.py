"""Offline tests for the foreign-private-issuer message upgrade (no network — foreign_filer_note
is stubbed). The live detection itself (20-F vs 10-K) is exercised end-to-end, not here."""
import agents.companyfacts as cf
from agents.finance_tools import _no_company_msg


def test_not_found_upgraded_to_foreign_filer_note(monkeypatch):
    monkeypatch.setattr(cf, "foreign_filer_note",
                        lambda q: "TOYOTA MOTOR CORP is a foreign private issuer — it files a Form 20-F ...")
    msg = _no_company_msg("TM")
    assert "foreign private issuer" in msg and "20-F" in msg
    assert "No company found" not in msg


def test_falls_back_to_delisted_hint_when_not_a_foreign_filer(monkeypatch):
    monkeypatch.setattr(cf, "foreign_filer_note", lambda q: None)
    msg = _no_company_msg("ATVI")
    assert "No company found" in msg and "COMPANY NAME" in msg
