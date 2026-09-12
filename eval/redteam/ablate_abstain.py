"""Ablation: how much does the `abstain` TOOL contribute to the zero-fabrication rate?

The claim being tested: giving the model an explicit decline *action* (a tool call, sitting in
the same menu as the data tools) makes declining a first-class option, so it fabricates less than
when its only way to decline is prose.

Two arms, same model, same judge, same 50-item red-team set, run back to back:

  WITH    — production config (TOOLS includes abstain; SYSTEM_PROMPT as shipped).
  WITHOUT — `abstain` removed from TOOLS, and every abstain instruction in SYSTEM_PROMPT
            mechanically rewritten into the equivalent prose-decline instruction.

Why the prompt must change too: SYSTEM_PROMPT tells the model to "call the `abstain` tool" in a
dozen places. Dropping the tool but keeping those lines would measure a broken config, not the
hypothesis. The rewrite below is deliberately MINIMAL — it keeps the same reason taxonomy
(not_reported / year_unavailable / not_in_filings / out_of_scope / off_topic) and the same
"verify BEFORE declining" discipline, so the arms differ only in the MECHANISM of declining.
That rewrite is the one subjective element here; the substitutions are listed explicitly so the
diff is auditable, and each one is asserted to land (a silent miss would corrupt the arm).

Run (costs API — 2 x 50 agent runs):
    OPENAI_API_KEY=... DATABASE_URL=... python -m eval.redteam.ablate_abstain
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import agents.sec_agent as sa
from eval.redteam.run_redteam import SET, _judge, report

HERE = Path(__file__).resolve().parent
OUT = HERE / "ablation_abstain.json"

# ---- prompt rewrite: tool-call decline -> prose decline (same conditions, same taxonomy) ----
SUBS = [
    # tool-list entry
    ("- abstain: call this (instead of answering) whenever you cannot answer from the data.",
     "- When you cannot answer from the data, do NOT call a tool for it: say so plainly in your "
     "answer and state which reason applies."),
    # HARD RULE 5 — the main decline instruction
    ("call the `abstain` tool with the matching reason \\\n(not_reported / year_unavailable / not_in_filings), THEN briefly explain",
     "say plainly in your answer that you cannot answer and state the matching reason \\\n(not_reported / year_unavailable / not_in_filings), THEN briefly explain"),
    ("But you MUST \\\nverify BEFORE abstaining:",
     "But you MUST \\\nverify BEFORE declining:"),
    ("Only abstain if \\\nthe tool confirms it.",
     "Only decline if \\\nthe tool confirms it."),
    ("do NOT use total liabilities to answer a question about debt) — abstain instead.",
     "do NOT use total liabilities to answer a question about debt) — decline instead."),
    # search budget line
    ("`abstain` (not_in_filings)", "say plainly it is not in the filings (not_in_filings)"),
    # organic-growth line
    ("use search_filings, or abstain, if the question demands organic",
     "use search_filings, or decline, if the question demands organic"),
    # HARD RULE 6 — out_of_scope / narrative
    ("ticker (the tool fetches its XBRL live) — do NOT abstain out_of_scope.",
     "ticker (the tool fetches its XBRL live) — do NOT decline as out_of_scope."),
    ("Only abstain out_of_scope if the tool returns no data",
     "Only decline as out_of_scope if the tool returns no data"),
    ("an event is \"too recent\" or \"not in my data\" and abstain — SEARCH FIRST;",
     "an event is \"too recent\" or \"not in my data\" and decline — SEARCH FIRST;"),
    ("NEVER abstain on a narrative \\\nor corporate-event question WITHOUT calling search_filings first",
     "NEVER decline on a narrative \\\nor corporate-event question WITHOUT calling search_filings first"),
    # off-topic
    ("predictions, or real-time market data like stock prices — call `abstain` with reason off_topic.",
     "predictions, or real-time market data like stock prices — say plainly that this is off_topic."),
    # HARD RULE 8 — derived metrics
    ("Abstain (not_reported) ONLY if a tool reports a required figure isn't available. A wrong number is worse than an honest abstention.",
     "Decline (not_reported) ONLY if a tool reports a required figure isn't available. A wrong number is worse than an honest refusal."),
    # numeric-guard line
    ("have made an error — do NOT report that number; abstain instead.",
     "have made an error — do NOT report that number; decline instead."),
]


def _neutralized_prompt(prompt: str) -> str:
    out = prompt
    for old, new in SUBS:
        old, new = old.replace("\\\n", ""), new.replace("\\\n", "")   # prompt is a joined f-string
        assert old in out, f"substitution did not match:\n{old!r}"
        out = out.replace(old, new)
    leftover = [ln.strip() for ln in out.splitlines() if "abstain" in ln.lower()]
    assert not leftover, "abstain still referenced in the control prompt:\n" + "\n".join(leftover)
    return out


def _run_arm(name, tools, prompt, items):
    orig_tools, orig_prompt = sa.TOOLS, sa.SYSTEM_PROMPT
    sa.TOOLS, sa.SYSTEM_PROMPT = tools, prompt          # build_agent reads these module globals
    try:
        agent = sa.build_agent()
        print(f"\n### arm={name}  tools={len(tools)}  "
              f"abstain_in_tools={'abstain' in [t.name for t in tools]}")
        rows = []
        for i, it in enumerate(items, 1):
            out = sa.run_agent(it["q"], agent=agent)
            rows.append((it, out["answer"], out.get("tools_used", []), out.get("trace")))
            print(f"  ran [{i:>2}/{len(items)}] {it['id']}", flush=True)
        return _judge(rows)
    finally:
        sa.TOOLS, sa.SYSTEM_PROMPT = orig_tools, orig_prompt


def main():
    items = [json.loads(l) for l in SET.read_text().splitlines() if l.strip()]
    with_tools, with_prompt = list(sa.TOOLS), sa.SYSTEM_PROMPT
    without_tools = [t for t in sa.TOOLS if t.name != "abstain"]
    without_prompt = _neutralized_prompt(sa.SYSTEM_PROMPT)
    assert len(without_tools) == len(with_tools) - 1

    results = {}
    for name, tools, prompt in (("WITH_abstain", with_tools, with_prompt),
                                ("WITHOUT_abstain", without_tools, without_prompt)):
        rows = _run_arm(name, tools, prompt, items)
        print(f"\n===== {name} =====")
        results[name] = {"summary": report(rows), "rows": rows}

    a = results["WITH_abstain"]["summary"]
    b = results["WITHOUT_abstain"]["summary"]
    print("\n" + "=" * 74 + "\n ABLATION: the `abstain` tool\n" + "=" * 74)
    print(f"  WITH    abstain : {a['fabrications']}/{a['traps']} fabrications "
          f"= {a['fabrication_rate']:.0%}   (over-abstain {a['over_abstentions']})")
    print(f"  WITHOUT abstain : {b['fabrications']}/{b['traps']} fabrications "
          f"= {b['fabrication_rate']:.0%}   (over-abstain {b['over_abstentions']})")
    print(f"  delta           : {b['fabrication_rate'] - a['fabrication_rate']:+.0%} "
          f"fabrication rate when the tool is removed")

    OUT.write_text(json.dumps(
        {"ts": datetime.now(timezone.utc).isoformat(),
         "model": os.environ.get("GEN_LLM_MODEL", "o4-mini"),
         "n_items": len(items), "substitutions": len(SUBS),
         "WITH_abstain": results["WITH_abstain"]["summary"],
         "WITHOUT_abstain": results["WITHOUT_abstain"]["summary"],
         "rows": {k: v["rows"] for k, v in results.items()}}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
