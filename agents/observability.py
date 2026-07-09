"""Audit-trail observability via Langfuse.

In a regulated domain the trace IS the audit trail: every run logs the question, the
answer, WHICH tools were called (the data sources touched), WHICH filings were cited
(accessions), whether the agent abstained and why, plus latency and token cost.

Active only when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set; otherwise a no-op,
so the agent runs unchanged without an account.
"""
import os
from contextlib import contextmanager

LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
_MODEL = os.environ.get("GEN_LLM_MODEL", "o4-mini")


def sum_usage(messages):
    """Sum input/output tokens across the run's messages (for $ cost tracking)."""
    inp = out = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None) or {}
        inp += u.get("input_tokens", 0)
        out += u.get("output_tokens", 0)
    return {"input": inp, "output": out} if (inp or out) else None


def _noop(**_):
    return None


@contextmanager
def trace_agent(question):
    """Wrap an agent run in one Langfuse generation observation (no-op without keys), so
    its duration is the real latency. Yields `record(answer, tools_used, usage, audit)`."""
    if not LANGFUSE_ENABLED:
        yield _noop
        return
    try:
        from langfuse import get_client
        lf = get_client()
    except Exception as e:  # never break the agent
        print(f"[langfuse] disabled ({type(e).__name__}: {e})")
        yield _noop
        return

    with lf.start_as_current_observation(name="sec-agent", as_type="generation", input=question):
        try:
            trace_id = lf.get_current_trace_id()   # so user feedback can be scored onto this trace
        except Exception:
            trace_id = None

        def record(answer=None, tools_used=None, usage=None, audit=None):
            try:
                lf.update_current_generation(
                    output=answer, model=_MODEL,
                    usage_details=({"input": usage["input"], "output": usage["output"]}
                                   if usage else None),
                    # the audit trail: tools touched, filings cited, abstention
                    metadata={"tools_used": tools_used,
                              "n_tool_calls": len(tools_used or []),
                              **(audit or {})})
                lf.set_current_trace_io(input=question, output=answer)
            except Exception as e:
                print(f"[langfuse] record skipped: {e}")
            return trace_id
        yield record
    try:
        lf.flush()
    except Exception:
        pass
