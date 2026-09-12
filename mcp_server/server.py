"""MCP server exposing the SEC filing tools (Model Context Protocol).

This makes the *capability* — exact figures pulled live from SEC XBRL for any U.S.
public company, deterministic ratio/growth/formula evaluation, and cited narrative
search over 10-K text — available to ANY MCP client (Claude Desktop, Cursor, ...),
so the client's own LLM can call them instead of recalling numbers from memory.

It reuses the exact same functions the in-app agent uses (agents.finance_tools /
agents.statements / agents.filings_retrieval) — the value is the grounded data
access, MCP is just the standard doorway to it.

Two deliberate differences from the agent's own tool list (agents.sec_agent.TOOLS):

  - `abstain` is NOT exposed. It is agent-internal control flow (it makes our
    ReAct loop's refusal explicit); an external client has its own way to decline.

  - The `_budgeted` wrappers are NOT applied. Those counters are module globals
    sized for one agent turn — over a long-lived MCP session they would never
    reset and would start refusing calls. Loop control belongs to the client.

Honest boundary: numbers reaching the client are grounded because they come from
the XBRL tools, but our output guardrail (which blocks any number in a *final
answer* that doesn't trace to a tool) runs inside our agent and cannot be enforced
on a third-party client's composition of these results.

`search_filings` needs DATABASE_URL (the pgvector narrative store); without it, it
returns a "not indexed" message rather than failing. The numeric tools need no DB.

Run (stdio transport, for Claude Desktop):
    OPENAI_API_KEY=... python mcp_server/server.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP

from agents.finance_tools import (get_financials, compute, get_ratio, get_growth,
                                  compute_formula)
from agents.statements import (get_statement, largest_line_item,
                               get_segment_breakdown, get_segment_growth)
from agents.filings_retrieval import search_filings

mcp = FastMCP("SEC Filing Agent")

# Registered straight from the agent's own tool functions: FastMCP reads each
# signature and docstring, which are already written as tool specs.
for _fn in (get_financials, compute, get_ratio, get_growth, compute_formula,
            get_statement, largest_line_item, get_segment_breakdown,
            get_segment_growth, search_filings):
    mcp.tool()(_fn)


if __name__ == "__main__":
    mcp.run()  # stdio transport
