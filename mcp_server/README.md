# SEC Filing Agent — MCP server

Exposes the filing tools over the **Model Context Protocol (MCP)** so any MCP client
(Claude Desktop, Cursor, …) can let *its own* LLM pull exact figures instead of
recalling them. It reuses the same functions the in-app agent uses — the value is
grounded access to SEC data; MCP is just the standard doorway to it.

**Numeric tools** (live SEC XBRL, any U.S. public company, every answer citing its
accession number):

- `get_financials(ticker, metric, fiscal_year, quarter)` — one exact figure
- `get_ratio(ratio, ticker, fiscal_year)` — standard ratios, formula fixed in code
- `get_growth(metric, ticker, fiscal_year)` — YoY against the adjacent fiscal year
- `compute_formula(expression, ticker, fiscal_year)` — a custom formula, evaluated deterministically
- `compute(op, a, b)` — yoy / diff / ratio on two figures
- `get_statement(ticker, statement, fiscal_year)` — full line items of a statement
- `largest_line_item(ticker, section, fiscal_year, smallest)` — max/min computed in code, not scanned by a model
- `get_segment_breakdown(ticker, dimension, metric, fiscal_year)` — by segment or geography
- `get_segment_growth(ticker, dimension, metric, fiscal_year)` — segment YoY on the filing's own recast priors

**Narrative tool** (pgvector over 10-K text, with citations):

- `search_filings(query, ticker, k, fiscal_year)` — risks, strategy, MD&A

The XBRL fetching, statement parsing, and vector store run **server-side**; only the
query action is exposed.

## Scope and boundary

Two tools the agent uses internally are deliberately **not** exposed:

- `abstain` — agent-internal control flow that makes our ReAct loop's refusal
  explicit. An external client declines its own way.
- the `_budgeted` wrappers — per-turn call counters that stop our loop from
  spinning. They are module globals sized for one agent turn; over a long-lived MCP
  session they would never reset and would start refusing calls. Loop control
  belongs to the client.

**What the grounding guarantee covers here.** Numbers reaching the client come from
the XBRL tools, so they are traceable to a filing. But the output guardrail — which
blocks any number in a *final answer* that doesn't trace back to a tool — runs
inside our agent, and cannot be enforced on how a third-party client composes these
results into prose. Tool-level grounding transfers; answer-level grounding does not.

`search_filings` needs `DATABASE_URL` (the pgvector narrative store); without it, it
returns a "not indexed" message rather than failing. The numeric tools need no DB.

## Run / test

```bash
pip install "mcp[cli]"          # on top of the project deps
export OPENAI_API_KEY=sk-...

# stdio server (what Claude Desktop launches):
python mcp_server/server.py

# or inspect interactively:
mcp dev mcp_server/server.py
```

## Add to Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "sec-filing-agent": {
      "command": "/ABS/PATH/venv/bin/python",
      "args": ["/ABS/PATH/sec-filing-agent/mcp_server/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "DATABASE_URL": "postgresql://..."
      }
    }
  }
}
```

Restart Claude Desktop, then ask: *"What was Apple's FY2024 revenue, and what does
its latest 10-K say about supply-chain risk?"* — Claude will call `get_financials`
and `search_filings` on this server. Paths resolve relative to the script, so the
launch cwd doesn't matter.

## Agent vs MCP server (two ways to consume the same tools)

- **In-app agent** (`service/`) — *our* LLM orchestrates the tools behind the output
  guardrail, packaged as a product (the web demo).
- **MCP server** (this) — exposes the same tools so *other* clients' LLMs can use
  them. Same tool functions, two consumption modes, different trust boundaries.
