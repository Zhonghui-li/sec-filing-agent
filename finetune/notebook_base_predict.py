"""
B · BASELINE CELL — paste into the Kaggle/Colab notebook (after the Unsloth model is loaded).

Goal: measure the routing GAP of Qwen2.5-7B-Instruct PROMPT-ONLY (no fine-tuning) on the 50 real
FinanceBench metrics questions, using Qwen's NATIVE tools template (standard function-calling eval,
apples-to-apples with how we'll evaluate the fine-tuned model later).

It writes preds_base.json ; download it and score locally (free, no GPU):
    python -m finetune.fb_routing_eval preds_base.json

Requires (already in the Unsloth notebook): a loaded `model`, `tokenizer`.
Upload finetune/fb_metrics_questions.json to the notebook first (Kaggle: Add Data / working dir).
"""
import json, re, torch
from unsloth import FastLanguageModel

# ---- the 4 tools the agent exposes, in Qwen/OpenAI function-calling schema -----------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "get_financials",
        "description": "Exact reported value of a single financial line item for a company/year from SEC XBRL "
                       "(e.g. revenue, net_income, capex, ebitda, free_cash_flow, total_assets, inventory, eps_diluted).",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string", "description": "stock ticker, e.g. AAPL"},
            "metric": {"type": "string", "description": "line item, e.g. revenue, net_income, capex, ebitda"},
            "fiscal_year": {"type": "integer", "description": "fiscal year, e.g. 2019"}},
            "required": ["ticker", "metric", "fiscal_year"]}}},
    {"type": "function", "function": {
        "name": "get_ratio",
        "description": "A named financial ratio for a company/year, computed deterministically from XBRL "
                       "(e.g. gross_margin, operating_margin, ebitda_margin, roe, roa, current_ratio, "
                       "debt_to_equity, asset_turnover, cash_conversion_cycle, dso, dio, dpo).",
        "parameters": {"type": "object", "properties": {
            "ratio": {"type": "string", "description": "ratio name, e.g. ebitda_margin"},
            "ticker": {"type": "string"},
            "fiscal_year": {"type": "integer"}},
            "required": ["ratio", "ticker", "fiscal_year"]}}},
    {"type": "function", "function": {
        "name": "get_growth",
        "description": "Year-over-year change of a metric for a company into a given fiscal year.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "description": "e.g. revenue, net_income, eps_diluted"},
            "ticker": {"type": "string"},
            "fiscal_year": {"type": "integer"}},
            "required": ["metric", "ticker", "fiscal_year"]}}},
    {"type": "function", "function": {
        "name": "compute_formula",
        "description": "Evaluate an arithmetic expression over line items for a company/year "
                       "(use when the question defines its own formula). Names are metrics like "
                       "operating_income, depreciation_amortization, revenue, total_assets; prev(x, n) = n years back.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. (operating_income + depreciation_amortization) / revenue"},
            "ticker": {"type": "string"},
            "fiscal_year": {"type": "integer"}},
            "required": ["expression", "ticker", "fiscal_year"]}}},
]

SYSTEM = ("You are a financial-analysis agent over SEC filings. For each question, call exactly ONE "
          "tool that would produce the answer. Prefer get_ratio for a named ratio, get_growth for a "
          "year-over-year change, compute_formula when the question defines its own formula, else "
          "get_financials for a single reported line item. Always emit a tool call.")

def _parse_tool_call(text):
    """Pull the first {name, arguments} out of Qwen's <tool_call>...</tool_call> block (or bare JSON)."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    blob = m.group(1) if m else None
    if blob is None:
        m = re.search(r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\{.*?\}\s*\}", text, re.S)
        blob = m.group(0) if m else None
    if blob is None:
        return {"name": None, "arguments": {}}     # no call -> counted wrong by the scorer
    try:
        d = json.loads(blob)
        return {"name": d.get("name"), "arguments": d.get("arguments", {})}
    except Exception:
        return {"name": None, "arguments": {}}

# ---- load the BASE model (prompt-only, NO LoRA) — reload clean so no adapter leaks in --------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

questions = json.load(open("fb_metrics_questions.json"))
preds = []
for i, q in enumerate(questions, 1):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}]
    inputs = tokenizer.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                           return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=256, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    call = _parse_tool_call(text)
    preds.append({"question": q["question"], "tool": call})
    print(f"[{i:>2}/50] {str(call['name']):16} {json.dumps(call['arguments'])[:60]}")

json.dump(preds, open("preds_base.json", "w"), indent=1)
print("\nwrote preds_base.json — download it, then locally:  python -m finetune.fb_routing_eval preds_base.json")
