# SEC Filing Agent — Video Demo Script

> 目标:2–3 分钟,纯屏幕 + 画外音(英文旁白,面向 startup)。
> 旁白(英文)直接念;**【操作】**是你要点/输入的动作(中文)。
> 服务:`http://localhost:8100`(gpt-4o + Google 登录 + Langfuse,env=demo)。

---

## 录前准备(关掉录制软件先做)

1. **三个浏览器标签准备好:**
   - ① `http://localhost:8100`(主界面)
   - ② Langfuse 的 demo 项目 traces 页面(已登录)
   - ③ GitHub README 页面:`github.com/Zhonghui-li/sec-filing-agent`
2. **主界面先登出**(从干净的登录页开始录)。
3. **硬刷新** `Cmd+Shift+R`,确保加载最新前端。
4. **账号状态确认**:用 `lizhonghui923@gmail.com` 登录后,文档选择器里应有 **Mosaic + Acme 两个**(已上传),且 **Berkshire 不在**(留给现场上传)。如不符,见文末"账号状态"。
5. **要现场上传的文件:** `ingest/sample/berkshire_2023_annual_report.pdf`
6. **先把要问的 4 个问题复制好**(录的时候直接粘贴,更顺):
   - `What was Apple's revenue in fiscal 2024?`
   - `What was Apple's net margin in fiscal 2024?`
   - `What were the operating earnings in 2023?` (先在下拉框选中 Berkshire 再问)
   - `What was the R&D spending in 2023?`
7. **先整套跑一遍**(确认答案/引用/拒答都正常),再正式录。

---

## ① 开场 + 自我介绍(~30s)

**【操作】** 停在**登录页**(Sign in with Google 那屏)。

**【画外音】**
> "Hi, I'm Zhonghui. I'm a recently graduate master student on computer science at UC santa cruz. Over the past year I've been building AI agents, focused on making them trustworthy in production. This is a finance-grade agent that answers questions over SEC filings — and over your own uploaded documents. The core idea: every number comes from a structured tool, never the model, and every claim is cited. Let me show you."

**【操作】** 点 **Sign in with Google** → 选账号 → 进入界面。

**【画外音】**(登录时)
> "It has Google sign-in, so each user's documents stay private to their account."

---

## ② 公开数据:精确数字 + 可点引用 + 审计轨迹(~50s)

**【操作】** 输入框粘贴 `What was Apple's revenue in fiscal 2024?` → 回车。

**【画外音】**(它在答时)
> "It's answering from public SEC data. The number doesn't come from the model's memory — it comes from a structured XBRL tool."

**【操作】** 答案出来后,鼠标指向数字,点 **📎 Sources 里的可点链接**(新标签打开 SEC EDGAR 原文)。

**【画外音】**
> "Here's the exact figure, and the citation is clickable — it deep-links straight to the source filing on SEC EDGAR. Every number is auditable."

**【操作】** 切回来,点答案下的 **"show audit trail"** 展开,指向出现的 `get_financials`。

**【画外音】**
> "And I can see exactly how it got there. It called `get_financials` — the tool that pulls the exact figure straight from structured XBRL data, so the number is never the model guessing. Full transparency on how the answer was produced."

---

## ②b 金融指标:确定性计算工具,不让模型乱算(~25s)

**【操作】** 输入框粘贴 `What was Apple's net margin in fiscal 2024?` → 回车 → 出 **24.0%**。

**【画外音】**
> "It's the same discipline for ratios. The agent doesn't do the arithmetic itself — it calls a deterministic tool with the formula and the right accounting conventions baked in. You can see it returns net income over revenue. I built this after external benchmarking showed models quietly get conventions wrong. Here the definition is fixed in code, so the number is reproducible and auditable."

> 💡 **可选加强**(想突出 proxy-substitution 故事):再问 `What was Apple's debt-to-equity in fiscal 2024?` → 出 **1.51**,答案口径写着 "long-term debt / equity (debt = interest-bearing debt, NOT total liabilities)" —— 这正是我修过的真实 bug。

---

## ③ 私有文档:上传 + 解析 + 文档选择 + 单元格溯源(~60s)

**【操作】** 点 **📎 Upload a document** → 选 `berkshire_2023_annual_report.pdf`。

**【画外音】**(解析时,~12s,用旁白填这段等待)
> "Now we try upload a document — I can bring my own document: here, Berkshire Hathaway's 2023 annual report. Notice I already have two documents from before; they persist across sessions, private to my Google account. I'm adding this as a third. It's parsed with a document-understanding model that preserves the tables."

**【操作】** 解析完,点 **"show what was parsed (docling)"** 闪一下抽取的数字表。

**【画外音】**
> "You can see exactly what it extracted — the table figures, structured."

**【操作】** 在下拉框 `#docscope` 里选中 **"Only: berkshire_2023_annual_report.pdf"**(此时旁边出现 🗑 Delete 按钮)。⚠️ **必须先选中 Berkshire 再问**(否则跨全部文档时 "operating earnings" 会和 Mosaic 的分部数据撞)。

**【画外音】**
> "With several documents loaded, this dropdown lets me pick exactly which one to read. The agent hard-scopes to it, so it never mixes figures across files — a real problem when you have multiple similar statements. Or I can leave it on 'across all my documents,' and when a figure could come from several files, it doesn't silently pick one — every number it returns is labeled with its source file, so figures never get misattributed. And I can manage these — add or remove a document anytime." let's try it.

**【操作】** 输入框粘贴 `What were the operating earnings in 2023?` → 回车 → 出 **$37,350 million**。

**【画外音】**
> "The figure — 37,350 million in operating earnings — comes straight from the extracted table cell, never from prose."

**【操作】** 指向 **"📎 From your document"** 的溯源(`berkshire… · p.13 · row 'Operating earnings' / col '2023'`),点 **p.13 链接**跳原文页;可选展开 audit trail 指向 `get_my_financials`。

**【画外音】**
> "Cited down to the exact cell — row 'Operating earnings', column 2023, page 13 — and the page links back to the original PDF. Same finance bar on private data: in the audit trail it called `get_my_financials`, which reads that figure from the document's table."

---

## ④ 诚实拒答(不编)(~20s)

**【操作】**(保持作用域在 Berkshire)粘贴 `What was the R&D spending in 2023?` → 回车。

**【画外音】**
> "And when the data isn't in the document, it doesn't invent a number — it tells me honestly. In finance, that matters more than always having an answer."

---

## ⑤ 幕后的严谨:eval + judge + Langfuse(~30s)

**【操作】** 切到 **Langfuse 标签**,停在 traces 列表 / 点开刚才那条 trace。

**【画外音】**
> "Now, making an agent trustworthy isn't just about the answer you see — in production you have to be able to see what the agent actually did, and measure it. So I instrument everything with Langfuse. Every request logs which tools were called, what was cited, latency, and cost — this is how I caught real production bugs."

**【操作】** 切到 **GitHub README 标签**,慢慢滚过这三处:**CI badge(顶部)→ "12-metric suite in CI" 那行 → "Cohen's κ = 0.95" 那段**。

**【画外音】**
> "And it's not just a demo — it's gated in CI by a 12-metric evaluation suite, plus a domain-tuned judge I calibrated against human labels at 0.95 agreement, to catch quality issues a generic metric misses."

---

## ⑥ 收尾(~15s)

**【操作】** 可停在 GitHub 或主界面。

**【画外音】**
> "The code's on GitHub, linked below. I also built a second agent, Slug Advisor, which is already live — feel free to try it. This finance agent's public demo link is coming very soon. Thanks for watching."

---

## 验证过的数字 / 答案速查(录前对一遍)

| 问题 | 期望答案 | 引用 |
|---|---|---|
| Apple FY2024 revenue | 391,035 (百万) ≈ $391B | SEC EDGAR 10-K 链接 |
| Apple FY2024 net margin | **24.0%** (net income / revenue) | 10-K 链接 + 公式 |
| Apple FY2024 debt-to-equity(可选) | **1.51** (long-term debt / equity, **非** total liabilities) | 10-K 链接 + 口径 |
| Berkshire operating earnings 2023 | **$37,350 million** | `berkshire… · p.13 · row 'Operating earnings' / col '2023'` |
| Berkshire R&D 2023 | **诚实拒答**(数据里没有) | — |

> ⚠️ **问 Berkshire 的 operating earnings 前,务必先在下拉框选中 Berkshire**(hard-scope)。Berkshire 那份里 "Operating earnings" 是**唯一**一行(p.13),scoped 后干净返回 $37,350;但**不选、留在 across-all** 时会和 Mosaic 的分部 operating earnings 撞,画面会乱。
> Acme / Mosaic 只是作为"之前已上传"的文档摆在选择器里,demo 不问它们。

---

## 录制小贴士

- **比平时慢 20%**,旁白和画面对得上;答案生成那几秒正好用旁白填。
- **每个链接录前先点一遍**确认能跳(EDGAR、p.1 原文页)。
- **Langfuse** 那段:确保你刚问的几条 trace 已经出现在列表里(env=demo),切过去才有东西看。
- **删除是真删**(DB + 文件):Mosaic / Acme 是**预置**文档,别误删;Berkshire 是**现场上传**的。演练时多次上传 Berkshire 没关系——同名会自动替换。误删了都能从 `ingest/sample/` 重传。
- gpt-4o 偶有波动,**正式录之前整套跑一遍**最稳。

---

## 账号状态(录前应满足 / 出问题时恢复)

**录前 `lizhonghui923@gmail.com` 账号应为:**
- ✅ 已上传:`mosaic_q1_2025_earnings.pdf`、`acme_internal_financials.pdf`
- ❌ 不在:`berkshire_2023_annual_report.pdf`(现场上传它,演示 2→3)

**如果状态不对(比如不小心把 Mosaic/Acme 删了,或 Berkshire 还在):**
- **缺 Mosaic 或 Acme** → 在网页里登录后上传对应文件(`ingest/sample/` 下),它们会变成"已上传"。
- **Berkshire 还在** → 在下拉框选中它 → 点 🗑 Delete 删掉,让现场上传是真的 2→3。

> 上传顺序不影响,只要录制开始时是「Mosaic + Acme 在、Berkshire 不在」即可。
