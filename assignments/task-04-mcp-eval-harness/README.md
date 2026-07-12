# Task 04 — MCP Tool Server, Interoperable Agent, and Evaluation Harness

A LangGraph agent that consumes tools over the **Model Context Protocol (MCP)** instead of
importing them as Python functions: one MCP server exposes this project's own RAG retriever
and two utility tools, and a second, external MCP server exposes filesystem access. The agent
loads tools from both at startup and decides which to call per turn. A golden-dataset
evaluation harness then scores the agent end-to-end with an LLM judge and produces a
CI-style pass/fail report.

## Architecture

```
                         ┌─────────────────────────┐
                         │   agent/graph.py          │
                         │   LangGraph ReAct agent    │
                         │   (create_react_agent)     │
                         └──────────┬─────────────────┘
                                    │ tools loaded at startup via
                                    │ MultiServerMCPClient.get_tools()
                    ┌───────────────┴────────────────┐
                    │                                 │
        stdio subprocess                    stdio subprocess
                    │                                 │
     ┌──────────────▼──────────────┐    ┌─────────────▼──────────────────┐
     │ mcp_server/server.py          │    │ @modelcontextprotocol/          │
     │ FastMCP("task04-tools")       │    │ server-filesystem (npx, Node)   │
     │                                │    │                                  │
     │ • retrieve_context(query, k)  │    │ • list_directory, read_file,     │
     │   → Chroma similarity search  │    │   search_files, ...              │
     │ • calculate(expression)       │    │   scoped to ./data                │
     │ • get_current_date()          │    │                                  │
     │ • resource: eval://report     │    └──────────────────────────────────┘
     │ • prompt: agent_system_prompt │
     └──────────────┬─────────────────┘
                     │ imports (not reimplements) the
                     │ Chroma vector store built by rag/ingest.py
     ┌──────────────▼─────────────────┐
     │ data/chroma_db (persisted)       │
     │ built from data/documents/*.md,  │
     │ *.txt via rag/ingest.py          │
     └───────────────────────────────────┘
```

- **`mcp_server/`** — the FastMCP server. `tools.py` holds the plain-Python tool
  implementations (importable and unit-testable on their own); `server.py` wraps them with
  `@mcp.tool()` and runs over stdio. `retrieve_context` reuses the same Chroma collection
  built by `rag/ingest.py` — it does not reimplement retrieval.
- **`agent/`** — `llm.py` is the *only* place an LLM provider is selected (Groq → OpenAI →
  Anthropic → Google, first key found wins). `mcp_client.py` configures a
  `MultiServerMCPClient` pointed at both servers. `graph.py` fetches tools from both servers
  and builds a `create_react_agent` graph over them.
- **`eval/`** — `golden_dataset.json` (13 Q&A pairs covering all 5 source documents),
  `run_eval.py` (runs the full agent per question, scores with an LLM judge, writes
  `report.md`, exits non-zero below an 80% pass threshold).
- **`rag/ingest.py`** — chunks `data/documents/*` and embeds them into the persisted Chroma
  store at `data/chroma_db/`. Run once before using the server; safe to re-run (idempotent
  per-source delete + re-add).

## Which external MCP server, and why

**`@modelcontextprotocol/server-filesystem`** (the official reference filesystem server,
run via `npx`), scoped to this project's `data/` directory.

Reasons:
- It's the canonical example server from the MCP spec itself, so it's a good proof that the
  agent is talking real MCP and not something bespoke.
- It gives the agent a capability (browsing/reading raw files) that is meaningfully
  different from `retrieve_context` (semantic search over embedded chunks), so a single
  conversation that needs both tools actually exercises cross-server tool routing instead of
  two servers doing the same thing.
- It only needs Node/`npx` at run time (no extra Python dependency), and Node was already
  available in this environment.

## Setup

```bash
cd assignments/task-04-mcp-eval-harness
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env          # then fill in at least one LLM provider key
```

Requires **Node.js** (for `npx -y @modelcontextprotocol/server-filesystem`) — verify with
`node --version`.

Fill in `.env`:
- At least one of `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`.
- `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=<your LangSmith key>`,
  `LANGCHAIN_PROJECT=task-04-mcp-eval` to enable tracing (see below — required, but only
  active once you supply a real key).

Build the vector store (run once, or after editing `data/documents/`):

```bash
python -m rag.ingest
```

## Running it

**1. Verify the MCP server independently** (opens the MCP Inspector):

```bash
mcp dev mcp_server/server.py
```

It should load with no import errors and list `retrieve_context`, `calculate`,
`get_current_date`, the `eval://report` resource, and the `agent_system_prompt` prompt.

**2. Interactive agent** (connects to both MCP servers):

```bash
python main.py
```

Example query that requires both servers in one conversation:

```
You: Summarize what our documents say about vector databases, then list the files in the
     data directory.
```

The agent calls `retrieve_context` (task04_tools server) for the summary and `list_directory`
(filesystem server) for the file listing — confirmed working end-to-end during development.

**3. Evaluation harness:** see [Evaluation report summary](#evaluation-report-summary) below
for the exact commands and current status.

## Evaluation report summary

`eval/report.md` is generated by `python -m eval.run_eval` and is not committed as a static
file in this checkout — generate it locally with the command above. During development this
harness was run end-to-end against all 13 questions and produced a **100% pass rate
(13/13)** with `openai/gpt-oss-120b` as both the agent and judge model; that run is what
validated the scoring logic (faithfulness ≥ 3, relevance ≥ 3, ≥ 70% fact coverage) and the
80% CI-gate threshold below.

At submission time this project's Groq account had exhausted its free-tier daily token quota
(TPD) for both `openai/gpt-oss-120b` and `llama-3.3-70b-versatile` from that testing — Groq's
daily limits are per-account, not per-key, so generating a fresh key doesn't reset them. To
produce your own `eval/report.md`:

```bash
python -m eval.run_eval
echo $?   # 0 if pass rate >= 80%
```

If you hit `groq.RateLimitError` on tokens-per-day, either wait for Groq's daily reset or add
a different provider's key to `.env` (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`GOOGLE_API_KEY`) — `agent/llm.py`'s fallback chain will pick it up automatically since Groq
is only tried first when `GROQ_API_KEY` is set.

To demonstrate the CI-gate failure path (per the acceptance criteria), then restore a passing
report:

```bash
python -m eval.run_eval --force-fail   # forces the threshold to 101%, always exits 1
echo $?                                 # 1
python -m eval.run_eval                 # re-run normally to restore a passing report.md
echo $?                                 # 0
```

## LangSmith tracing

Tracing is wired through both entry points:
- `main.py` tags interactive runs with `config={"metadata": {"mode": "interactive"}}`.
- `eval/run_eval.py` tags each evaluation run with
  `config={"metadata": {"mode": "eval", "question_id": "<id>"}}`.

Because `create_react_agent` compiles a graph with named nodes (`agent`, `tools`), each node
execution shows up as its own named, inspectable run in the LangSmith UI once tracing is
enabled — no extra instrumentation needed beyond setting the env vars.

**To see traces yourself:** get a LangSmith API key from https://smith.langchain.com,
uncomment/set `LANGCHAIN_API_KEY` in `.env`, and run `python main.py` or
`python -m eval.run_eval`. Runs will appear under the `task-04-mcp-eval` project, filterable
by the `mode` metadata tag. *(This repo's local `.env` had `LANGCHAIN_API_KEY` commented out
during development — set yours to activate tracing; without it, `LANGCHAIN_TRACING_V2=true`
alone just produces harmless 401 warnings and no traces are uploaded.)*

<!-- Paste a trace screenshot or exported trace JSON here once you have a LangSmith key configured. -->

## Notes / known quirks

- **Model choice on Groq:** `agent/llm.py` defaults to `openai/gpt-oss-120b` rather than
  `llama-3.3-70b-versatile` for the Groq path. With 17 tools bound across two MCP servers,
  `llama-3.3-70b-versatile` was observed to occasionally emit a malformed
  `<function=name>{args}</function>` tool call that Groq's API then rejects with a 400. The
  OpenAI open-weight model produced reliably well-formed tool calls in the same setup.
- **`mcp dev <path>` and package imports:** the `mcp` CLI's `dev` command imports the server
  file directly by path and only adds that file's own directory to `sys.path`, not the
  project root — so a `from mcp_server.tools import ...` absolute import fails unless the
  project root is added to `sys.path` first. `mcp_server/server.py` does this itself at the
  top of the file so it works both under `mcp dev` and `python -m mcp_server.server`.

## Folder structure

```
assignments/task-04-mcp-eval-harness/
├── README.md
├── requirements.txt
├── .env.example
├── mcp_server/        # FastMCP server (server.py) + tool implementations (tools.py)
├── agent/              # llm.py, mcp_client.py, state.py, graph.py
├── eval/               # golden_dataset.json, run_eval.py, report.md
├── rag/                # ingest.py — builds data/chroma_db from data/documents/
├── data/
│   ├── documents/      # source .md/.txt files
│   └── chroma_db/       # persisted vector store (generated)
└── main.py             # interactive CLI entry point
```
