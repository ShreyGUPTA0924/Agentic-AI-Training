# Personal Research Agent using LangGraph

## Overview

This project implements a stateful conversational research agent using LangGraph and Groq LLM.

The agent can:

- Research a topic using web search
- Perform calculations
- Retrieve the current date
- Summarize search results
- Maintain conversation memory using MemorySaver

---

## Project Structure

```
assignments/task-01-langgraph-research-agent/

├── README.md
├── requirements.txt
├── .env.example
├── main.py
└── agent/
    ├── __init__.py
    ├── state.py
    ├── tools.py
    ├── nodes.py
    └── graph.py
```

---

## Features

- LangGraph-based workflow
- Groq LLM integration
- Tool calling
- Web search using DuckDuckGo
- Safe calculator tool
- Date retrieval tool
- Conversation memory with MemorySaver

---

## Installation

### Clone repository

```bash
git clone <repository-url>
cd task-01-langgraph-research-agent
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_actual_api_key
```

---

## Run

```bash
python main.py
```

---

## Example Usage

### Research Query

```
You: Research latest trends in multimodal AI

Agent:
Multimodal AI trends include...
```

### Calculator Query

```
You: Calculate 25 * 48

Agent:
1200
```

### Date Query

```
You: What is today's date?

Agent:
2026-06-12
```

---

## Architecture

```
START
  |
  v
agent_node
  |
  v
tool_node
  |
  v
summarizer_node
  |
  v
END
```

### Nodes

- agent_node → Reasoning and tool selection
- tool_node → Tool execution
- summarizer_node → Response generation

### Tools

- search_web()
- calculate()
- get_current_date()

### Memory

Conversation state is persisted using LangGraph MemorySaver.

---

## Technologies Used

- LangGraph
- LangChain
- Groq
- DuckDuckGo Search
- Python
