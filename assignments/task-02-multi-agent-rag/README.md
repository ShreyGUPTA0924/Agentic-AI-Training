# Multi-Agent RAG System with Human-in-the-Loop using LangGraph

This repository contains the implementation of **Assignment Task 02: Multi-Agent RAG System with Human-in-the-Loop (HiTL)** using LangGraph.

The system uses a multi-agent team (Supervisor, Researcher, and Writer) to answer user questions about technical documents using local vector search (ChromaDB) with HuggingFace embeddings. It employs a Human-in-the-Loop mechanism that pauses execution after RAG retrieval, presents the chunks for inspection, and prompts the user to approve, reject, or rewrite the search query.

---

## 🎯 Architecture Diagram

Below is the workflow topology of the Multi-Agent system:

```mermaid
graph TD
    START([START]) --> SV[Supervisor Node]
    SV -->|next_agent = researcher| QR[Query Rewriter Node]
    SV -->|next_agent = writer| WR[Writer Node]
    SV -->|next_agent = end| END([END])
    
    QR --> RE[Researcher Node]
    RE --> HR_Pause{{"human_review_node (Pauses via interrupt_before)"}}
    
    HR_Pause -->|Resumes with None| HR[human_review_node (Dynamic interrupt)]
    HR -->|y / yes (approved = True)| WR
    HR -->|n / no (approved = False)| SV
    HR -->|r <new query> (rewrites query)| QR
    
    WR --> SV
```

---

## 📂 Folder Structure

```
assignments/task-02-multi-agent-rag/
├── README.md                      # Detailed notes on setup, execution, and concepts
├── requirements.txt               # Project python dependencies
├── .env                           # Local environment keys (kept private)
├── .env.example                   # Shared template for environment variables
│
├── data/
│   ├── chroma_db/                 # Persisted local Chroma vector store
│   └── documents/                 # Raw context source text documents
│       ├── sample_ai.txt          # Transformer architecture source document
│       └── sample_ml.txt          # RLHF source document
│
├── agents/
│   ├── __init__.py
│   ├── supervisor.py              # Routes execution using Pydantic structured output
│   ├── query_rewriter.py          # Bonus: Rewrites search query using Groq LLM
│   ├── researcher.py              # Coordinates Chroma similarity searches
│   └── writer.py                  # Synthesizes final cited technical answers
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py                  # Idempotent segmenting, embedding, indexing script
│   └── retriever.py               # Chroma database similarity search wrapper
│
├── state.py                       # Shared TypedDict PipelineState
├── graph.py                       # LangGraph compilation & routing logic
└── main.py                        # Interactive CLI harness
```

---

## 🛠️ Installation & Setup

1. **Activate Virtual Environment**:
   Ensure you use the existing environment inside `.venv/`.
   ```powershell
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your Groq API Key:
   ```env
   GROQ_API_KEY=gsk_yioQSqrFGO...
   ```

---

## 📖 Document Ingestion (RAG Setup)

Run the ingestion script to load files from `data/documents/` into the ChromaDB vector database.
The ingestion is fully **idempotent**; running it multiple times will not duplicate embeddings. It automatically computes unique, deterministic chunk IDs based on the document source name and chunk index.

```bash
python rag/ingest.py
```

---

## 🚀 Running the Project

Launch the main CLI runner:
```bash
python main.py
```

---

## 🤝 Human Approval Workflow

When you ask a question, the graph executes up to the **Human Review Node** and pauses:
1. It displays all the chunks retrieved from the Chroma database.
2. It prompts you with three options:
   * **`y` / `yes`**: Approves the context. The graph moves forward to the **Writer Agent** to generate the final cited answer.
   * **`n` / `no`**: Rejects the context. The graph goes back to the **Supervisor** which can decide to research again or exit.
   * **`r <new query>`**: Overrides the query. The graph clears existing retrieved documents, updates the query to `<new query>`, and runs the researcher again to fetch better results.

---

## 💡 Multi-Agent & RAG Explanation

### 1. The Supervisor/Worker Pattern
This architecture is modeled after human organizations. The **Supervisor** acts as the project manager, observing the shared state (`PipelineState`) and routing execution using Pydantic structured output. The Workers (**Researcher** and **Writer**) are experts in their own narrow domains (retrieval and generation, respectively). By decoupling retrieval from generation, we can include checks (like human feedback) in between and ensure LLM boundaries are respected.

### 2. Retrieval-Augmented Generation (RAG)
Large Language Models have static knowledge cutoff dates and are prone to hallucinations. RAG fixes this by searching external databases for context relevant to the user query and supplying that context directly to the LLM's prompt. 
* **Embeddings**: Text chunks are converted into 384-dimensional dense vectors using a local `sentence-transformers/all-MiniLM-L6-v2` model.
* **Vector Search**: ChromaDB compares the query vector against chunk vectors using cosine similarity to return the top `k=4` most relevant segments.

---

## 🌟 Bonus Features Implemented

* **Query Rewriter Node**: Before documents are retrieved, the system passes the user query through a query-rewriting LLM node. This expands terms, formats the query for semantic search, and significantly increases vector search recall.
* **ASCII Console Compatibility**: All UI outputs have been tailored for Windows Command Prompt/PowerShell consoles to prevent `UnicodeEncodeError` crashes.

---

## 📝 Example Session Output

```
============================================================
      MULTI-AGENT RAG PIPELINE WITH HUMAN-IN-THE-LOOP      
============================================================

Ask a question (e.g. about Transformer architecture or RLHF):
> Tell me about RLHF and reward models.

[System] Starting Multi-Agent workflow...
2026-06-24 20:37:06,249 | INFO | Researcher retrieved 4 document chunks.
2026-06-24 20:37:06,263 | INFO | Graph paused before human_review_node. Resuming to run node...
2026-06-24 20:37:06,264 | INFO | Human Review Node entered. Pausing for human feedback...

============================================================
                 HUMAN CONTEXT REVIEW                 
============================================================
Current Query: 'What are the key concepts and applications of Reinforcement Learning from Human Feedback (RLHF) and how do reward models play a role in this process?'

Retrieved Chunks (4):

[Doc 1]:
Reinforcement Learning from Human Feedback (RLHF)
Reinforcement Learning from Human Feedback (RLHF) is a powerful machine learning technique used to align large language models with human preferences, values, and safety standards...

[Doc 2]:
rejected ones. The reward model learns to output a scalar score representing the quality of a given response.

[Doc 3]:
3. Reinforcement Learning Fine-Tuning: The SFT model is further fine-tuned using reinforcement learning, typically with the Proximal Policy Optimization (PPO) algorithm. The language model acts as the policy, the prompt is the state, and the generated token sequence is the action...

[Doc 4]:
2. Reward Model Training: Prompts are fed to the SFT model to generate multiple candidate completions. Human evaluators rank these completions based on criteria like helpfulness, accuracy, and safety. A separate reward model (often a modified version of the SFT model) is trained...
============================================================

Options:
  y / yes : Approve context and proceed to Writer
  n / no  : Reject context and return to Supervisor
  r <query>: Refine search with a new custom query

Your choice: y

[System] Resuming workflow with user feedback...

============================================================
                     FINAL ANSWER                      
============================================================
Reinforcement Learning from Human Feedback (RLHF) is a machine learning technique used to align large language models with human preferences, values, and safety standards [Doc 1]. The key concepts of RLHF involve bridging the gap between next-token prediction and human utility, allowing language models to generate content that is more aligned with human values.

One of the key applications of RLHF is to prevent the generation of toxic, untruthful, or unhelpful content. This is achieved by using a reward model that learns to output a scalar score representing the quality of a given response [Doc 2, Doc 4]. The reward model is trained on a pair-wise ranking dataset, where human evaluators rank candidate completions based on criteria like helpfulness, accuracy, and safety.

In the RLHF process, the reward model plays a crucial role in providing feedback to the language model. The language model acts as the policy, the prompt is the state, and the generated token sequence is the action [Doc 3]. The reward model scores the generation, providing feedback that helps the language model to improve its performance.

The reward model is typically trained using a loss function that maximizes the score of preferred completions relative to rejected ones [Doc 4]. This allows the reward model to learn the preferences and values of human evaluators and provide more accurate feedback to the language model.

Overall, the key concepts and applications of RLHF involve using a reward model to provide feedback to a language model, allowing it to generate content that is more aligned with human values and preferences.
============================================================

[System] Session ended. Goodbye!
```

---

## 🔮 Future Improvements

1. **Re-Ranking Node**: Integrate a Cohere reranker or Cross-Encoder model to score retrieved chunks and pass only the top 2-3 highest-scoring chunks to the Writer to save context space.
2. **Evaluation Node**: Add a final node to evaluate the Writer's response for faithfulness (grounding in context) and relevance before printing the final answer.
3. **Database Clearing Script**: Implement a utility command to wipe local database collections when updating source documents.
