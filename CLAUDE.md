# CLAUDE.md — Kaggle Multi-Agent ML System

## What to build

An autonomous multi-agent system that takes a Kaggle competition directory as input
and produces `submission.csv` as output. LangGraph orchestration, RAG via ChromaDB,
OpenRouter LLM backend.

All work happens in `workspace/` — never modify the input competition directory.

---

## Models (OpenRouter)

```python
MODELS = {
    "coder": "qwen/qwen3-coder-480b-a35b",       # Data Agent, Model Agent — best coding
    "coder_fallback": "deepseek/deepseek-v3.2",   # fallback — cheap, strong
    "reasoning": "deepseek/deepseek-v3.2",         # Eval Agent — analysis, decisions
    "reasoning_fallback": "deepseek/deepseek-r1",  # deep analysis fallback
    "embedding": "all-MiniLM-L6-v2",              # local, sentence-transformers
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
```

On rate limit or error → auto-switch to fallback model. Log which model was used.

---

## Architecture (LangGraph StateGraph)

```
competition_dir/
    │
    ▼
[Supervisor] ── pure Python router, no LLM ──
    │                                         │
    ├──→ [Data Agent]    → cleans data, creates features
    ├──→ [Model Agent]   → trains models, tunes hyperparams
    ├──→ [Eval Agent]    → validates, checks leakage, scores
    │                                         │
    └──← feedback loop (max 3 iterations) ←───┘
    │
    ▼
workspace/submission.csv + workspace/report.md
```

### Supervisor (router) — NO LLM

```python
def route(state):
    if state["phase"] == "init":           return "data_agent"
    if state["phase"] == "data_ready":     return "model_agent"
    if state["phase"] == "model_trained":  return "eval_agent"
    if state["phase"] == "eval_done":
        if state["iteration"] >= state["max_iterations"]:
            return "submit"
        if state["metrics"]["score"] < state["target_threshold"]:
            return "data_agent"   # improve features
        return "submit"
```

### Agents — each is a LangGraph node

Every agent follows the same pattern:

1. **Forced RAG** — always query ChromaDB before LLM call
2. **LLM call** — send system prompt + RAG context + state info → get code
3. **Execute code** — run generated Python via subprocess with timeout
4. **Update state** — write results back

```python
def data_agent(state: PipelineState) -> PipelineState:
    # 1. Forced RAG
    context = rag_query(f"feature engineering {state['column_types']}", "ml_knowledge")

    # 2. LLM generates Python script
    code = call_llm(
        model=MODELS["coder"],
        system=f"You are a data scientist.\n\nKnowledge:\n{context}",
        user=f"Dataset summary:\n{state['data_summary']}\n\nGenerate a Python script...",
        fallback=MODELS["coder_fallback"],
    )

    # 3. Execute safely
    result = run_sandboxed(code, cwd=state["workspace_dir"], timeout=300)

    # 4. Update state
    return {**state, "phase": "data_ready", "feature_code": code, ...}
```

---

## State

```python
class PipelineState(TypedDict):
    # Input
    competition_dir: str
    task_description: str
    target_column: str
    metric_name: str            # accuracy, rmse, roc_auc, f1...

    # Working
    workspace_dir: str
    data_summary: str           # JSON: dtypes, shape, nulls
    column_types: str           # "datetime, categorical, numerical"
    cleaned_data_path: str
    model_path: str
    feature_code: str
    training_code: str
    cv_scores: list[float]
    metrics: dict
    submission_path: str

    # Control
    phase: str                  # init → data_ready → model_trained → eval_done → complete
    iteration: int
    max_iterations: int         # default 3

    # Logs
    agent_logs: list[dict]      # {agent, timestamp, duration_s, tokens, model_used, result}
    errors: list[str]
```

---

## RAG (ChromaDB + sentence-transformers)

### Two-level pattern

**Level 1 — Forced**: always runs before LLM call. Guarantees context for small models.
**Level 2 — Tool**: agent can call `search_knowledge()` as a tool if it needs more info.

### Collections

| Collection | Content | Size |
|---|---|---|
| `ml_knowledge` | Model guides, feature engineering, metrics | ~40 docs |
| `error_solutions` | Common errors → fixes | ~20 docs |
| `experiment_log` | Results from current run (self-updating) | grows at runtime |

### Knowledge base files to create

```
knowledge_base/sources/
├── models/
│   ├── xgboost.md          # when to use, key params, imbalanced data tips
│   ├── catboost.md          # native categorical handling, ordered boosting
│   ├── lightgbm.md          # speed advantages, large datasets
│   ├── sklearn_overview.md  # RandomForest, LogReg, SVM — when each fits
│   ├── model_selection.md   # decision tree: task type → model recommendation
│   └── hyperparameters.md   # default ranges per model per dataset size
├── features/
│   ├── datetime.md          # sin/cos encoding, lag, rolling, is_weekend
│   ├── categorical.md       # target encoding, frequency, ordinal, one-hot
│   ├── numerical.md         # log transform, binning, standardization
│   ├── missing_values.md    # strategies by column type, when to drop vs impute
│   └── feature_selection.md # correlation, mutual info, importance-based
└── evaluation/
    ├── metrics.md           # accuracy, F1, ROC-AUC, RMSE — when to use each
    ├── cross_validation.md  # stratified k-fold, time series split, group k-fold
    ├── data_leakage.md      # target encoding leak, temporal leak, common pitfalls
    └── ensembles.md         # averaging, stacking, blending — practical guide
```

Each file: 50–150 lines of real ML best practices (not stubs).

### Indexing

```python
# knowledge_base/build_index.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb, glob, pathlib

encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./knowledge_base/chroma_db")
collection = client.get_or_create_collection("ml_knowledge")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=64,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". "]
)

for filepath in glob.glob("knowledge_base/sources/**/*.md", recursive=True):
    text = pathlib.Path(filepath).read_text()
    topic = pathlib.Path(filepath).parent.name  # models / features / evaluation
    for i, chunk in enumerate(splitter.split_text(text)):
        collection.add(
            ids=[f"{filepath}_{i}"],
            documents=[chunk],
            embeddings=encoder.encode([chunk]).tolist(),
            metadatas=[{"source": filepath, "topic": topic}]
        )
```

---

## Safety

### Code sandbox

```python
import subprocess

def run_sandboxed(code: str, cwd: str, timeout: int = 300) -> dict:
    # Block dangerous patterns
    blocked = ["os.system", "subprocess", "shutil.rmtree", "rm -rf", "__import__"]
    for pattern in blocked:
        if pattern in code:
            return {"ok": False, "error": f"Blocked pattern: {pattern}"}

    script_path = f"{cwd}/code/agent_script.py"
    Path(script_path).write_text(code)

    result = subprocess.run(
        ["python", script_path],
        capture_output=True, text=True,
        cwd=cwd, timeout=timeout
    )
    return {"ok": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
```

### Input validation

Before starting: check files exist, train.csv has >10 rows, target column present, files <2GB.

### LLM guardrails

- If LLM returns malformed output → retry 2x with simpler prompt
- If code fails 3x in a row → skip to next phase, log error
- Track total tokens → hard stop at budget limit (default 200k tokens/run)

---

## Monitoring & Benchmarking

### Every agent action logs:

```python
{
    "agent": "data_agent",
    "timestamp": "2026-03-26T12:00:00Z",
    "model_used": "qwen/qwen3-coder-480b-a35b",
    "duration_s": 15.2,
    "tokens": {"input": 2400, "output": 800},
    "rag_queries": ["datetime feature engineering"],
    "code_executed": True,
    "exit_code": 0,
    "result": "success"
}
```

### Final report (workspace/report.md)

Auto-generated after pipeline completion:
- Architecture summary
- Per-agent stats: time, tokens, retries, model used
- Model comparison table: model name, CV score (mean ± std), training time
- RAG usage: total queries, top retrieved docs
- Total token cost estimate
- Iteration history: metric improvement per loop

---

## Project structure

```
kaggle-ml-agents/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example                 # OPENROUTER_API_KEY=sk-or-...
│
├── src/
│   ├── main.py                  # CLI entry point
│   ├── config.py                # MODELS dict, paths, thresholds
│   ├── graph/
│   │   ├── state.py             # PipelineState
│   │   ├── builder.py           # Build LangGraph StateGraph
│   │   └── supervisor.py        # Route function (pure Python)
│   ├── agents/
│   │   ├── base.py              # Shared: forced RAG → LLM call → sandbox exec
│   │   ├── data_agent.py
│   │   ├── model_agent.py
│   │   └── eval_agent.py
│   ├── rag/
│   │   ├── indexer.py           # Build ChromaDB from sources/
│   │   ├── retriever.py         # rag_query(question, collection, top_k)
│   │   └── tools.py             # @tool search_ml_knowledge, search_error_solutions
│   ├── safety/
│   │   ├── sandbox.py           # run_sandboxed()
│   │   ├── validators.py        # check_competition_dir()
│   │   └── guardrails.py        # sanitize_llm_output(), retry logic
│   ├── monitoring/
│   │   ├── logger.py            # log_agent_action()
│   │   └── report.py            # generate_report()
│   └── utils/
│       └── llm.py               # call_llm() with auto-fallback
│
├── knowledge_base/
│   ├── build_index.py
│   ├── sources/                 # ~15 markdown files (real content, not stubs)
│   │   ├── models/
│   │   ├── features/
│   │   └── evaluation/
│   └── chroma_db/               # gitignored
│
├── tests/
│   ├── test_sandbox.py
│   ├── test_rag.py
│   └── test_graph.py
│
└── workspace/                   # gitignored, created at runtime
```

---

## Dependencies

```toml
[project]
name = "kaggle-ml-agents"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "httpx>=0.27",
    "pandas>=2.0",
    "scikit-learn>=1.5",
    "xgboost>=2.0",
    "catboost>=1.2",
    "lightgbm>=4.0",
    "pyarrow>=15.0",
]
```

---

## Usage

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Build knowledge base (once)
python knowledge_base/build_index.py

# Run
python -m src.main \
    --competition-dir ./data/my-competition/ \
    --target-column target \
    --metric roc_auc \
    --task "Binary classification: predict whether..." \
    --max-iterations 3
```

Competition directory must contain `train.csv` and `test.csv`.

---

## Build order

1. `src/config.py` — models, paths, constants
2. `src/utils/llm.py` — OpenRouter client with retry + fallback
3. `src/graph/state.py` — PipelineState
4. `src/safety/sandbox.py` — subprocess execution
5. `knowledge_base/` — write source .md files + build_index.py
6. `src/rag/` — indexer, retriever, tools
7. `src/agents/base.py` — shared agent logic
8. `src/agents/data_agent.py` — first agent, test end-to-end
9. `src/agents/model_agent.py`
10. `src/agents/eval_agent.py`
11. `src/graph/supervisor.py` + `builder.py` — wire the graph
12. `src/main.py` — CLI
13. `src/monitoring/` — logger + report
14. `tests/`
15. `README.md`

Test on Titanic dataset after step 11 before proceeding.
