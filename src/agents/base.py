"""Shared agent logic: forced RAG → LLM call → sandbox exec → state update."""

import re
import time
import logging
from datetime import datetime, timezone

from src.utils.llm import call_llm
from src.rag.retriever import rag_query
from src.rag.indexer import add_document
from src.safety.sandbox import run_sandboxed
from src.safety.guardrails import extract_python_code, sanitize_llm_output
from src.config import MODELS, MAX_CODE_FAILURES

logger = logging.getLogger(__name__)


def _strip_variable_redefinitions(code: str, prefix: str) -> str:
    """Remove lines from LLM code that redefine variables set in our prefix."""
    # Extract variable names from prefix (lines like: VAR_NAME = '...')
    prefix_vars = re.findall(r"^([A-Z_]+)\s*=", prefix, re.MULTILINE)
    if not prefix_vars:
        return code

    cleaned_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        # Skip lines that redefine our variables
        skip = False
        for var in prefix_vars:
            if re.match(rf"^{var}\s*=", stripped):
                skip = True
                break
        if not skip:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def run_agent(
    agent_name: str,
    state: dict,
    rag_question: str,
    system_prompt: str,
    user_prompt: str,
    model_key: str = "coder",
    fallback_key: str = "coder_fallback",
    code_prefix: str = "",
) -> dict:
    """Execute the standard agent pattern: RAG → LLM → sandbox → update state.

    Returns {"code": str, "result": dict, "llm_response": dict, "rag_context": str}.
    """
    start_time = time.time()

    # 1. Forced RAG
    rag_context = rag_query(rag_question, "ml_knowledge", top_k=5)
    logger.info("[%s] RAG retrieved %d chars of context", agent_name, len(rag_context))

    # 2. LLM call
    full_system = f"{system_prompt}\n\nKnowledge from RAG:\n{rag_context}" if rag_context else system_prompt

    llm_response = call_llm(
        model=MODELS[model_key],
        system=full_system,
        user=user_prompt,
        fallback=MODELS[fallback_key],
        max_tokens=8192,
    )

    raw_content = sanitize_llm_output(llm_response["content"])
    code = extract_python_code(raw_content)

    # Inject hardcoded constants: remove any LLM redefinitions, then prepend ours
    if code_prefix:
        code = _strip_variable_redefinitions(code, code_prefix)
        code = code_prefix + "\n" + code

    # 3. Execute code with retries
    result = None
    for attempt in range(MAX_CODE_FAILURES):
        result = run_sandboxed(code, cwd=state["workspace_dir"])
        if result["ok"]:
            break
        logger.warning(
            "[%s] Code execution failed (attempt %d/%d): %s",
            agent_name, attempt + 1, MAX_CODE_FAILURES, result.get("error") or result.get("stderr", "")[:200],
        )
        if attempt < MAX_CODE_FAILURES - 1:
            # Ask LLM to fix the code
            fix_prompt = (
                f"The previous code failed with error:\n"
                f"stderr: {result.get('stderr', '')[:1000]}\n"
                f"error: {result.get('error', '')}\n\n"
                f"Fix the code. Return only the corrected Python code.\n"
                f"IMPORTANT: Write FLAT script code, no functions. "
                f"Use these pre-defined variables: {', '.join(re.findall(r'^([A-Z_]+)\\s*=', code_prefix, re.MULTILINE)) if code_prefix else 'N/A'}"
            )
            fix_response = call_llm(
                model=MODELS[model_key],
                system=system_prompt,
                user=fix_prompt,
                fallback=MODELS[fallback_key],
            )
            fixed_code = extract_python_code(sanitize_llm_output(fix_response["content"]))
            # Re-apply prefix stripping and injection
            if code_prefix:
                fixed_code = _strip_variable_redefinitions(fixed_code, code_prefix)
                code = code_prefix + "\n" + fixed_code
            else:
                code = fixed_code

    duration = time.time() - start_time

    # 4. Log agent action
    log_entry = {
        "agent": agent_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": llm_response["model_used"],
        "duration_s": round(duration, 2),
        "tokens": llm_response["tokens"],
        "rag_queries": [rag_question],
        "code_executed": True,
        "exit_code": 0 if result["ok"] else 1,
        "result": "success" if result["ok"] else "failure",
    }

    # Log experiment to ChromaDB
    try:
        add_document(
            "experiment_log",
            f"{agent_name}_{state.get('iteration', 0)}_{int(time.time())}",
            f"Agent: {agent_name}, Result: {result.get('stdout', '')[:500]}",
            {"agent": agent_name, "success": result["ok"]},
        )
    except Exception as e:
        logger.debug("Failed to log experiment: %s", e)

    return {
        "code": code,
        "result": result,
        "llm_response": llm_response,
        "rag_context": rag_context,
        "log_entry": log_entry,
    }
