"""
Main investigation agent with two subagents exposed as tools:
one reads SHAP contributions, the other finds similar confirmed-fraud cases.

Mirrors the pattern prototyped in llm.ipynb (create_agent + subagents-as-tools),
promoted here so it can be called from a script/API instead of a notebook cell.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool

from src.agent.prompts import (
    JUDGE_PROMPT,
    MAIN_AGENT_PROMPT,
    SHAP_SUBAGENT_PROMPT,
    SIMILARITY_SUBAGENT_PROMPT,
)
from src.agent.tools import find_similar_fraud_cases_tool, read_shap_values_tool

# TODO(user): confirm model name / fallback per CLAUDE.md (deepseek-r1:7b,
# fallback llama3.1:8b) — llm.ipynb prototype used qwen3.5:27b instead.
OLLAMA_MODEL = "ollama:qwen3.5:27b"


def build_shap_subagent():
    return create_agent(
        model=OLLAMA_MODEL,
        tools=[read_shap_values_tool],
        system_prompt=SHAP_SUBAGENT_PROMPT,
    )


def build_similarity_subagent():
    return create_agent(
        model=OLLAMA_MODEL,
        tools=[find_similar_fraud_cases_tool],
        system_prompt=SIMILARITY_SUBAGENT_PROMPT,
    )


def build_main_agent():
    """Wrap each subagent as a tool for the main agent, per the notebook pattern.

    The main agent only gathers both subagents' raw output — it does not synthesize
    a final answer. That's the judge's job (see judge_verdict / investigate below).
    """
    shap_subagent = build_shap_subagent()
    similarity_subagent = build_similarity_subagent()

    @tool("explain_with_shap", description="Delegate to the SHAP subagent to explain why a transaction was flagged")
    def call_shap_subagent(query: str) -> str:
        result = shap_subagent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    @tool("find_similar_cases", description="Delegate to the similarity subagent to find comparable confirmed-fraud transactions")
    def call_similarity_subagent(query: str) -> str:
        result = similarity_subagent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    return create_agent(
        model=OLLAMA_MODEL,
        tools=[call_shap_subagent, call_similarity_subagent],
        system_prompt=MAIN_AGENT_PROMPT,
    )


def build_judge():
    """The final reviewer — has no tools, only reasons over what's handed to it."""
    return create_agent(
        model=OLLAMA_MODEL,
        tools=[],
        system_prompt=JUDGE_PROMPT,
    )


def investigate(transaction_id: str) -> str:
    """Run main agent -> judge end-to-end for a single transaction.

    The main agent gathers both subagents' findings; the judge critiques them and
    produces the answer that actually reaches the analyst.
    """
    main_agent = build_main_agent()
    query = f"Transaction {transaction_id} was predicted fraudulent. Gather both subagents' findings."
    main_result = main_agent.invoke({"messages": [{"role": "user", "content": query}]})
    gathered = main_result["messages"][-1].content

    judge = build_judge()
    judge_query = (
        f"Transaction ID: {transaction_id}\n\n"
        f"Subagent findings:\n{gathered}\n\n"
        "Review the above and produce the final justification."
    )
    judge_result = judge.invoke({"messages": [{"role": "user", "content": judge_query}]})
    return judge_result["messages"][-1].content
