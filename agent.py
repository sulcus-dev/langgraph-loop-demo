"""A slow, multi-step LangGraph agent for trying Sulcus without an API key.

It runs a short "research" loop: plan -> search -> reason -> repeat. The model
is a local stand-in that makes no network calls and needs no key. It reports
token usage on every call, the way a real provider does, so you can watch the
run live in Sulcus, press Stop run, or set a per-run token limit and see the
run fail when it is passed.

Settings (environment variables, all optional):
  DEMO_STEPS        number of research rounds (default 30)
  DEMO_STEP_SECONDS pause per round, so there is time to watch (default 2.0)
"""

import os
import time
from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

STEPS = int(os.environ.get("DEMO_STEPS", "30"))
STEP_SECONDS = float(os.environ.get("DEMO_STEP_SECONDS", "2.0"))

# Made-up usage numbers reported for every model call.
INPUT_TOKENS = 800
OUTPUT_TOKENS = 200


class LocalDemoModel(BaseChatModel):
    """A stand-in chat model: no network, no API key, fixed reply."""

    @property
    def _llm_type(self) -> str:
        return "local-demo-model"

    def get_num_tokens_from_messages(self, messages: List[BaseMessage], tools: Optional[Any] = None) -> int:
        # Simple word count, so no tokenizer has to be downloaded.
        return sum(len(str(m.content).split()) for m in messages)

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        usage = {
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
        }
        message = AIMessage(content="Noted. Next I should look for more evidence.", usage_metadata=usage)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "token_usage": {
                    "prompt_tokens": INPUT_TOKENS,
                    "completion_tokens": OUTPUT_TOKENS,
                    "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
                }
            },
        )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    round: int


model = LocalDemoModel()


def plan(state: State) -> dict:
    print(f"[plan] round {state['round'] + 1} of {STEPS}", flush=True)
    return {"messages": [HumanMessage(content="Decide the next question to research.")]}


def search(state: State) -> dict:
    # A stand-in tool call that takes a moment, like a real web search.
    time.sleep(STEP_SECONDS)
    print("[search] fetched 3 results", flush=True)
    return {"messages": [HumanMessage(content="Search results: three short snippets.")]}


def reason(state: State) -> dict:
    reply = model.invoke(state["messages"][-4:])
    print("[reason] model call finished", flush=True)
    return {"messages": [reply], "round": state["round"] + 1}


def should_continue(state: State) -> str:
    return END if state["round"] >= STEPS else "plan"


graph = StateGraph(State)
graph.add_node("plan", plan)
graph.add_node("search", search)
graph.add_node("reason", reason)
graph.add_edge(START, "plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "reason")
graph.add_conditional_edges("reason", should_continue, {"plan": "plan", END: END})
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"messages": [], "round": 0}, {"recursion_limit": STEPS * 4 + 10})
    print(f"Finished after {result['round']} rounds.", flush=True)
