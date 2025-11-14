"""LLM adapter using proper LangGraph tool calling pattern."""
from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import Msg
from .config import config


# System prompt with GAIA formatting rules
SYSTEM_PROMPT = (
    "You are an AI assistant that helps answer questions by using tools.\n"
    "\n## Available Tools:\n"
    "\n**Search & Web:**\n"
    "- search_with_content: Search with full page content (1 result default, 3 with expand_search=True)\n"
    "\n**Files:**\n"
    "- file_read: Read files (supports CSV, XLSX, PDF, DOCX, TXT, JSON, YAML, XML, HTML)\n"
    "\n**Computation:**\n"
    "- calculate: Evaluate SINGLE math expressions (e.g., (356400/42.195)/(2+1/60+9/3600))\n"
    "- code_exec: Execute multi-step Python code with variables in a sandbox\n"
    "- code_generate: Generate Python code for a task\n"
    "\n**Vision:**\n"
    "- vision_analyze: Analyze images\n"
    "- vision_ocr: Extract text from images\n"
    "\n## Important:\n"
    "- Use expand_search=True only when you need more options or are confused\n"
    "- When you have the final answer, respond directly without calling tools\n"
    "\n## Final Answer Format:\n"
    "Report your thoughts, and finish your answer with: FINAL ANSWER: [YOUR FINAL ANSWER]\n"
    "\n**Formatting Rules:**\n"
    "- YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list\n"
    "- For numbers: Don't use commas or units (like $ or %) unless specified\n"
    "- For strings: Don't use articles or abbreviations, write digits in plain text\n"
    "- Pay attention to the question's units (e.g., 'how many thousand hours' means answer in thousands)\n"
    "- For lists: Apply above rules per element\n"
)


# Cache for models (initialized on first use, after .env is loaded)
_actor_model = None
_spec_model = None


def get_actor_model():
    """Get the actor model with tools bound (lazy initialization)."""
    global _actor_model
    if _actor_model is None:
        from .tools_langchain import ALL_TOOLS
        
        model = init_chat_model(config.actor_model, model_provider="openai")
        _actor_model = model.bind_tools(ALL_TOOLS)
    return _actor_model


def get_spec_model():
    """Get the speculator model with tools bound (lazy initialization)."""
    global _spec_model
    if _spec_model is None:
        from .tools_langchain import READ_ONLY_TOOLS
        
        model = init_chat_model(config.spec_model, model_provider="openai")
        _spec_model = model.bind_tools(READ_ONLY_TOOLS)
    return _spec_model


def convert_msg_to_langchain(messages: List[Msg]) -> List:
    """Convert our Msg objects to LangChain messages."""
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        elif m.role == "tool":
            # Convert tool results to user messages for simplicity
            tool_result = f"[TOOL: {m.name}]\n{m.content}"
            lc_messages.append(HumanMessage(content=tool_result))
    return lc_messages
