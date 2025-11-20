"""LLM adapter using proper LangGraph tool calling pattern."""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import Msg
from .config import config


# System prompt for OpenAI models
SYSTEM_PROMPT_OPENAI = (
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

# System prompt for open-source models (vLLM) - more explicit about tool usage
SYSTEM_PROMPT_VLLM = (
    "You are an AI assistant that solves questions by using tools. You MUST use tools to gather information.\n"
    "\n## 🔧 CRITICAL: USE TOOLS FREQUENTLY!\n"
    "- DO NOT try to answer from memory or reasoning alone\n"
    "- ALWAYS use search_with_content to look up facts, data, and information\n"
    "- Use multiple tool calls if needed to gather all required information\n"
    "- Use code_exec for calculations and data processing\n"
    "\n## Available Tools:\n"
    "\n**Search & Web (USE THIS OFTEN!):**\n"
    "- search_with_content: Search with full page content - USE THIS to find facts and data\n"
    "\n**Files:**\n"
    "- file_read: Read files (CSV, XLSX, PDF, DOCX, TXT, JSON, YAML, XML, HTML)\n"
    "\n**Computation:**\n"
    "- calculate: Evaluate math expressions like (356400/42.195)/(2+1/60+9/3600)\n"
    "- code_exec: Execute Python code for complex calculations\n"
    "- code_generate: Generate Python code\n"
    "\n**Vision:**\n"
    "- vision_analyze: Analyze images\n"
    "- vision_ocr: Extract text from images\n"
    "\n## Step-by-Step Process:\n"
    "1. Read the question carefully\n"
    "2. Identify what information you need\n"
    "3. USE TOOLS to gather that information (search, calculate, read files, etc.)\n"
    "4. If you need more information, USE MORE TOOLS\n"
    "5. Only after gathering ALL needed information, provide your final answer\n"
    "\n## Final Answer Format:\n"
    "When ready to answer, respond with: FINAL ANSWER: [YOUR FINAL ANSWER]\n"
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


def get_system_prompt() -> str:
    """Get the appropriate system prompt based on model provider."""
    if config.model_provider == "vllm":
        return SYSTEM_PROMPT_VLLM
    return SYSTEM_PROMPT_OPENAI


# For backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_OPENAI


def _create_model(model_name: str):
    """Create a chat model based on configured provider."""
    if config.model_provider == "vllm":
        # Use vLLM via OpenAI-compatible API
        return ChatOpenAI(
            model=model_name,
            openai_api_key=config.vllm_api_key,
            openai_api_base=config.vllm_base_url,
            temperature=0,
            max_tokens=config.llm_max_tokens,
        )
    else:
        # Use OpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=config.llm_max_tokens,
        )


def get_actor_model():
    """Get the actor model with tools bound (lazy initialization)."""
    global _actor_model
    if _actor_model is None:
        from .tools_langchain import ALL_TOOLS
        
        model = _create_model(config.actor_model)
        _actor_model = model.bind_tools(ALL_TOOLS)
    return _actor_model


def get_spec_model():
    """Get the speculator model with tools bound (lazy initialization)."""
    global _spec_model
    if _spec_model is None:
        from .tools_langchain import READ_ONLY_TOOLS
        
        model = _create_model(config.spec_model)
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
