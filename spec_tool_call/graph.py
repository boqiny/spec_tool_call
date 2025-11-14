"""LangGraph nodes and graph construction using proper tool calling."""
import asyncio
import time
from typing import Literal, List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .models import RunState, Msg
from .llm_adapter import get_actor_model, get_spec_model, convert_msg_to_langchain, SYSTEM_PROMPT
from .tools_langchain import TOOLS_BY_NAME
from .config import config

# -----------------------------
# Speculation Helpers
# -----------------------------

def calculate_similarity(args1: Dict[str, Any], args2: Dict[str, Any]) -> float:
    """Calculate similarity between two argument dictionaries.
    TODO: Add sophisticated matching logic
    """
    return 1.0


async def launch_speculation(state: RunState) -> None:
    """Launch spec model to predict and PRE-EXECUTE next tool call in parallel.
    
    Simple 1-step speculation with pre-execution:
    1. Spec model predicts next tool call
    2. IMMEDIATELY execute the tool with predicted args
    3. Cache both prediction and result
    
    This runs in parallel while actor is thinking.
    """
    if not config.enable_speculation:
        return
    
    spec_model = get_spec_model()
    
    # Convert messages to LangChain format
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + convert_msg_to_langchain(state.messages)
    
    # Call spec model to predict next tool call
    try:
        response = spec_model.invoke(lc_messages)
        
        # Get the first tool call prediction
        if hasattr(response, 'tool_calls') and response.tool_calls:
            prediction = response.tool_calls[0]
            tool_name = prediction["name"]
            tool_args = prediction["args"]
            
            # PRE-EXECUTE the tool with predicted args
            if tool_name in TOOLS_BY_NAME:
                tool = TOOLS_BY_NAME[tool_name]
                try:
                    exec_start = time.time()
                    result = await asyncio.to_thread(tool.invoke, tool_args)
                    exec_time = time.time() - exec_start
                    
                    # Store prediction WITH pre-executed result
                    state.speculative_predictions = [{
                        "name": tool_name,
                        "args": tool_args,
                        "result": str(result),
                        "exec_time": exec_time
                    }]
                    state.speculative_launched += 1
                    
                except Exception as e:
                    # Tool execution failed - store prediction without result
                    state.speculative_predictions = [{
                        "name": tool_name,
                        "args": tool_args,
                        "result": None,
                        "error": str(e),
                        "exec_time": 0
                    }]
            else:
                # Unknown tool - just store prediction
                state.speculative_predictions = [{
                    "name": tool_name,
                    "args": tool_args,
                    "result": None,
                    "exec_time": 0
                }]
        else:
            state.speculative_predictions = []
            
    except Exception as e:
        print(f"⚠️  Speculation error: {e}")
        state.speculative_predictions = []


def find_best_match(
    actor_tool_call: Dict[str, Any],
    spec_predictions: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find matching speculative prediction for actor's tool call.
    
    Simple placeholder: just matches tool names for now.
    TODO: Add sophisticated arg matching.
    
    Args:
        actor_tool_call: {"name": str, "args": dict}
        spec_predictions: List of predicted tool calls
    
    Returns:
        First prediction with matching tool name, or None
    """
    if not spec_predictions:
        return None
    
    actor_name = actor_tool_call["name"]
    
    # Just find first prediction with matching tool name
    for spec_pred in spec_predictions:
        if spec_pred["name"] == actor_name:
            return spec_pred
    
    return None


# -----------------------------
# Graph nodes - ReAct Pattern
# -----------------------------

async def node_llm(state: RunState) -> RunState:
    """
    LLM node: Decides what to do next (Reason + Act).
    Either calls tools or provides final answer.
    
    Runs actor and spec model IN PARALLEL for 1-step speculation.
    """
    if state.done:
        return state
    
    # Increment step at the start (this is LLM call #N)
    state.step += 1
    
    # Check if max steps reached - force final answer
    if state.step > config.max_steps:
        print(f"\n⚠️  MAX STEPS REACHED ({config.max_steps}). Forcing final answer...")
        
        # Get model WITHOUT tools to force text response
        from langchain.chat_models import init_chat_model
        model_no_tools = init_chat_model(config.actor_model, model_provider="openai")
        
        # Build messages with conversation history (formatting rules already in SYSTEM_PROMPT)
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + convert_msg_to_langchain(state.messages)
        lc_messages.append(HumanMessage(content=(
            "You have reached the maximum number of steps. Based on all the information gathered above, "
            "provide your best final answer now. Remember to follow the formatting rules in the system prompt."
        )))
        
        start_time = time.time()
        response = model_no_tools.invoke(lc_messages)
        elapsed = time.time() - start_time
        
        state.last_llm_time = elapsed
        
        # Extract answer
        content = response.content
        if "FINAL ANSWER:" in content.upper():
            answer_part = content.split("FINAL ANSWER:")[-1] if "FINAL ANSWER:" in content else content
            answer = answer_part.strip()
        else:
            # Use entire content as answer
            answer = content.strip()
        
        state.messages.append(Msg(role="assistant", content=content))
        state.answer = answer
        state.done = True
        
        return state
    
    # Get model with tools
    model = get_actor_model()
    
    # Convert messages to LangChain format
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + convert_msg_to_langchain(state.messages)
    
    # ⚡ PARALLEL EXECUTION: Run actor and spec model at the same time
    start_time = time.time()
    
    # Create tasks for parallel execution
    actor_task = asyncio.create_task(asyncio.to_thread(model.invoke, lc_messages))
    spec_task = asyncio.create_task(launch_speculation(state))
    
    # Wait for both to complete
    response, _ = await asyncio.gather(actor_task, spec_task)
    
    elapsed = time.time() - start_time
    
    # Store timing in state
    state.last_llm_time = elapsed
    
    # Check if final answer (no tool calls)
    if not response.tool_calls:
        content = response.content
        
        # Check if this is a final answer
        if "FINAL ANSWER:" in content.upper():
            # Extract answer
            answer_part = content.split("FINAL ANSWER:")[-1] if "FINAL ANSWER:" in content else content
            answer = answer_part.strip()
            
            state.messages.append(Msg(role="assistant", content=content))
            state.answer = answer
            state.done = True
        else:
            # Just thinking/reasoning
            state.messages.append(Msg(role="assistant", content=content))
        
        return state
    
    # Has tool calls - store them for execution
    # Store the AI message with metadata about tool calls
    ai_msg_content = response.content or "[Calling tools]"
    state.messages.append(Msg(role="assistant", content=ai_msg_content))
    
    # Store tool calls in state for the tool node to execute
    state.pending_tool_calls = response.tool_calls
    
    return state


async def node_tools(state: RunState) -> RunState:
    """
    Tools node: Executes pending tool calls (Observe).
    Returns results to LLM.
    
    Checks speculation cache first - if match found, uses predicted args.
    Otherwise executes with actor's args.
    """
    if state.done:
        return state
        
    if not hasattr(state, 'pending_tool_calls') or not state.pending_tool_calls:
        return state
    
    # Execute each tool call
    for tool_call in state.pending_tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Check if we have speculative predictions from previous step
        spec_predictions = getattr(state, 'speculative_predictions', [])
        
        # Only show cache matching if speculation is enabled
        if config.enable_speculation:
            print(f"\n🔍 CACHE MATCHING: Checking {len(spec_predictions)} prediction(s)")
            print("=" * 80)
        
        # Show all predictions and their match scores
        if spec_predictions and config.enable_speculation:
            for i, spec_pred in enumerate(spec_predictions, 1):
                spec_name = spec_pred["name"]
                spec_args = spec_pred["args"]
                
                # Check tool name match
                tool_match = (spec_name == tool_name)
                
                # Calculate similarity
                similarity = calculate_similarity(tool_args, spec_args)
                
                # Determine status
                has_cached_result = spec_pred.get("result") is not None
                if tool_match and has_cached_result:
                    status = f"✅ MATCH (tool name matches, result cached)"
                elif tool_match and not has_cached_result:
                    status = f"⚠️  MATCH (tool name matches, but no cached result)"
                else:
                    status = f"❌ REJECTED (tool name mismatch: {spec_name} ≠ {tool_name})"
                
                print(f"Prediction #{i}: {spec_name}")
                print(f"  Status: {status}")
                
                if tool_match:
                    # Show arg differences and cache info
                    print(f"  Actor args:  {tool_args}")
                    print(f"  Spec args:   {spec_args}")
                    if has_cached_result:
                        exec_time = spec_pred.get("exec_time", 0)
                        print(f"  Cached result: ✓ (pre-executed in {exec_time:.2f}s)")
        elif config.enable_speculation:
            print("  No predictions available from previous step")
        
        if config.enable_speculation:
            print("=" * 80)
        
        # Try to find matching speculation (just tool name for now)
        best_match = find_best_match(
            {"name": tool_name, "args": tool_args},
            spec_predictions
        )
        
        if tool_name in TOOLS_BY_NAME:
            tool = TOOLS_BY_NAME[tool_name]
            try:
                start_time = time.time()
                
                if best_match and best_match.get("result") is not None:
                    # ⚡ CACHE HIT: Use pre-executed result!
                    if config.enable_speculation:
                        print(f"\n✅ USING CACHED RESULT (pre-executed by spec model)")
                        print(f"   Saved {best_match.get('exec_time', 0):.2f}s from cache")
                    result_str = best_match["result"]
                    state.hits += 1
                    elapsed = 0.001  # Negligible time to retrieve from cache
                else:
                    # No match or no cached result - execute with actor's args
                    if config.enable_speculation:
                        if best_match:
                            print(f"\n⚠️  CACHE MISS (prediction had no result, executing actor's args)")
                        else:
                            print(f"\n❌ NO MATCH (executing actor's args)")
                    result = await asyncio.to_thread(tool.invoke, tool_args)
                    result_str = str(result)
                    state.misses += 1
                    elapsed = time.time() - start_time
                
                # Store timing
                state.last_tool_time = elapsed
                
                state.messages.append(Msg(role="tool", name=tool_name, content=result_str))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
                state.misses += 1
        else:
            error_msg = f"Error: Unknown tool '{tool_name}'"
            state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
    
    # Clear pending tool calls and predictions
    state.pending_tool_calls = []
    state.speculative_predictions = []
    
    return state


def route_after_llm(state: RunState) -> Literal["tools", END]:
    """Route after LLM: to tools if there are tool calls, otherwise end."""
    if state.done:
        return END
    if hasattr(state, 'pending_tool_calls') and state.pending_tool_calls:
        return "tools"
    return END


def route_after_tools(state: RunState) -> Literal["llm", END]:
    """Route after tools: back to LLM for reasoning."""
    if state.done:
        return END
    # Always go back to LLM (even if max steps reached - LLM will be forced to answer)
    return "llm"


# -----------------------------
# Build graph
# -----------------------------

def build_graph():
    """Build and compile the LangGraph agent with ReAct pattern."""
    workflow = StateGraph(RunState)
    
    # Add nodes
    workflow.add_node("llm", node_llm)
    workflow.add_node("tools", node_tools)
    
    # Define edges
    workflow.add_edge(START, "llm")
    
    # LLM can go to tools or end
    workflow.add_conditional_edges(
        "llm",
        route_after_llm,
        {"tools": "tools", END: END}
    )
    
    # Tools always go back to LLM (for reasoning about results)
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"llm": "llm", END: END}
    )
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=MemorySaver())
