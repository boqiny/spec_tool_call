#!/usr/bin/env python3
"""Evaluation script for GAIA examples."""
import asyncio
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from spec_tool_call import build_graph, Msg
from spec_tool_call.models import RunState
from spec_tool_call.config import config

load_dotenv()


async def run_example(example_dir: Path, app=None, verbose=True):
    """Run a single GAIA example."""
    
    # Build graph if not provided
    if app is None:
        app = build_graph()
    
    # Load metadata
    metadata_path = example_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    task_id = metadata.get("task_id", "unknown")
    question = metadata.get("question", "")
    final_answer = metadata.get("final_answer", "")
    level = metadata.get("level", "")
    
    if verbose:
        print("=" * 80)
        print(f"GAIA Example: {task_id} (Level {level})")
        print("=" * 80)
        print(f"\nQuestion:\n{question}\n")
        print(f"Ground Truth: {final_answer}")
        print("\n" + "=" * 80)
        print(f"Actor Model: {config.actor_model}")
        if config.enable_speculation:
            print(f"Spec Model:  {config.spec_model}")
            print(f"Speculation: ENABLED")
        else:
            print(f"Speculation: DISABLED")
        print(f"Max Steps:   {config.max_steps}")
        print("=" * 80 + "\n")
    
    # Build initial state
    init_state = RunState(messages=[Msg(role="user", content=question)])
    
    start_time = time.time()
    final_state = None
    
    try:
        async for event in app.astream(
            init_state,
            config={"configurable": {"thread_id": f"eval-{task_id}"}}
        ):
            for node_name, state in event.items():
                final_state = state
                
                # Extract state info
                if isinstance(state, dict):
                    messages = state.get('messages', [])
                    done = state.get('done', False)
                    answer = state.get('answer', None)
                    pending_tools = state.get('pending_tool_calls', [])
                    llm_time = state.get('last_llm_time', 0)
                    tool_time = state.get('last_tool_time', 0)
                    step = state.get('step', 0)
                else:
                    messages = state.messages
                    done = state.done
                    answer = state.answer
                    pending_tools = state.pending_tool_calls
                    llm_time = state.last_llm_time
                    tool_time = state.last_tool_time
                    step = state.step
                
                # Show LLM step (only if verbose)
                if verbose and node_name == "llm":
                    print(f"\n{'='*80}")
                    print(f"[Step {step}] LLM")
                    print(f"{'='*80}")
                    print(f"⏱️  LLM call: {llm_time:.2f}s\n")
                    
                    # Show LLM's text content if it exists
                    if messages and messages[-1].role == "assistant":
                        content = messages[-1].content
                        # Filter out placeholder text
                        if content and content.strip() and content.strip() not in ["[Calling tools]", "[Tool call]"]:
                            print(f"💭 Reasoning:")
                            if len(content) > 300:
                                print(f"   {content[:300]}...")
                            else:
                                print(f"   {content}")
                            print()
                    
                    # Show tool calls if any
                    if pending_tools:
                        print(f"🔧 Tool Call:")
                        for tc in pending_tools:
                            print(f"   Tool: {tc['name']}")
                            print(f"   Args:")
                            for k, v in tc['args'].items():
                                if isinstance(v, str) and len(v) > 80:
                                    v_display = v[:80] + "..."
                                else:
                                    v_display = v
                                print(f"      {k} = {v_display}")
                    
                    # Show final answer if done
                    if done:
                        print(f"✅ Final Answer: {answer}")
                
                # Show tool execution (only if verbose)
                elif verbose and node_name == "tools":
                    print(f"\n{'='*80}")
                    print(f"[Step {step}] TOOLS")
                    print(f"{'='*80}")
                    print(f"⏱️  Execution: {tool_time:.2f}s\n")
                    
                    tool_msgs = [m for m in messages if m.role == "tool"]
                    if tool_msgs:
                        latest_tool = tool_msgs[-1]
                        tool_name = latest_tool.name if hasattr(latest_tool, 'name') else "unknown"
                        result = latest_tool.content
                        
                        print(f"📤 Output from '{tool_name}':")
                        print("-" * 80)
                        
                        if "Error:" in result:
                            print(f"   ❌ {result}")
                        else:
                            lines = result.split('\n')
                            if len(lines) > 10:
                                preview = '\n'.join(lines[:10])
                                print(f"   {preview}")
                                print(f"   ... ({len(lines)-10} more lines)")
                            elif len(result) > 500:
                                print(f"   {result[:500]}")
                                print(f"   ... (truncated, {len(result)} total chars)")
                            else:
                                print(f"   {result}")
                        print("-" * 80)
                
                if done:
                    print(f"\n{'='*80}")
                    print("✅ EXECUTION COMPLETE")
                    print(f"{'='*80}")
                    break
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    elapsed = time.time() - start_time
    
    # Extract final results
    if final_state:
        if isinstance(final_state, dict):
            predicted = final_state.get('answer', None)
            hits = final_state.get('hits', 0)
            misses = final_state.get('misses', 0)
            spec_launched = final_state.get('speculative_launched', 0)
            steps = final_state.get('step', 0)
        else:
            predicted = final_state.answer
            hits = final_state.hits
            misses = final_state.misses
            spec_launched = final_state.speculative_launched
            steps = final_state.step
        
        correct = False
        if predicted and final_answer:
            correct = predicted.lower().strip() == final_answer.lower().strip()
        
        if verbose:
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)
            print(f"\nPredicted: {predicted if predicted else 'No answer'}")
            print(f"Truth:     {final_answer}")
            print(f"\nCorrect:   {'✓ YES' if correct else '✗ NO'}")
            print(f"Steps:     {steps}")
            print(f"Time:      {elapsed:.1f}s")
            
            if config.enable_speculation:
                print("\n" + "=" * 80)
                print("SPECULATION")
                print("=" * 80)
                print(f"Hits:        {hits}")
                print(f"Misses:      {misses}")
                print(f"Predictions: {spec_launched}")
                total_checks = hits + misses
                if total_checks > 0:
                    hit_rate = hits / total_checks * 100
                    print(f"Hit Rate:    {hit_rate:.1f}%")
            
            print("\n" + "=" * 80 + "\n")
        
        # Save results to JSON
        spec_status = "spec" if config.enable_speculation else "baseline"
        result_file = f"result_{task_id}_{spec_status}.json"
        
        result_data = {
            "task_id": task_id,
            "level": level,
            "question": question,
            "ground_truth": final_answer,
            "predicted": predicted,
            "correct": correct,
            "steps": steps,
            "time_seconds": elapsed,
            "actor_model": config.actor_model,
            "spec_model": config.spec_model if config.enable_speculation else None,
            "speculation_enabled": config.enable_speculation,
            "spec_hits": hits if config.enable_speculation else 0,
            "spec_misses": misses if config.enable_speculation else 0,
            "spec_predictions": spec_launched if config.enable_speculation else 0,
        }
        
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        
        if verbose:
            print(f"💾 Results saved to: {result_file}\n")
        
        return result_data


async def run_batch(level_dir: Path, output_dir: Path, max_examples: int = None):
    """Run batch evaluation on all examples in a directory."""
    
    # Find all examples
    examples = sorted([
        d for d in level_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])
    
    if not examples:
        print(f"No examples found in {level_dir}")
        return
    
    # Limit to max_examples if specified
    if max_examples:
        examples = examples[:max_examples]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spec_status = "spec" if config.enable_speculation else "baseline"
    print("=" * 80)
    print(f"BATCH EVALUATION: {level_dir.name}")
    print("=" * 80)
    print(f"Examples:    {len(examples)}")
    print(f"Max Steps:   {config.max_steps}")
    print(f"Speculation: {'ENABLED' if config.enable_speculation else 'DISABLED (baseline)'}")
    print(f"Output Dir:  {output_dir}")
    print("=" * 80 + "\n")
    
    # Build graph once
    app = build_graph()
    
    count = 0
    correct = 0
    total_time = 0
    
    for i, example_dir in enumerate(examples, 1):
        example_name = example_dir.name
        count += 1
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(examples)}] {example_name}")
        print(f"{'='*80}\n")
        
        try:
            result = await run_example(example_dir, app, verbose=False)
            
            # Move result file to output directory
            result_files = list(Path(".").glob(f"result_*_{spec_status}.json"))
            if result_files:
                result_file = result_files[0]
                dest = output_dir / result_file.name
                result_file.rename(dest)
                
                # Read for stats
                with open(dest) as f:
                    data = json.load(f)
                
                is_correct = data.get("correct", False)
                time_taken = data.get("time_seconds", 0)
                
                if is_correct:
                    correct += 1
                    print(f"\n✅ CORRECT ({time_taken:.1f}s)")
                else:
                    pred = data.get("predicted", "")
                    truth = data.get("ground_truth", "")
                    print(f"\n❌ WRONG ({time_taken:.1f}s)")
                    print(f"   Predicted: {pred}")
                    print(f"   Truth:     {truth}")
                
                total_time += time_taken
            else:
                print(f"\n❌ ERROR: No result file found")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
    
    # Final summary
    accuracy = correct / count * 100 if count > 0 else 0
    avg_time = total_time / count if count > 0 else 0
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {correct}/{count} ({accuracy:.1f}%)")
    print(f"Avg Time:  {avg_time:.1f}s")
    print(f"Total:     {total_time:.1f}s")
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GAIA examples")
    parser.add_argument("example", nargs="?", help="Path to example directory (for single mode)")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                       help="GAIA level for batch mode (default: 1)")
    parser.add_argument("--max", type=int, help="Maximum number of examples to run (default: all)")
    parser.add_argument("--output", help="Output directory for batch results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        level_dir = Path(f"gaia_dataset/level{args.level}")
        if not level_dir.exists():
            print(f"Error: {level_dir} not found")
            sys.exit(1)
        
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            spec_status = "spec" if config.enable_speculation else "baseline"
            output_dir = Path(f"results_level{args.level}_{spec_status}_{timestamp}")
        
        asyncio.run(run_batch(level_dir, output_dir, max_examples=args.max))
    
    # Single example mode
    else:
        if args.example:
            example_dir = Path(args.example)
        else:
            example_dir = Path("gaia_dataset/level1/example_000")
        
        if not example_dir.exists():
            print(f"Error: {example_dir} not found")
            sys.exit(1)
        
        asyncio.run(run_example(example_dir))


if __name__ == "__main__":
    main()
