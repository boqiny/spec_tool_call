# Speculative Tool Calling

Reduce LLM agent latency by speculatively pre-executing tool calls in parallel with the actor model's reasoning.

## Core Idea

```
Traditional Sequential Execution:
┌─────────────────────────────────────────────┐
│ Actor thinks (10s) → Execute tool (2s)     │
│ Total: 12s                                  │
└─────────────────────────────────────────────┘

Speculative Parallel Execution:
┌─────────────────────────────────────────────┐
│ Actor:  🧠 Thinking... (10s)                │
│ Spec:   🧠 Predict (2s) → 🔧 Execute (2s)  │
│         └→ 💾 Cache result                  │
│                                             │
│ If match: Use cached result ⚡ (0.001s)     │
│ Total: ~10s (2s saved)                      │
└─────────────────────────────────────────────┘
```

**Key insight**: While the actor model (GPT-5) is thinking, a lightweight spec model (GPT-5-mini) predicts and pre-executes the next tool call. If the prediction matches, we use the cached result instead of executing again.

## Architecture

```
                    ┌─────────────────────┐
                    │      START          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     node_llm        │
                    │                     │
                    │  ┌───────────────┐  │
                    │  │ Actor Model   │  │  Run in parallel
                    │  │ (GPT-5)       │  │  
                    │  └───────────────┘  │
                    │         +           │
                    │  ┌───────────────┐  │
                    │  │ Spec Model    │  │
                    │  │ (GPT-5-mini)  │  │
                    │  │ + Pre-execute │  │
                    │  │ + Cache       │  │
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
              ┌────→│    node_tools       │
              │     │                     │
              │     │  Check cache:       │
              │     │  - Match? Use cache │
              │     │  - No? Execute      │
              │     └──────────┬──────────┘
              │                │
              │                ▼
              │     ┌─────────────────────┐
              │     │   should_end?       │
              │     └──────────┬──────────┘
              │                │
              │         ┌──────┴──────┐
              │         │             │
              │       Done       Continue
              │         │             │
              │         ▼             │
              │     ┌───────┐         │
              │     │  END  │         │
              │     └───────┘         │
              │                       │
              └───────────────────────┘
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"
export SERPER_API_KEY="your-key"  # For web search

# Download GAIA dataset
python download_gaia.py
```

## Running Evaluations

### Single Example (Detailed)

```bash
# Run specific example
python eval.py gaia_dataset/level1/example_000

# With speculation enabled (default)
python eval.py gaia_dataset/level1/example_000

# Baseline (no speculation)
DISABLE_SPECULATION=1 python eval.py gaia_dataset/level1/example_000
```

### Batch Evaluation (Multiple Examples)

```bash
# Run all level 1 examples with speculation
python eval.py --batch --level 1

# Run first 10 examples
python eval.py --batch --level 1 --max 10

# Baseline (no speculation)
DISABLE_SPECULATION=1 python eval.py --batch --level 1 --max 10

# Custom output directory
python eval.py --batch --level 1 --output my_results/
```

## Configuration

Set via environment variables:

```bash
# Models
export GAIA_ACTOR_MODEL="gpt-5"           # Main reasoning model
export GAIA_SPEC_MODEL="gpt-5-mini"       # Speculation model

# Limits
export GAIA_MAX_STEPS="15"                # Max reasoning steps

# Speculation
export DISABLE_SPECULATION="1"            # Disable for baseline
```

## Output

**Single mode** shows detailed step-by-step execution:
```
[Step 1] LLM
⏱️  LLM call: 8.5s

🔧 Tool Call:
   Tool: search_with_content
   Args: query = Moon perigee...

[Step 1] TOOLS
⏱️  Execution: 0.001s

✅ USING CACHED RESULT (pre-executed by spec model)
   Saved 1.25s from cache
```

**Batch mode** shows compact progress:
```
[1/10] example_000
✅ CORRECT (89.2s)

[2/10] example_001
❌ WRONG (102.3s)

...

FINAL RESULTS
Accuracy:  8/10 (80.0%)
Avg Time:  95.4s
```

## Results

Results are saved to JSON files:

```json
{
  "task_id": "e1fc63a2-...",
  "question": "If Eliud Kipchoge could maintain...",
  "ground_truth": "17",
  "predicted": "17",
  "correct": true,
  "steps": 6,
  "time_seconds": 89.2,
  "spec_hits": 5,
  "spec_misses": 0,
  "spec_predictions": 5
}
```