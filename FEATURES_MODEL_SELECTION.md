# ShipAI Model Selection: User Suggestions + Gemini Reasoning

## Overview

ShipAI now supports **intelligent model selection** for your hardware with two enhancement features:

1. **User Model Suggestions** — Override automatic model selection with your preferences
2. **Gemini Reasoning** — Get AI-powered explanations and validation of model choices

Both features work **alongside** the deterministic scoring system (they don't replace it).

---

## Architecture

```
Setup Fleet runs in 5 phases:

Scout              → Detect hardware, runtimes, inventory
    ↓
Librarian          → Load capability matrix + catalog
    ↓
Negotiator         → (ENHANCED) Assign models + apply suggestions + Gemini reasoning
    ↓
Acquirer           → Download missing models
    ↓
Validator          → Re-scan + finalize model plan
```

### Negotiator Enhancement

```python
# Before: Deterministic scoring only
negotiator = ModelNegotiator(hardware, matrix, catalog)
plan = negotiator.negotiate()

# After: Supports suggestions + Gemini
negotiator = EnhancedNegotiator(hardware, matrix, catalog, use_gemini=True)
negotiator.set_user_suggestions({"interview_node": "qwen3:4b"})
plan = await negotiator.negotiate()
```

---

## Configuration

### Enable Gemini API

Set your Gemini API key:

```bash
# Via environment variable
export SHIPAI_GEMINI_API_KEY="AIzaSyD1WlVkiZCRn11GSZgJvzZ4nauCl8Jqp9g"

# Or in .env file
echo "SHIPAI_GEMINI_API_KEY=AIzaSyD1WlVkiZCRn11GSZgJvzZ4nauCl8Jqp9g" >> .env
```

### Optional: Enable Gemini by Default

```bash
# Via environment variable
export SHIPAI_GEMINI_ENABLE_REASONING=true

# Or in .env
echo "SHIPAI_GEMINI_ENABLE_REASONING=true" >> .env
```

---

## Usage

### 1. CLI: User Model Suggestions

Override specific models before negotiation:

```bash
# Suggest models for specific nodes
python -m shipai plan interview_node:qwen3:4b research_node:deepseek-r1:7b

# With Gemini explanation
python -m shipai plan --explain interview_node:qwen3:4b

# Both
python -m shipai plan --explain interview_node:qwen3:4b research_node:deepseek-r1:7b
```

### 2. CLI: Interactive Model Override

Override models after initial planning:

```bash
# Show current assignments and allow interactive override
python -m shipai suggest

# Example output:
# Current Model Assignments:
# -----------
#   interview_node      qwen2.5:7b        (confidence: matrix_estimate)
#   research_node       deepseek-r1:7b    (confidence: matrix_estimate)
#   
# Override any models (or press Enter to skip):
#   interview_node (currently qwen2.5:7b): qwen3:4b
#   research_node (currently deepseek-r1:7b): 
#   
# Validate with Gemini? (y/n): y
#
# Re-negotiating with suggestions...
# Updated Model Assignments:
# -----------
#   interview_node      qwen3:4b          [USER]
#   research_node       deepseek-r1:7b    
#   
# Plan saved.
```

### 3. API: Get Current Plan

```bash
curl http://localhost:8000/api/models/current-plan
```

Response:
```json
{
  "primary_runtime": "ollama",
  "embedding": "nomic-embed-text",
  "node_assignments": {
    "interview_node": {
      "model": "qwen2.5:7b",
      "status": "installed",
      "confidence": "installed_verified",
      "reason": "Installed model matched capability matrix",
      "gemini_reasoning": "For your 16GB RAM system, qwen2.5:7b offers good balance...",
      "user_suggested": false
    }
  },
  "user_suggestions": {},
  "warnings": [],
  "gemini_enabled": true,
  "generated_at": "2026-05-24T10:30:45Z"
}
```

### 4. API: Suggest a Single Model

```bash
curl -X POST http://localhost:8000/api/models/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "node": "interview_node",
    "model": "qwen3:4b",
    "reason": "Prefer smaller, faster model for my setup"
  }'
```

Response:
```json
{
  "suggestion": {
    "node": "interview_node",
    "model": "qwen3:4b",
    "reason": "Prefer smaller, faster model for my setup"
  },
  "validation": {
    "valid": true,
    "reasoning": "qwen3:4b fits well on your hardware. It has good json_reliable score (0.88) and fast latency.",
    "concerns": []
  },
  "status": "valid"
}
```

### 5. API: Get Gemini Explanation

```bash
curl -X POST http://localhost:8000/api/models/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "node": "interview_node"
  }'
```

Response:
```json
{
  "model": "qwen2.5:7b",
  "node": "interview_node",
  "explanation": "qwen2.5:7b is ideal for your 16GB RAM system. It balances quality (json_reliable: 0.92) with reasonable latency. Alternatives like qwen2.5:14b would require more VRAM.",
  "confidence": 0.85,
  "concerns": []
}
```

### 6. API: Replan With Suggestions

```bash
curl -X POST http://localhost:8000/api/models/replan-with-suggestions \
  -H "Content-Type: application/json" \
  -d '{
    "suggestions": {
      "interview_node": "qwen3:4b",
      "research_node": "deepseek-r1:7b"
    },
    "validate_with_gemini": true
  }'
```

Response:
```json
{
  "status": "success",
  "suggestions_applied": 2,
  "node_assignments": {
    "interview_node": {
      "model": "qwen3:4b",
      "user_suggested": true,
      "reason": "User suggested: qwen3:4b"
    },
    "research_node": {
      "model": "deepseek-r1:7b",
      "user_suggested": true,
      "reason": "User suggested: deepseek-r1:7b"
    }
  },
  "warnings": []
}
```

---

## Model Plan Structure

The saved `~/.shipai/model_plan.json` now includes:

```json
{
  "schema_version": "1",
  "generated_at": "2026-05-24T10:30:45Z",
  "primary_runtime": "ollama",
  "node_assignments": {
    "interview_node": {
      "model": "qwen3:4b",
      "runtime": "ollama",
      "status": "installed",
      "confidence": "installed_verified",
      "quality_score": 0.88,
      "reason": "Installed model matched capability matrix",
      "gemini_reasoning": "For your 16GB RAM system...",
      "user_suggested": true
    }
  },
  "user_suggestions": {
    "interview_node": "qwen3:4b"
  },
  "gemini_enabled": true,
  "warnings": []
}
```

---

## How It Works

### Phase 1: Deterministic Scoring (Always Runs)

```
Hardware: 16GB RAM, RTX 4090
         ↓
Score each model:
  qwen3:4b    → score: 0.88 (good fit)
  qwen2.5:7b  → score: 0.92 (better fit)
  qwen2.5:14b → score: 0.95 (best fit but uses more VRAM)
         ↓
Pick highest score → qwen2.5:7b ✅
```

### Phase 2: User Suggestions (Optional)

```
User says: "I prefer qwen3:4b"
         ↓
Validate against hardware & capabilities
         ↓
If valid → Apply suggestion
If invalid → Reject + warn
```

### Phase 3: Gemini Reasoning (Optional, Requires API Key)

```
"Why did you pick qwen2.5:7b for interview_node?"
         ↓
Gemini analyzes:
  - Hardware specs
  - Model capabilities
  - Alternatives
         ↓
Returns explanation:
"qwen2.5:7b is ideal because it balances quality
with performance on your hardware..."
```

---

## Data Flow Example

```
User runs: python -m shipai plan --explain interview_node:qwen3:4b

         ↓

Scout finds: 16GB RAM, RTX 4090, Ollama running

         ↓

Librarian loads capability matrix

         ↓

Negotiator:
  1. Deterministic scoring
     → Picks qwen2.5:7b (highest score)
  
  2. Apply user suggestion
     → User wants qwen3:4b
     → Validate: OK (fits hardware)
     → Override to qwen3:4b
  
  3. Gemini reasoning
     → "Why qwen3:4b?"
     → Gemini: "It's fast, fits VRAM, good json_reliable score"
     → Add to model_plan.json

         ↓

Acquirer: Models already installed, skip pull

         ↓

Validator: Re-scan + save final plan with user suggestion + reasoning

         ↓

Result: ~/.shipai/model_plan.json updated with:
  - Selected: qwen3:4b (user_suggested: true)
  - Reasoning: "Fast, good balance"
```

---

## When to Use Each Feature

| Scenario | Feature | Example |
|----------|---------|---------|
| **Automatic best-fit** | None | `python -m shipai install` |
| **Want explanations** | `--explain` | `python -m shipai plan --explain` |
| **Prefer specific model** | Suggestions | `python -m shipai plan interview_node:qwen3:4b` |
| **Both** | Suggestions + Explain | `python -m shipai plan --explain interview_node:qwen3:4b` |
| **Interactive override** | `suggest` command | `python -m shipai suggest` |

---

## Important Notes

### ✅ What Works

- Deterministic scoring: **Always works**, no dependencies
- User suggestions: **Works with or without Gemini** (Gemini adds validation)
- Gemini reasoning: **Optional, enhances confidence**
- All data persists in `~/.shipai/model_plan.json`

### ⚠️ Limitations

- Gemini requires API key and internet connection
- Gemini adds ~1-2 seconds latency per explanation
- User can override any model (no hard constraints enforced)

### 🔒 Privacy

- Hardware specs sent to Gemini only if `--explain` is used
- Model names and hardware data are not logged
- Gemini responses are not stored server-side

---

## Troubleshooting

### "Gemini API key not configured"

```bash
export SHIPAI_GEMINI_API_KEY="your-key-here"
python -m shipai plan --explain
```

### "User suggestion invalid for hardware"

Check warnings:
```bash
python -m shipai plan interview_node:very_large_model
# Output: "User suggestion invalid: model too large for available VRAM"
```

Fallback to automatic selection:
```bash
python -m shipai plan  # No suggestions → uses best-fit automatically
```

### "Model not found in capability matrix"

The model exists locally but isn't in the matrix. It will still work but with lower confidence:

```json
"confidence": "installed_unknown"
```

---

## Environment Variables

```bash
# Required for Gemini reasoning
SHIPAI_GEMINI_API_KEY=AIzaSyD1WlVkiZCRn11GSZgJvzZ4nauCl8Jqp9g

# Optional: Always use Gemini when available
SHIPAI_GEMINI_ENABLE_REASONING=true

# Optional: Which Gemini model to use (default: gemini-flash-latest)
SHIPAI_GEMINI_MODEL=gemini-2.0-flash-exp
```

---

## Next Steps

1. ✅ **Set your Gemini API key**
2. ✅ **Try automatic planning**: `python -m shipai plan`
3. ✅ **Try with suggestions**: `python -m shipai plan --explain interview_node:qwen3:4b`
4. ✅ **Try interactive override**: `python -m shipai suggest`
5. ✅ **Check model plan**: `cat ~/.shipai/model_plan.json`
