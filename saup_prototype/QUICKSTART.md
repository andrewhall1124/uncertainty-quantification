# SAUP Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd saup_prototype
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.template .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

### 3. Run Demo
```bash
python main.py
```

## What to Expect

The demo will:
1. Initialize LLM, agent, and SAUP components
2. Run 3 test questions through the ReAct agent
3. Calculate SAUP uncertainty for each trajectory
4. Display results with rich formatting
5. Show evaluation metrics (AUROC, accuracy)

**Estimated time**: 2-3 minutes
**Estimated cost**: $0.25-$0.50 (using GPT-4)

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║           SAUP Uncertainty Quantification System             ║
║        Situation-Aware Uncertainty Propagation v1.0          ║
╚══════════════════════════════════════════════════════════════╝

[●] Loading models...
  ├─ ✓ LLM Wrapper initialized (GPT-4)
  ├─ ✓ Semantic distance calculator loaded
  └─ ✓ HMM Proxy initialized

┌─────────────────────────────────────────────────────────────┐
│ Question: Can drinking coffee help with weight loss?        │
└─────────────────────────────────────────────────────────────┘

━━━ Step 1/7 ━━━

💭 Thought: Coffee contains caffeine...
🎯 Action: search[caffeine weight loss]
📊 Observation: Caffeine can boost metabolism...
📈 Uncertainty: 0.15 (Low) ✓
```

## Commands

| Command | Purpose |
|---------|---------|
| `python main.py` | Run full demo with 3 questions |
| `python examples.py` | See component examples |
| `python test_imports.py` | Verify installation |

## File Guide

| File | What It Does |
|------|-------------|
| [main.py](main.py) | Main demo - runs questions through SAUP |
| [examples.py](examples.py) | Component demos (no API key needed) |
| [README.md](README.md) | Full documentation |
| [INSTALL.md](INSTALL.md) | Detailed installation guide |

## Key Components

```python
# 1. Initialize system
from models.llm_wrapper import LLMWrapper
from agents.react_agent import ReactAgent
from uncertainty.propagation import SAUPPropagator

llm = LLMWrapper(model_name="gpt-4")
agent = ReactAgent(llm=llm, max_steps=7)
propagator = SAUPPropagator(use_hmm_proxy=True)

# 2. Run agent on question
result = agent.run("Your question here")

# 3. Propagate uncertainty
saup_result = propagator.propagate_uncertainty(
    question="Your question here",
    trajectory=result['trajectory']
)

# 4. Get uncertainty score
uncertainty = saup_result['agent_uncertainty']
confidence = (1 - uncertainty) * 100
```

## Customization

### Change Model
Edit `.env`:
```bash
MODEL_NAME=gpt-3.5-turbo  # Cheaper but less capable
```

### Change Questions
Edit `main.py` line ~130:
```python
test_questions = [
    ("Your custom question?", True),  # True/False = correct
]
```

### Adjust Max Steps
Edit `.env`:
```bash
MAX_STEPS=10  # Allow more reasoning steps
```

## Common Issues

| Issue | Solution |
|-------|----------|
| `No module named 'openai'` | Run `pip install -r requirements.txt` |
| `OPENAI_API_KEY not found` | Create `.env` with your API key |
| `Wikipedia connection error` | Check internet connection |
| Out of memory | Restart Python, close other programs |

## Cost Optimization

Using GPT-3.5-turbo instead of GPT-4:
- 10x cheaper (~$0.02 per run vs $0.25)
- Slightly lower quality reasoning
- Still demonstrates the system well

Edit `.env`:
```bash
MODEL_NAME=gpt-3.5-turbo
```

## What's Next?

1. ✅ Run the demo: `python main.py`
2. ✅ Try examples: `python examples.py`
3. ✅ Read [README.md](README.md) for details
4. ✅ Modify questions in `main.py`
5. ✅ Explore individual components

## Understanding SAUP

**SAUP** = Situation-Aware Uncertainty Propagation

It combines:
1. **Token probabilities** → Single-step uncertainty
2. **Semantic distances** → Context awareness
3. **HMM proxy** → Situational weights
4. **Weighted RMS** → Final uncertainty score

**Why it's better**: Standard methods ignore context. SAUP weights steps based on whether the agent is on track or deviating.

## Visual Guide

```
Question → [ReAct Agent] → Trajectory (steps)
              ↓
    Thought → Action → Observation
              ↓
    Token Probs → Uncertainty (U_n)
                       ↓
         [Semantic Distance Calculator]
                       ↓
    Inquiry Drift (D_a) + Inference Gap (D_o)
                       ↓
              [HMM Proxy]
                       ↓
         Situational Weight (W_n)
                       ↓
            [SAUP Propagator]
                       ↓
    U_agent = sqrt(1/N * Σ(W_i × U_i)²)
```

## Support

- Installation issues → [INSTALL.md](INSTALL.md)
- Usage questions → [README.md](README.md)
- Component details → [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- API documentation → Check docstrings in code

## Quick Test (No API Key)

Run examples without API key:
```bash
python examples.py
```

This demonstrates:
- Uncertainty calculation
- Semantic distance
- HMM proxy
- SAUP propagation
- Evaluation metrics

Press Enter to step through examples.

---

**Ready to start?**

```bash
cd saup_prototype
pip install -r requirements.txt
cp .env.template .env
# Add your API key to .env
python main.py
```

🚀 **Good luck!**
