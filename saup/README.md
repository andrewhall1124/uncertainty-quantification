# SAUP: Situation-Aware Uncertainty Propagation

A complete Python implementation of the SAUP (Situation-Aware Uncertainty Propagation) system for quantifying uncertainty in multi-step reasoning agents.

## Overview

SAUP enhances uncertainty quantification for LLM-based agents by considering:
- **Single-step uncertainty**: Predictive entropy from token probabilities
- **Semantic distances**: Inquiry drift and inference gaps
- **Situational awareness**: Context-aware weighting via HMM proxy

## Features

- ✅ ReAct agent with Wikipedia search tool
- ✅ Token-level uncertainty extraction from LLMs
- ✅ Semantic distance calculation (inquiry drift & inference gap)
- ✅ HMM proxy for situational weight estimation
- ✅ SAUP Algorithm 1 implementation
- ✅ Rich terminal UI with progress tracking
- ✅ Evaluation metrics (AUROC, accuracy)
- ✅ Comparison with baseline methods

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

## Project Structure

```
saup_prototype/
├── main.py                 # Main demo script
├── requirements.txt        # Dependencies
├── .env.template          # Environment variables template
├── agents/
│   ├── base_agent.py      # Base agent class
│   └── react_agent.py     # ReAct agent with Wikipedia tool
├── uncertainty/
│   ├── single_step.py     # Single-step uncertainty (entropy)
│   └── propagation.py     # SAUP algorithm implementation
├── models/
│   ├── llm_wrapper.py     # LLM interface with uncertainty
│   └── hmm_proxy.py       # HMM proxy for situational weights
├── evaluation/
│   └── metrics.py         # Evaluation metrics (AUROC, etc.)
└── utils/
    ├── __init__.py        # Rich console setup
    └── embeddings.py      # Semantic distance calculator
```

## Usage

Run the main demo:
```bash
python main.py
```

This will:
1. Initialize the system components
2. Run test questions through the ReAct agent
3. Calculate SAUP uncertainty for each trajectory
4. Display results with rich formatting
5. Show evaluation metrics and comparisons

## How It Works

### 1. ReAct Agent
The agent follows a Thought-Action-Observation loop:
```
Thought: "I need to search for information about coffee and metabolism"
Action: search[coffee metabolism weight loss]
Observation: "Caffeine can slightly boost metabolism by 3-11%..."
```

### 2. Single-Step Uncertainty
For each step, calculate uncertainty from token probabilities:
- **Thinking uncertainty (U^T)**: Entropy over thought tokens
- **Action uncertainty (U^A)**: Entropy over action tokens
- **Step uncertainty (U_n)**: Combined uncertainty

### 3. Semantic Distances
Calculate semantic awareness metrics:
- **Inquiry drift (D_a)**: How far reasoning has drifted from question
- **Inference gap (D_o)**: Gap between thought and observation

### 4. Situational Weighting
HMM proxy estimates situational weights based on:
- Combined semantic distances
- Step position in trajectory
- Hidden state inference (correct/moderate/high_deviation)

### 5. SAUP Propagation
Calculate agent-level uncertainty using weighted RMS:
```
U_agent = sqrt(1/N * sum((W_i * U_i)^2))
```

## Configuration

Edit `.env` file:
```bash
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4              # or gpt-3.5-turbo
MAX_STEPS=7                   # Maximum reasoning steps
UNCERTAINTY_THRESHOLD=0.7     # High uncertainty threshold
```

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

💭 Thought: Coffee contains caffeine which affects metabolism...
🎯 Action: search[caffeine weight loss metabolism]
📊 Observation: Caffeine can slightly boost metabolism...
📈 Uncertainty: 0.15 (Low) ✓

[... continues for all steps ...]

╔══════════════════════════════════════════════════════════════╗
║                     SAUP Results                             ║
║ Agent Uncertainty: 0.67 (Moderate) ⚠️                        ║
║ Confidence: 33%                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Implementation Details

### Key Classes

- **LLMWrapper**: Interfaces with OpenAI API, extracts token probabilities
- **ReactAgent**: Implements ReAct reasoning loop with Wikipedia
- **SemanticDistanceCalculator**: Computes inquiry drift and inference gaps
- **HMMProxy**: Rule-based proxy for situational weight estimation
- **SAUPPropagator**: Implements main SAUP algorithm

### Key Functions

- `calculate_predictive_entropy()`: Normalized entropy from token probs
- `calculate_step_uncertainty()`: Combines thinking and action uncertainty
- `propagate_uncertainty()`: Main SAUP algorithm (Algorithm 1)
- `calculate_auroc()`: Evaluation metric for uncertainty calibration

## Evaluation

The system calculates:
- **AUROC**: How well uncertainty separates correct/incorrect predictions
- **Accuracy**: Overall correctness rate
- **Selective Accuracy**: Accuracy when abstaining on high uncertainty
- **Comparison**: SAUP vs. simple average, RMS, max uncertainty

## References

This implementation is based on the SAUP paper:
- Single-step uncertainty via predictive entropy
- Semantic distance metrics (inquiry drift, inference gap)
- HMM-based situational awareness
- Weighted RMS propagation

## Testing

Test individual components:
```python
# Test LLM wrapper
from models.llm_wrapper import LLMWrapper
llm = LLMWrapper()
text, probs = llm.generate_with_uncertainty("What is 2+2?")

# Test semantic distance
from utils.embeddings import SemanticDistanceCalculator
calc = SemanticDistanceCalculator()
drift = calc.calculate_inquiry_drift("Question?", trajectory)

# Test uncertainty calculation
from uncertainty.single_step import calculate_predictive_entropy
entropy = calculate_predictive_entropy([0.9, 0.85, 0.7])
```

## Troubleshooting

**Import errors**: Make sure all `__init__.py` files are present
**API errors**: Check OPENAI_API_KEY in .env
**Wikipedia errors**: Check internet connection
**AUROC errors**: Need at least 2 predictions with different labels

## Future Enhancements

- [ ] Support for local models via transformers
- [ ] Full CHMM implementation (instead of rule-based proxy)
- [ ] Additional reasoning benchmarks (HotpotQA, etc.)
- [ ] Batch processing for efficiency
- [ ] Web interface with Gradio/Streamlit

## License

MIT License

## Citation

If you use this implementation, please cite the original SAUP paper.
