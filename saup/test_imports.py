"""
Quick test script to verify all imports work correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from utils import console, format_uncertainty
    print("✓ utils")
except Exception as e:
    print(f"✗ utils: {e}")

try:
    from utils.embeddings import SemanticDistanceCalculator
    print("✓ utils.embeddings")
except Exception as e:
    print(f"✗ utils.embeddings: {e}")

try:
    from models.llm_wrapper import LLMWrapper
    print("✓ models.llm_wrapper")
except Exception as e:
    print(f"✗ models.llm_wrapper: {e}")

try:
    from models.hmm_proxy import HMMProxy
    print("✓ models.hmm_proxy")
except Exception as e:
    print(f"✗ models.hmm_proxy: {e}")

try:
    from uncertainty.single_step import calculate_predictive_entropy
    print("✓ uncertainty.single_step")
except Exception as e:
    print(f"✗ uncertainty.single_step: {e}")

try:
    from uncertainty.propagation import SAUPPropagator
    print("✓ uncertainty.propagation")
except Exception as e:
    print(f"✗ uncertainty.propagation: {e}")

try:
    from agents.base_agent import BaseAgent
    print("✓ agents.base_agent")
except Exception as e:
    print(f"✗ agents.base_agent: {e}")

try:
    from agents.react_agent import ReactAgent
    print("✓ agents.react_agent")
except Exception as e:
    print(f"✗ agents.react_agent: {e}")

try:
    from evaluation.metrics import calculate_auroc, visualize_results
    print("✓ evaluation.metrics")
except Exception as e:
    print(f"✗ evaluation.metrics: {e}")

print("\nAll imports successful!")
print("\nTo run the demo:")
print("1. Copy .env.template to .env")
print("2. Add your OPENAI_API_KEY to .env")
print("3. Run: python main.py")
