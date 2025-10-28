# Installation Guide for SAUP System

## Quick Start

Follow these steps to set up and run the SAUP system:

### 1. Navigate to the project directory
```bash
cd saup_prototype
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

This will install:
- openai (GPT-4 API access)
- transformers (LLM support)
- sentence-transformers (semantic embeddings)
- numpy, scipy, scikit-learn (numerical computing)
- torch (deep learning backend)
- rich (terminal UI)
- wikipedia (Wikipedia API)
- python-dotenv (environment variables)
- tiktoken (token counting)

### 4. Set up environment variables
```bash
cp .env.template .env
```

Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
MODEL_NAME=gpt-4
MAX_STEPS=7
UNCERTAINTY_THRESHOLD=0.7
```

### 5. Test the installation
```bash
python test_imports.py
```

You should see:
```
Testing imports...
✓ utils
✓ utils.embeddings
✓ models.llm_wrapper
✓ models.hmm_proxy
✓ uncertainty.single_step
✓ uncertainty.propagation
✓ agents.base_agent
✓ agents.react_agent
✓ evaluation.metrics

All imports successful!
```

### 6. Run the demo
```bash
python main.py
```

## System Requirements

- Python 3.8 or higher
- 4GB+ RAM (for sentence-transformers models)
- Internet connection (for OpenAI API and Wikipedia)
- OpenAI API key with GPT-4 access

## Troubleshooting

### ImportError: No module named 'X'
Make sure you activated the virtual environment and installed requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### OpenAI API Key Error
Ensure your `.env` file exists and contains a valid API key:
```bash
cat .env  # Check file contents
```

### Wikipedia Connection Error
Check your internet connection. The system needs access to:
- OpenAI API (api.openai.com)
- Wikipedia API (en.wikipedia.org)

### Out of Memory Error
Reduce the model size or use a smaller sentence-transformers model:
Edit `utils/embeddings.py` line 31:
```python
self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model
```

### GPU/CUDA Warnings
These are safe to ignore. The system will run on CPU by default.

## Cost Estimates

Using GPT-4:
- ~3 questions × 7 steps × 200 tokens/step = ~4,200 tokens
- Estimated cost: $0.25 - $0.50 per run

Using GPT-3.5-turbo (change MODEL_NAME in .env):
- Same usage: ~$0.01 - $0.02 per run

## Optional: Use Local Models

To avoid API costs, you can modify `models/llm_wrapper.py` to use local models via Hugging Face transformers. However, you won't get token probabilities (logprobs) from all models.

## Next Steps

After installation:
1. Read [README.md](README.md) for system overview
2. Review example questions in `main.py`
3. Modify test questions to try your own queries
4. Explore individual components in each module

## Support

For issues:
1. Check this troubleshooting guide
2. Verify all dependencies are installed
3. Check Python version: `python --version`
4. Ensure environment is activated: `which python`

## Development Setup

If you want to modify the code:

```bash
# Install development tools
pip install pytest black flake8 mypy

# Format code
black .

# Type checking
mypy .

# Run tests (if you create them)
pytest
```

## Uninstallation

To remove the system:
```bash
# Deactivate virtual environment
deactivate

# Remove the entire directory
cd ..
rm -rf saup_prototype
```

Or just deactivate the environment to free up system resources:
```bash
deactivate
```
