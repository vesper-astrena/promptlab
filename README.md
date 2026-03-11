# PromptLab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Test and compare LLM prompts with one command. Measure response time, token usage, and cost.

## Quick Start

```bash
export OPENAI_API_KEY=sk-...
pip install requests pyyaml

# Test a single prompt
python promptlab.py "Summarize this text: {{input}}" --var input="The quick brown fox..."

# Compare multiple prompts from a YAML file
python promptlab.py templates/summarization.yaml --var input="Your long text here"
```

## What It Does

- **Template variables** — Use `{{variable}}` placeholders in prompts
- **Side-by-side comparison** — Test up to 3 prompt variations in one run
- **Metrics** — Response time, token count, estimated cost per call
- **YAML templates** — Define and reuse prompt collections
- **Cost estimation** — Per-model pricing for accurate cost tracking

## Included Templates

- `summarization.yaml` — 3 summarization strategies
- `extraction.yaml` — 3 data extraction approaches
- `classification.yaml` — 3 classification methods
- `code_review.yaml` — 3 code review styles
- `rewriting.yaml` — 3 rewriting techniques

## Pro Version

[PromptLab Pro](https://vesperfinch.gumroad.com/l/promptlab) ($24) adds:

- **Multi-model comparison** — OpenAI, Anthropic, Google Gemini, Ollama (local)
- **Batch testing** — Test prompts against CSV datasets
- **Auto-scoring** — LLM judge rates accuracy, completeness, clarity
- **A/B test significance** — Welch's t-test for statistical confidence
- **Cost optimization** — Recommendations for cheaper models
- **Prompt chains** — Multi-step prompt pipelines
- **HTML reports** — Beautiful dark-theme reports with visualizations
- **Unlimited variations** — No cap on prompt comparisons

## Also Check Out

- [CSV Cleaner](https://github.com/vesper-astrena/csv-cleaner) — Fix messy CSV files in one command
- [Polymarket Scanner](https://github.com/vesper-astrena/polymarket-scanner) — Scan prediction markets for mispricings
- [All Tools](https://vesper-astrena.github.io/devtools/) — Full product catalog

## License

MIT — free for personal and commercial use.
