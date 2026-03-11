#!/usr/bin/env python3
"""
PromptLab - LLM Prompt Testing & Comparison Toolkit (Free Version)
Test, compare, and optimize your LLM prompts from the command line.

Usage:
    python promptlab.py --prompt "Summarize: {{text}}" --var text="Hello world"
    python promptlab.py --file prompts.yaml --var input="test data"
    python promptlab.py --compare prompt1.yaml prompt2.yaml --var input="test"
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    name: str
    prompt: str
    description: str = ""
    category: str = ""

    def render(self, variables: dict[str, str]) -> str:
        result = self.prompt
        for key, value in variables.items():
            result = result.replace("{{" + key + "}}", value)
        missing = re.findall(r"\{\{(\w+)\}\}", result)
        if missing:
            raise ValueError(f"Missing variables: {', '.join(missing)}")
        return result


@dataclass
class RunResult:
    template_name: str
    model: str
    prompt_text: str
    response_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_seconds: float
    estimated_cost_usd: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pricing (USD per 1K tokens, as of 2026)
# ---------------------------------------------------------------------------

OPENAI_PRICING = {
    "gpt-4o":        {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini":   {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":   {"input": 0.01, "output": 0.03},
    "gpt-4":         {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1":            {"input": 0.015, "output": 0.06},
    "o1-mini":       {"input": 0.003, "output": 0.012},
    "o3-mini":       {"input": 0.0011, "output": 0.0044},
}

DEFAULT_MODEL = "gpt-4o-mini"


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = OPENAI_PRICING.get(model)
    if not pricing:
        return 0.0
    return (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])


# ---------------------------------------------------------------------------
# OpenAI API caller (requests only, no SDK)
# ---------------------------------------------------------------------------

def call_openai(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
) -> RunResult:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return RunResult(
            template_name="", model=model, prompt_text=prompt,
            response_text="", input_tokens=0, output_tokens=0,
            total_tokens=0, elapsed_seconds=0, estimated_cost_usd=0,
            error="OPENAI_API_KEY not set",
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        elapsed = time.perf_counter() - start

        if resp.status_code != 200:
            return RunResult(
                template_name="", model=model, prompt_text=prompt,
                response_text="", input_tokens=0, output_tokens=0,
                total_tokens=0, elapsed_seconds=elapsed, estimated_cost_usd=0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        usage = data.get("usage", {})
        input_tok = usage.get("prompt_tokens", 0)
        output_tok = usage.get("completion_tokens", 0)
        total_tok = usage.get("total_tokens", 0)
        text = data["choices"][0]["message"]["content"]

        return RunResult(
            template_name="",
            model=model,
            prompt_text=prompt,
            response_text=text,
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=total_tok,
            elapsed_seconds=elapsed,
            estimated_cost_usd=estimate_cost(model, input_tok, output_tok),
        )
    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start
        return RunResult(
            template_name="", model=model, prompt_text=prompt,
            response_text="", input_tokens=0, output_tokens=0,
            total_tokens=0, elapsed_seconds=elapsed, estimated_cost_usd=0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml_templates(path: str) -> list[PromptTemplate]:
    if yaml is None:
        print("Error: 'pyyaml' package required for YAML files. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f)

    templates: list[PromptTemplate] = []
    file_name = data.get("name", Path(path).stem)
    file_desc = data.get("description", "")
    file_cat = data.get("category", "")

    for t in data.get("templates", []):
        templates.append(PromptTemplate(
            name=t.get("name", "unnamed"),
            prompt=t["prompt"],
            description=t.get("description", file_desc),
            category=t.get("category", file_cat),
        ))
    return templates


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
DIM = "\033[2m"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(text: str, code: str) -> str:
    if supports_color():
        return f"{code}{text}{RESET}"
    return text


def truncate(text: str, max_len: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def print_results_table(results: list[RunResult]) -> None:
    if not results:
        print("No results to display.")
        return

    print()
    print(c("=" * 90, BOLD))
    print(c("  PROMPTLAB RESULTS", BOLD + CYAN))
    print(c("=" * 90, BOLD))

    for i, r in enumerate(results):
        print()
        label = r.template_name or f"Prompt #{i + 1}"
        print(c(f"  [{label}]", BOLD + GREEN))
        print(f"  Model:          {r.model}")
        print(f"  Time:           {r.elapsed_seconds:.2f}s")
        print(f"  Tokens:         {r.input_tokens} in / {r.output_tokens} out / {r.total_tokens} total")
        print(f"  Est. Cost:      ${r.estimated_cost_usd:.6f}")
        if r.error:
            print(c(f"  Error:          {r.error}", RED))
        else:
            print(f"  Response:")
            for line in r.response_text.strip().split("\n"):
                print(f"    {line}")
        print(c("  " + "-" * 86, DIM))

    if len(results) > 1:
        print()
        print(c("  COMPARISON SUMMARY", BOLD + YELLOW))
        print(c("  " + "-" * 86, DIM))
        header = f"  {'Name':<25} {'Time':>8} {'Tokens':>8} {'Cost':>12} {'Response (preview)':>30}"
        print(c(header, BOLD))
        for r in results:
            label = (r.template_name or "prompt")[:24]
            preview = truncate(r.response_text, 30) if not r.error else c("ERROR", RED)
            print(f"  {label:<25} {r.elapsed_seconds:>7.2f}s {r.total_tokens:>8} ${r.estimated_cost_usd:>11.6f} {preview:>30}")

        fastest = min(results, key=lambda x: x.elapsed_seconds if not x.error else float("inf"))
        cheapest = min(results, key=lambda x: x.estimated_cost_usd if not x.error else float("inf"))
        print()
        print(c(f"  Fastest: {fastest.template_name or 'prompt'} ({fastest.elapsed_seconds:.2f}s)", GREEN))
        print(c(f"  Cheapest: {cheapest.template_name or 'prompt'} (${cheapest.estimated_cost_usd:.6f})", GREEN))

    print()
    print(c("=" * 90, BOLD))
    print()


def print_results_json(results: list[RunResult]) -> None:
    out = []
    for r in results:
        out.append({
            "template_name": r.template_name,
            "model": r.model,
            "prompt_text": r.prompt_text,
            "response_text": r.response_text,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "total_tokens": r.total_tokens,
            "elapsed_seconds": round(r.elapsed_seconds, 4),
            "estimated_cost_usd": round(r.estimated_cost_usd, 8),
            "error": r.error,
        })
    print(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Variable parsing
# ---------------------------------------------------------------------------

def parse_variables(var_args: list[str] | None) -> dict[str, str]:
    variables: dict[str, str] = {}
    if not var_args:
        return variables
    for v in var_args:
        if "=" not in v:
            print(f"Warning: Ignoring malformed variable '{v}' (expected key=value)")
            continue
        key, value = v.split("=", 1)
        variables[key.strip()] = value.strip()
    return variables


def load_variables_from_file(path: str) -> dict[str, str]:
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        elif path.endswith((".yaml", ".yml")):
            if yaml is None:
                print("Error: 'pyyaml' required for YAML variable files.")
                sys.exit(1)
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

MAX_FREE_VARIATIONS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptlab",
        description="PromptLab - LLM Prompt Testing & Comparison Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "Translate to French: {{text}}" --var text="Hello world"
  %(prog)s --file templates/summarization.yaml --var input="Long article..."
  %(prog)s --compare templates/summarization.yaml templates/extraction.yaml --var input="data"
  %(prog)s --prompt "Explain {{topic}}" --model gpt-4o --temperature 0.3

Supported models: """ + ", ".join(OPENAI_PRICING.keys()),
    )

    parser.add_argument("--prompt", "-p", type=str, help="Inline prompt template (use {{var}} for variables)")
    parser.add_argument("--file", "-f", type=str, help="YAML file with prompt templates")
    parser.add_argument("--compare", nargs="+", help="Compare multiple YAML template files")
    parser.add_argument("--var", "-v", action="append", help="Variable in key=value format (repeatable)")
    parser.add_argument("--var-file", type=str, help="JSON/YAML file with variables")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens (default: 1024)")
    parser.add_argument("--system", "-s", type=str, help="System prompt")
    parser.add_argument("--output", "-o", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument("--dry-run", action="store_true", help="Show rendered prompts without calling the API")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--list-models", action="store_true", help="List supported models and pricing")
    parser.add_argument("--version", action="version", version="PromptLab 1.0.0 (Free)")
    return parser


def cmd_list_models() -> None:
    print()
    print(c("  Supported Models & Pricing (per 1K tokens)", BOLD))
    print(c("  " + "-" * 60, DIM))
    print(f"  {'Model':<20} {'Input':>12} {'Output':>12}")
    print(c("  " + "-" * 60, DIM))
    for model, prices in sorted(OPENAI_PRICING.items()):
        print(f"  {model:<20} ${prices['input']:>10.5f} ${prices['output']:>10.5f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        cmd_list_models()
        return 0

    # Collect variables
    variables = {}
    if args.var_file:
        variables.update(load_variables_from_file(args.var_file))
    variables.update(parse_variables(args.var))

    # Collect templates
    templates: list[PromptTemplate] = []

    if args.prompt:
        templates.append(PromptTemplate(name="inline", prompt=args.prompt))

    if args.file:
        templates.extend(load_yaml_templates(args.file))

    if args.compare:
        for fp in args.compare:
            templates.extend(load_yaml_templates(fp))

    if not templates:
        parser.print_help()
        return 1

    # Free version limit
    if len(templates) > MAX_FREE_VARIATIONS:
        print(c(f"\n  Free version supports up to {MAX_FREE_VARIATIONS} prompt variations per run.", YELLOW))
        print(c("  Upgrade to PromptLab Pro for unlimited variations.", YELLOW))
        print(c("  https://vesperfinch.gumroad.com/l/promptlab\n", DIM))
        templates = templates[:MAX_FREE_VARIATIONS]

    # Render templates
    rendered: list[tuple[PromptTemplate, str]] = []
    for tmpl in templates:
        try:
            text = tmpl.render(variables)
            rendered.append((tmpl, text))
        except ValueError as e:
            print(c(f"  Error in '{tmpl.name}': {e}", RED))
            return 1

    # Dry run
    if args.dry_run:
        print()
        print(c("  DRY RUN - Rendered Prompts", BOLD + YELLOW))
        print(c("  " + "-" * 60, DIM))
        for tmpl, text in rendered:
            print(c(f"\n  [{tmpl.name}]", BOLD))
            for line in text.split("\n"):
                print(f"    {line}")
        print()
        return 0

    # Run prompts
    results: list[RunResult] = []
    for tmpl, text in rendered:
        print(c(f"  Running '{tmpl.name}' with {args.model}...", DIM), end="", flush=True)
        result = call_openai(
            prompt=text,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system,
        )
        result.template_name = tmpl.name
        results.append(result)
        status = c(" done", GREEN) if not result.error else c(f" error: {result.error}", RED)
        print(status)

    # Output
    if args.output == "json":
        print_results_json(results)
    else:
        print_results_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
