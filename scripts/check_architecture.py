#!/usr/bin/env python3
"""Architecture enforcement checks for ContextRouter.

This script runs as part of CI/pre-commit to catch architectural violations.
Exit code 0 = all checks passed, non-zero = violations found.

Violations:
1. Direct LLM provider imports in cortex/graphs/ (must use model_registry)
2. Hardcoded provider checks like `config.openai.api_key` (provider-agnostic)
"""

import subprocess
import sys
from pathlib import Path

# Where to check
CORTEX_GRAPHS = "src/contextrouter/cortex/graphs"

# Patterns that violate architecture
VIOLATIONS = [
    {
        "name": "Direct LLM provider import",
        "pattern": r"from contextrouter\.modules\.models\.llm\.(openai|anthropic|vertex|groq|runpod|local_openai)",
        "message": "Use `from contextrouter.modules.models import model_registry` instead",
        "exclude": [],
    },
    {
        "name": "Hardcoded OpenAI API key check",
        "pattern": r"config\.openai\.api_key",
        "message": "Models are provider-agnostic. Use model_registry which handles provider selection.",
        "exclude": [],
    },
    {
        "name": "Hardcoded Anthropic API key check",
        "pattern": r"config\.anthropic\.api_key",
        "message": "Models are provider-agnostic. Use model_registry which handles provider selection.",
        "exclude": [],
    },
    {
        "name": "Direct model instantiation",
        "pattern": r"(OpenAILLM|AnthropicLLM|VertexLLM|GroqLLM)\s*\(",
        "message": "Use `model_registry.get_llm_with_fallback()` for centralized model selection.",
        "exclude": [],
    },
    {
        "name": "Deprecated SerperSearch import (moved to connectors)",
        "pattern": r"from contextrouter\.modules\.models\.llm\.serper",
        "message": "Serper moved to connectors: `from contextrouter.modules.connectors.serper import SerperSearchConnector`",
        "exclude": [],
    },
    {
        "name": "Direct PerplexityLLM instantiation",
        "pattern": r"PerplexityLLM\s*\(",
        "message": "Use `model_registry.create_llm('perplexity/sonar', ...)` instead",
        "exclude": [],
    },
]


def run_grep(pattern: str, path: str, exclude: list[str]) -> list[str]:
    """Run ripgrep and return matching files with line numbers."""
    cmd = ["rg", "--no-heading", "--line-number", "--color=never", pattern, path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Filter out excluded files
            filtered = []
            for line in lines:
                skip = False
                for exc in exclude:
                    if exc in line:
                        skip = True
                        break
                if not skip:
                    filtered.append(line)
            return filtered
        return []
    except FileNotFoundError:
        # ripgrep not available, try grep
        cmd = ["grep", "-rn", "-E", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        return []


def main() -> int:
    """Run all architecture checks."""
    project_root = Path(__file__).parent.parent
    check_path = project_root / CORTEX_GRAPHS
    
    if not check_path.exists():
        print(f"Path not found: {check_path}")
        return 1
    
    violations_found = 0
    
    print("🔍 Running architecture enforcement checks...")
    print(f"   Checking: {check_path}\n")
    
    for check in VIOLATIONS:
        matches = run_grep(check["pattern"], str(check_path), check.get("exclude", []))
        
        if matches:
            violations_found += len(matches)
            print(f"❌ {check['name']}")
            print(f"   → {check['message']}")
            print()
            for match in matches:
                print(f"   {match}")
            print()
    
    if violations_found == 0:
        print("✅ All architecture checks passed!")
        return 0
    else:
        print(f"\n❌ Found {violations_found} violation(s). Please fix before committing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
