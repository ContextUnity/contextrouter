"""Test: no direct os.getenv / os.environ usage outside core/config.

This conformance test scans the contextrouter codebase and **fails** if any
Python file outside `core/config/` reads `os.environ` or `os.getenv` directly.

All configuration must flow through the Config system:
  - `from contextrouter.core import get_core_config`
  - `get_core_config().section.field`

The ONLY permitted files are:
  - core/config/base.py    — the single sanctioned env-reader
  - core/config/main.py    — uses base.get_env / base.get_bool_env (imported helpers)

Everything else MUST go through get_core_config().
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Root of the contextrouter package
# tests/unit/test_no_direct_env.py → ../../src/contextrouter
_TESTS_DIR = Path(__file__).resolve().parent.parent  # tests/
_PROJECT_ROOT = _TESTS_DIR.parent  # services/contextrouter/
_PKG_ROOT = _PROJECT_ROOT / "src" / "contextrouter"

# Files that are allowed to access os.environ / os.getenv
_ALLOWED_FILES = {
    _PKG_ROOT / "core" / "config" / "base.py",
    _PKG_ROOT / "core" / "config" / "main.py",
}


def _scan_file_for_os_env(filepath: Path) -> list[tuple[int, str]]:
    """Parse a Python file's AST and find os.getenv / os.environ usages.

    Returns:
        List of (line_number, code_snippet) tuples for each violation.
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # Detect: os.getenv(...)
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            ):
                violations.append((node.lineno, "os.getenv(...)"))

            # Detect: os.environ.get(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
            ):
                violations.append((node.lineno, "os.environ.get(...)"))

        # Detect: os.environ[...] (subscript access)
        if isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "environ"
                and isinstance(value.value, ast.Name)
                and value.value.id == "os"
            ):
                violations.append((node.lineno, "os.environ[...]"))

        # Detect: os.environ.setdefault(...) — only allowed in base.py
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "setdefault"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
            ):
                violations.append((node.lineno, "os.environ.setdefault(...)"))

    return violations


def test_no_direct_os_environ_usage():
    """Ensure no Python file outside core/config uses os.getenv or os.environ."""
    all_violations: list[str] = []

    for py_file in sorted(_PKG_ROOT.rglob("*.py")):
        # Skip allowed files
        resolved = py_file.resolve()
        if resolved in _ALLOWED_FILES:
            continue

        # Skip generated protobuf files
        if py_file.name.endswith("_pb2.py") or py_file.name.endswith("_pb2_grpc.py"):
            continue

        violations = _scan_file_for_os_env(py_file)
        if violations:
            rel = py_file.relative_to(_PKG_ROOT)
            for lineno, snippet in violations:
                all_violations.append(f"  {rel}:{lineno} — {snippet}")

    if all_violations:
        msg = (
            "\n\n❌ DIRECT os.environ / os.getenv USAGE DETECTED!\n\n"
            "All configuration must go through get_core_config().\n"
            "Only core/config/base.py is allowed to access os.environ.\n\n"
            "Violations:\n" + "\n".join(all_violations) + "\n\n"
            "Fix: Replace with `from contextrouter.core import get_core_config`\n"
            "     then use `get_core_config().section.field`\n"
        )
        raise AssertionError(msg)


def test_allowed_files_exist():
    """Verify the sanctioned config files actually exist."""
    for path in _ALLOWED_FILES:
        assert path.exists(), f"Sanctioned config file missing: {path}"


if __name__ == "__main__":
    # Allow running standalone: python tests/test_no_direct_env.py
    try:
        test_allowed_files_exist()
        test_no_direct_os_environ_usage()
        print("✅ No os.environ / os.getenv violations found!")
    except AssertionError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
