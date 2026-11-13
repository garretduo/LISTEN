"""Checks that the documentation of the splits work is complete and still correct.

Purpose
    Teammates call ``listen.text``, ``listen.splits`` and ``listen.data`` without
    reading their source, so the docstrings are part of the interface. This file
    makes sure of three things:

    (a) every public function and class in those three modules has a docstring that
        names its inputs (``Args``), its result (``Returns``) and its failures
        (``Raises``) when it has them;
    (b) every Python example in ``docs/using_splits.md`` that is marked
        ``<!-- tested -->`` still runs from the repository root;
    (c) the teammate-facing docs contain no ``TODO-LEAD-FILL`` placeholder.

    Two smaller checks keep the docs honest: every ``make <target>`` named in the docs
    exists in the Makefile, and every repository path named in the docs exists.

Input
    ``docs/using_splits.md``, ``README.md``, ``results/splits/README.md``, the
    ``Makefile``, ``docs/decisions.md``, ``docs/data_notes.md`` and the modules above.
    An example in part (b) that calls ``get_split`` also needs the data CSV in
    ``data/`` and the frozen split files in ``results/splits/``; it is skipped (not
    failed) when either is missing, for example on a fresh clone before ``make data``.
    An example that only calls ``clean_text`` always runs.

Output
    pytest results only. Nothing is written to disk and no story text is printed.

Run it with ``make test`` or, from the repository root, ``python -m pytest
tests/test_docs.py``. Set the environment variable ``LISTEN_DOCS_DRAFT=1`` to skip
check (c) while the docs still hold placeholders, so the other checks can run early.
"""

import ast
import importlib
import inspect
import os
import re
import textwrap
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
USING_SPLITS = ROOT / "docs" / "using_splits.md"
README = ROOT / "README.md"
SPLITS_README = ROOT / "results" / "splits" / "README.md"
MAKEFILE = ROOT / "Makefile"
CONFIG_PATH = ROOT / "configs" / "default.yaml"

# Modules whose public names must be documented. Imported inside the tests, not here,
# so that one broken or missing module cannot stop the other checks from being collected.
DOCUMENTED_MODULES = ["listen.text", "listen.splits", "listen.data"]

# The marker that makes a Python block in using_splits.md run as a test.
TESTED_MARKER = "<!-- tested -->"

# A tested example must finish well inside this many seconds.
EXAMPLE_TIME_LIMIT_SECONDS = 30

# Teammate-facing docs that must be free of placeholders and must name real things.
CHECKED_DOCS = [
    README,
    USING_SPLITS,
    SPLITS_README,
    ROOT / "docs" / "decisions.md",
    ROOT / "docs" / "data_notes.md",
]


# --- helpers ------------------------------------------------------------------------


def _public_members(module):
    """Return {name: object} for the public functions and classes defined in module.

    "Public" means the name has no leading underscore. "Defined in module" means
    ``obj.__module__ == module.__name__``, which drops imported names such as numpy
    functions and names that listen.data re-exports from another listen module.
    Constants such as ``CLEANING_VERSION`` are not functions or classes, so they are
    not checked.
    """
    members = {}
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        members[name] = obj
    return members


def _has_return_value(func):
    """True if func has a return annotation other than None (or the string "None")."""
    annotation = inspect.signature(func).return_annotation
    if annotation is inspect.Signature.empty:
        return False
    return annotation is not None and annotation != "None"


def _raises_something(func):
    """True if the source of func contains a ``raise`` statement.

    The source is parsed instead of searched for the text "raise ", so a docstring or
    comment that merely contains the word does not count, and a bare ``raise`` does.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return any(isinstance(node, ast.Raise) for node in ast.walk(tree))


def _function_problems(name, func):
    """Return a list of docstring problems for one function (empty if none)."""
    doc = inspect.getdoc(func)
    if not doc or not doc.strip():
        return [f"{name}: no docstring"]
    problems = []
    parameters = [p for p in inspect.signature(func).parameters if p not in ("self", "cls")]
    if parameters and "Args" not in doc:
        problems.append(f"{name}: has parameters {parameters} but the docstring has no 'Args'")
    if _has_return_value(func) and "Returns" not in doc:
        problems.append(f"{name}: has a return annotation but the docstring has no 'Returns'")
    if _raises_something(func) and "Raises" not in doc:
        problems.append(f"{name}: contains a raise statement but the docstring has no 'Raises'")
    return problems


def _class_problems(name, cls):
    """Return a list of docstring problems for one class (empty if none).

    Reads ``cls.__dict__`` on purpose: ``inspect.getdoc`` would inherit the docstring of
    a base class such as ``tuple``, and a dataclass gets an automatic one-line signature
    as its docstring. Neither tells a teammate anything.
    """
    doc = cls.__dict__.get("__doc__")
    if not doc or not doc.strip():
        return [f"{name}: class has no docstring of its own"]
    if doc.startswith(f"{cls.__name__}("):
        return [f"{name}: class docstring is the auto-generated signature, write one"]
    return []


def _doc_text(path):
    """Return the text of a doc file, or fail the test with a clear message."""
    assert path.exists(), f"{path.relative_to(ROOT)} does not exist"
    return path.read_text(encoding="utf-8")


def _tested_blocks():
    """Return the Python code of every block marked ``<!-- tested -->`` in using_splits.md.

    A block counts only when the marker is the line immediately before a line that opens
    a fenced block with ``python``. An unmarked block is illustrative and not run.
    """
    if not USING_SPLITS.exists():
        return []
    lines = USING_SPLITS.read_text(encoding="utf-8").splitlines()
    blocks = []
    for i, line in enumerate(lines[:-1]):
        if line.strip() == TESTED_MARKER and lines[i + 1].strip().startswith("```python"):
            code = []
            for body in lines[i + 2 :]:
                if body.strip().startswith("```"):
                    break
                code.append(body)
            blocks.append("\n".join(code) + "\n")
    return blocks


def _data_and_frozen_files():
    """Return the paths the tested examples need: the data CSV and the frozen split files."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = config.get("paths", {})
    data_set = config["data"]["default_set"]
    csv_name = config["data"]["files"][data_set]["name"]
    splits_dir = ROOT / paths.get("splits_dir", "results/splits")
    return [
        ROOT / paths.get("data_dir", "data") / csv_name,
        splits_dir / f"{data_set}.csv",
        splits_dir / f"{data_set}.meta.json",
    ]


# --- (a) docstrings of the public API -----------------------------------------------


@pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
def test_public_names_are_documented(module_name):
    """Every public function and class has a docstring with Args, Returns and Raises where due."""
    module = importlib.import_module(module_name)
    members = _public_members(module)
    assert members, f"{module_name} defines no public function or class to document"
    problems = []
    for name, obj in sorted(members.items()):
        if inspect.isclass(obj):
            problems.extend(_class_problems(name, obj))
        else:
            problems.extend(_function_problems(name, obj))
    assert not problems, f"{module_name}:\n  " + "\n  ".join(problems)


# --- (b) the examples in docs/using_splits.md ---------------------------------------


def test_using_splits_has_tested_examples():
    """docs/using_splits.md has at least two Python blocks marked as tested."""
    _doc_text(USING_SPLITS)
    blocks = _tested_blocks()
    assert len(blocks) >= 2, (
        f"found {len(blocks)} block(s) preceded by the line {TESTED_MARKER!r}; "
        "docs/using_splits.md needs at least two"
    )


@pytest.mark.parametrize("number", range(1, len(_tested_blocks()) + 1))
def test_tested_example_runs(number, monkeypatch):
    """Each tested example runs from the repository root and finishes quickly."""
    code = _tested_blocks()[number - 1]
    if "get_split" in code or "load_dataset" in code:  # an example that reads the data
        missing = [p for p in _data_and_frozen_files() if not p.exists()]
        if missing:
            pytest.skip("needs " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__main__"}  # fresh namespace: no names leak between examples
    started = time.perf_counter()
    exec(compile(code, f"docs/using_splits.md (tested block {number})", "exec"), namespace)
    elapsed = time.perf_counter() - started
    assert elapsed < EXAMPLE_TIME_LIMIT_SECONDS, (
        f"tested block {number} took {elapsed:.1f} s; keep examples under "
        f"{EXAMPLE_TIME_LIMIT_SECONDS} s"
    )


# --- (c) no placeholders left in the teammate-facing docs ---------------------------


@pytest.mark.skipif(
    os.environ.get("LISTEN_DOCS_DRAFT") == "1",
    reason="LISTEN_DOCS_DRAFT=1: docs may still contain TODO-LEAD-FILL placeholders",
)
@pytest.mark.parametrize("path", [USING_SPLITS, README], ids=lambda p: p.name)
def test_no_placeholders_left(path):
    """docs/using_splits.md and README.md contain no TODO-LEAD-FILL token."""
    text = _doc_text(path)
    count = text.count("TODO-LEAD-FILL")
    assert count == 0, f"{path.relative_to(ROOT)} still has {count} TODO-LEAD-FILL placeholder(s)"


# --- the docs name real commands and real files -------------------------------------


def _makefile_targets():
    """Return the set of target names declared in the Makefile's .PHONY line."""
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^\.PHONY:(.*)$", text, flags=re.MULTILINE)
    assert match, "Makefile has no .PHONY line"
    return set(match.group(1).split())


@pytest.mark.parametrize("path", CHECKED_DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_make_targets_named_in_docs_exist(path):
    """Every `make <target>` written in the docs is a target of the Makefile."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} does not exist yet")
    targets = _makefile_targets()
    named = set(re.findall(r"`make ([a-z][a-z_-]*)", path.read_text(encoding="utf-8")))
    unknown = sorted(named - targets)
    assert not unknown, f"{path.relative_to(ROOT)} names make target(s) {unknown}; Makefile has {sorted(targets)}"


@pytest.mark.parametrize("path", CHECKED_DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_paths_named_in_docs_exist(path):
    """Every repository path written in backticks in the docs exists."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} does not exist yet")
    text = path.read_text(encoding="utf-8")
    pattern = r"`((?:docs|src|scripts|tests|configs|app|analysis|results)/[A-Za-z0-9_./-]*)`"
    missing = []
    for token in sorted(set(re.findall(pattern, text))):
        target = ROOT / token.rstrip(".,;:")
        if token.startswith("results/splits/") and token.endswith((".csv", ".json")):
            continue  # the frozen files may not be generated yet; tests/test_frozen.py checks them
        if not target.exists():
            missing.append(token)
    assert not missing, f"{path.relative_to(ROOT)} names path(s) that do not exist: {missing}"
