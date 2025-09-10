# Python Code Quality Guide

**Goal:** Generate maintainable, testable Python by default. Prefer simple, explicit designs; small, pure functions; typed, documented interfaces; and a clean project layout.

## 1) Project layout & packaging

* Use a **src layout** and modern packaging metadata:

  ```
  pyproject.toml
  src/<package>/__init__.py
  tests/
  ```

  Add `[project]` and `[build-system]` in `pyproject.toml`.
* Create isolated **virtual environments**; pin dependencies and use a lock (e.g., Poetry or pip-tools).

## 2) Style & imports

* Follow **PEP 8** naming/formatting (snake\_case functions, CamelCase classes, UPPER\_CASE constants). Avoid wildcard imports; prefer absolute imports.
* Autoformat and lint in CI/`pre-commit`: **Black** (format), **isort** (imports), **Ruff** (lint).

## 3) Functions & APIs

* Aim for **small, single-purpose functions** with minimal **side effects**; prefer returning values over mutating globals. Keep **cyclomatic complexity** low; target `< 10` per function.
* Prefer simple parameters; group related ones with **dataclasses** or small objects. (General practice consistent with PEP 8’s readability guidance.)
* Never use **mutable default arguments**; use `None` and create the object inside.

## 4) Errors & control flow

* Choose **EAFP** (try/except around the operation) when it avoids race conditions and improves clarity; don’t pre-check unnecessarily. Keep `try` blocks minimal.
* **Don’t catch bare exceptions**; catch specific ones or `Exception` at most.

## 5) Data & iterables

* Prefer iteration idioms: `for x in xs`, `enumerate(xs)`, `zip(a, b)`; avoid `range(len(xs))` unless you need indices.
* Use comprehensions and generator expressions for clear, single-pass transforms; avoid deeply nested comprehensions (readability).

## 6) Files & resources

* Use **context managers** (`with`) for files, locks, network connections, etc., to ensure cleanup. Prefer `pathlib.Path` for paths.

## 7) Logging

* Prefer `logging` over `print` for diagnosability; configure handlers/levels centrally (e.g., via `basicConfig`, dictConfig).

## 8) Typing & documentation

* Add **type hints** for public functions/classes/modules (PEP 484). They improve IDE support, refactoring, and static analysis.
* Write **docstrings** for public APIs (PEP 257). Use a short summary line, then details. Google-style or NumPy-style sections are fine.

## 9) Testing (default to pytest)

* Organize tests under `tests/` mirroring package structure; name tests `test_*.py`. Keep tests *fast, isolated, and deterministic*. Use **AAA** (Arrange-Act-Assert).
* Prefer **pytest fixtures** for setup/teardown; **parametrize** for scenario coverage; use **mocks** only at boundaries.
* Measure **coverage**; optimize gaps rather than chasing 100%. Use `coverage.py` and `pytest-cov`.
* You can mix in `unittest` when needed; pytest runs it too.

## 10) Module patterns

* Guard scripts with `if __name__ == "__main__":` and keep logic in importable functions for testability. Follow **absolute imports** and keep module interfaces explicit (optionally via `__all__`).

## 11) Tooling defaults for `pyproject.toml`

Minimal tool config to enforce the above (tweak as needed):

```toml
[tool.black]
line-length = 100

[tool.isort]
profile = "black"

[tool.ruff]
select = ["E", "F", "I", "B", "UP", "S"]
line-length = 100

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]

[tool.coverage.run]
branch = true
source = ["src/<package>"]
```

---

## Quick checklist for Codex

* [ ] Use PEP 8 names/format; no wildcard imports. ([Python Enhancement Proposals (PEPs)][7])
* [ ] Small, pure functions; complexity < \~10. ([GitHub][12])
* [ ] No mutable defaults; catch specific exceptions; use `with` for resources. ([Python documentation][13], [Stack Overflow][16], [Real Python][19])
* [ ] Type hints + docstrings on public APIs. ([Python Enhancement Proposals (PEPs)][21])
* [ ] Tests with pytest: fixtures, parametrize, AAA; measure coverage. ([docs.pytest.org][24], [coverage.readthedocs.io][26])
* [ ] Enforce Black + isort + Ruff via pre-commit/CI. ([black.readthedocs.io][8], [pycqa.github.io][9], [Astral Docs][10], [pre-commit.com][11])
