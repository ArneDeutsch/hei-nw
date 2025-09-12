# M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)— Milestone Review

### 1) Executive summary (2–4 short paragraphs)

**Intended delivery.** M1 was scoped to add a read-only **Episodic Adapter** (single cross-attention) and a deterministic **Memory Token Packer**, wire **`--mode B1`** in the harness (empty memory), and prove **B1(empty) ≈ B0** while **recording adapter latency**. Artifacts include code, tests, scripts, and reports; CI must run black/ruff/mypy/pytest and enforce a “no stubs” guardrail.

**What’s in the repo.** The codebase contains `EpisodicAdapter` with a strict no-op path when memory is absent, a deterministic `pack_trace(...)`, a harness B1 path that constructs the adapter and forwards `{adapter, mem_tokens=None}` into generation, scripts to run/compare B0 vs B1, CI workflows, and example reports under `reports/m1-episodic-adapter/`.

**Verdict: Partial.** Core functionality and tests are present and largely aligned with the design/validation plan. However, I found (1) a **syntax error** candidate in `models/base.py` import (newline inside a `from … import …`), and (2) two **scripts with likely broken lines/regex** (`grep_no_stubs.sh`, `run_b1_empty.sh`). These are small but blocking if executed as-is. Fixes are straightforward.

---

### 2) Evidence snapshot: repo & planning anchors

* **Repo tree (short)**

  * `src/hei_nw/adapter.py` (EpisodicAdapter), `src/hei_nw/pack.py` (packer), `src/hei_nw/models/base.py` (generation+adapter factory), `src/hei_nw/eval/harness.py` (B0/B1 handlers, reports), `src/hei_nw/metrics/*`, `src/hei_nw/datasets/*`
  * `tests/` with unit/integration: `test_adapter.py`, `test_pack.py`, `test_b1_equivalence.py`, `test_harness_b1.py`, `tests/eval/*`, etc.
  * `scripts/compare_b0_b1.py`, `scripts/run_b1_empty.sh`, `scripts/grep_no_stubs.sh`
  * `reports/baseline/…`, `reports/m1-episodic-adapter/…` (sample JSON/MD already present)
  * CI & tooling: `.github/workflows/ci.yml`, `pyproject.toml`, `.pre-commit-config.yaml`
* **Planning anchors used**

  * `project-plan.md` → “## M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)” (Scope/Verification/DoD/Artifacts)
  * `design.md` → “### 5.2 Episodic Adapter”, “### 5.6 Memory Token Packer”, “## 6) Public Interfaces”
  * `validation-plan.md` → “# 1) Test harness (four switches)”, “# 3) Metrics…”, scenarios A–E
  * `milestone-1-plan.md` → “M1 — … — **Executable Task Plan**” + tasks **M1-T1..T8**, DoD, artifacts, QA gates
* **Assumptions/limits**

  * I cannot verify PR/branch naming or external CI runs. The “no stubs” script appears suspicious (regex truncated with `...`), but code itself has no stubs.

---

### 3) DoD / Acceptance verification table

| Item                                                                             | Evidence (files/funcs/CLI)                                                                                                                                  | Status      | Notes                                                                              |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------- |
| Episodic Adapter API exists & **no-op identical** when memory empty              | `EpisodicAdapter.forward(...)` returns `H_t` unchanged if `M_t is None or len==0` (`adapter.py`; test `tests/test_adapter.py::test_noop_when_memory_empty`) | **Pass**    | Docstring: “When no memory is supplied the input sequence is returned unchanged.”  |
| Memory Token Packer deterministic & capped                                       | `pack_trace(trace, tok, max_mem_tokens)` (`pack.py`); tests `tests/test_pack.py`                                                                            | **Pass**    | Stable field order & truncation verified.                                          |
| Harness `--mode B1` runs, writes metrics/MD under `reports/m1-episodic-adapter/` | `_evaluate_mode_b1(...)` + default outdir switch in `harness.py`; test `tests/test_harness_b1.py`                                                           | **Pass**    | Sample files present: `reports/m1-episodic-adapter/*_metrics.json`, `*_report.md`. |
| **Equivalence:** A–E tiny splits B1(empty) ≈ B0 (±0.1 EM/F1)                     | `tests/test_b1_equivalence.py`                                                                                                                              | **Pass**    | Compares predictions/metrics; comparator script available.                         |
| **Latency recorded** in B1 summaries                                             | `harness.py` adds `"adapter_latency_overhead_s"`; present in sample JSON                                                                                    | **Pass**    | Example JSON contains `adapter_latency_overhead_s`.                                |
| **Docs & scripts** included                                                      | `documentation/m1_adapter_notes.md` + `scripts/*`                                                                                                           | **Pass**    | Notes, run and compare scripts present.                                            |
| **No stubs/mocks**                                                               | Code scan clean; guard script `scripts/grep_no_stubs.sh`                                                                                                    | **Partial** | The guard script looks truncated (regex contains `...`), but code has no stubs.    |
| **Quality gates:** black/ruff/mypy/pytest, coverage gate                         | `.github/workflows/ci.yml`, `pyproject.toml`                                                                                                                | **Pass**    | CI runs quality + diff-coverage; tests present.                                    |
| Single PR & human review                                                         | Not verifiable from repo                                                                                                                                    | **N/A**     | Out of scope for on-disk review.                                                   |

---

### 4) Task-by-task review (mirror the milestone plan order)

#### M1-T1 \[CODEX] Create `EpisodicAdapter` (read-only, no-op when `M_t` empty)

* **Intent:** Real cross-attention adapter; strict no-op path when no memory.
* **Findings:** `src/hei_nw/adapter.py` defines `EpisodicAdapter` using `nn.MultiheadAttention` and residual add; early return of `H_t` if `M_t` is `None` or has zero len. Test `tests/test_adapter.py` checks identity and shape.
* **Gaps / Risks:** None functionally.
* **Status:** **Pass**.

#### M1-T2 \[CODEX] Memory Token Packer (deterministic, capped)

* **Intent:** Deterministic template; stable field order; truncate to `max_mem_tokens`.
* **Findings:** `src/hei_nw/pack.py` formats `"<episodic>\nwho:…\n…</episodic>"`, uses tokenizer to get `input_ids`, truncates. Tests verify determinism and missing fields.
* **Gaps / Risks:** None.
* **Status:** **Pass**.

#### M1-T3 \[CODEX] Extend base generation API (plumbing only)

* **Intent:** `generate(...)` accepts `mem_tokens` & optional `adapter` (read-only; warn unused).
* **Findings:** `src/hei_nw/models/base.py` `generate(prompt, …, mem_tokens=None, adapter=None, **kwargs)`; explicit `warnings.warn("…memory tokens are ignored.")`. Test `tests/test_models_base.py::test_generate_signature_accepts_mem_tokens`.
* **Gaps / Risks:** **Bug:** import appears as `from \nhei_nw.adapter import EpisodicAdapter` (newline after `from`). This would raise a SyntaxError on import.
* **Status:** **Partial**.

#### M1-T4 \[CODEX] Wire `--mode B1` in harness (empty memory) and record latency

* **Intent:** Enable B1, construct adapter, call `generate(..., adapter=…, mem_tokens=None)`, compute latency overhead, route outputs.
* **Findings:** `src/hei_nw/eval/harness.py` defines `_evaluate_mode_b1(...)` creating adapter via `build_default_adapter(...)`, then `_evaluate_records(..., adapter=adapter, mem_tokens=None)`; summary includes `"adapter_latency_overhead_s"`. Reports saved under `reports/m1-episodic-adapter/` when B1.
* **Gaps / Risks:** None functionally.
* **Status:** **Pass**.

#### M1-T5 \[CODEX] Equivalence test: `B1(empty)` ≈ `B0`

* **Intent:** Comparator within ±0.1 EM/F1.
* **Findings:** `tests/test_b1_equivalence.py` parametrizes scenarios A–E; also `scripts/compare_b0_b1.py` prints deltas and returns non-zero on exceed.
* **Gaps / Risks:** None.
* **Status:** **Pass**.

#### M1-T6 \[CODEX] CLI scripts + report template

* **Intent:** Provide helper scripts; markdown report includes core sections.
* **Findings:** `scripts/make_report.sh`, `scripts/run_b1_empty.sh`, `scripts/compare_b0_b1.py`; MD builder in `src/hei_nw/eval/report.py` includes EM/F1/Latency, lag bins, compute, dataset notes.
* **Gaps / Risks:** `run_b1_empty.sh` shows a line break inside `--model "$MODEL"` (split into two lines), which will break a command.
* **Status:** **Partial**.

#### M1-T7 \[CODEX] CI & quality plumbing updates

* **Intent:** Ensure CI runs black/ruff/mypy/pytest, diff-coverage, and “no stubs”.
* **Findings:** `.github/workflows/ci.yml` installs deps and runs quality gates and coverage; `.pre-commit-config.yaml` configured.
* **Gaps / Risks:** “No stubs” step calls `scripts/grep_no_stubs.sh` which appears truncated (`...`) in its regex; may not behave as intended.
* **Status:** **Partial**.

#### M1-T8 \[CODEX] Stub-removal guardrail

* **Intent:** Provide grep-based guard; ensure codebase clean.
* **Findings:** Code scan shows no `TODO/FIXME/NotImplementedError` in sources; the guard script itself likely malformed.
* **Gaps / Risks:** Script correctness.
* **Status:** **Partial**.

*(HUMAN tasks H1–H3)*

* **H1 Design/API review:** Docs present (`documentation/m1_adapter_notes.md`); adapter/packer docstrings are clear.
* **H2 Equivalence & latency sanity run:** `tests/test_harness_b1.py` runs B1 and writes JSON/MD; sample reports exist.
* **H3 PR review & CI gate:** Not verifiable here.

---

### 5) Design & validation alignment

* **Design mapping.**

  * **Adapter**: Implements “5.2 Episodic Adapter” as a *read-only* cross-attention with residual add, matching the doc (“inputs: `H_t`, `M_t`… outputs: `H'_t`”; placement is deferred to later milestones). Code: `src/hei_nw/adapter.py`, and `build_default_adapter(...)` in `src/hei_nw/models/base.py`.
  * **Packer**: Implements “5.6 Memory Token Packer” with a deterministic template. Code: `src/hei_nw/pack.py`.
  * **Interfaces**: `generate(...)` exposes `mem_tokens`/`adapter` consistent with “Public Interfaces”.
* **Validation mapping.**

  * Modes **B0/B1** exist; **B1** uses empty memory path and records **latency**. Scenarios **A–E** generators and harness run paths exist. Metrics align with “#3 Metrics”: EM/F1/Latency plus compute estimates; lag binning (`report.py`) implemented. Comparator script enforces equivalence tolerance per plan.

---

### 6) Quality & CI assessment

* **Tooling:** `pyproject.toml` configures Black/Ruff/Mypy/Pytest; `.pre-commit-config.yaml` present.
* **CI:** `.github/workflows/ci.yml` runs format, lint, types, tests, diff-coverage ≥85%, and the no-stubs check.
* **Testing depth:** Good mix of unit (adapter/packer/metrics) and integration (harness CLI, baselines, E2E tiny model). Deterministic seeds are used (`utils/seed.py`). Risk of flakiness is low on tiny models; external embedding test is marked `@slow` and skips on network failure.
* **Notable risk:** two shell scripts look malformed (regex/line breaks). One Python import likely malformed (`models/base.py`).

---

### 7) Gaps and **Follow-up tasks**

````
### M1-F1 [CODEX] Fix import newline and add import smoke test

* **Goal:** Ensure `hei_nw.models.base` imports cleanly in CI.
* **Key changes:**
  1) Edit `src/hei_nw/models/base.py` to replace `from \nhei_nw.adapter import EpisodicAdapter` with `from hei_nw.adapter import EpisodicAdapter`.
  2) Add a minimal import test.
* **Tests:**
  - `tests/models/test_imports.py::test_import_models_base`
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  PYTHONPATH=src pytest -q tests/models/test_imports.py
````

### M1-F2 \[CODEX] Repair helper scripts and harden with shellcheck

* **Goal:** Make `run_b1_empty.sh` and `grep_no_stubs.sh` robust and executable.
* **Key changes:**

  1. Fix broken line in `scripts/run_b1_empty.sh` (split `--model "$MODEL"` line).
  2. Replace `...` in `scripts/grep_no_stubs.sh` with the full regex: `(TODO|FIXME|pass  # stub|raise NotImplementedError)`, and exclude `planning/`, `reviews/`, `.github/workflows/ci.yml`.
* **Tests:**

  * Add `tests/utils/test_scripts.py::test_compare_b0_b1_runs_help` (sanity run `--help` for compare script).
  * Add `tests/utils/test_scripts.py::test_no_stubs_regex` (simulate grep via Python on a temp tree).
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  bash scripts/grep_no_stubs.sh || echo "No stubs."
  ```

### M1-F3 \[CODEX] Surface adapter overhead in Markdown reports

* **Goal:** Include `adapter_latency_overhead_s` in MD for B1 runs.
* **Key changes:**

  1. Update `src/hei_nw/eval/report.py::build_markdown_report` to append a line when `summary.get("adapter_latency_overhead_s")` exists.
* **Tests:**

  * Extend `tests/eval/test_report_details.py::test_markdown_includes_baseline_and_notes` to also assert overhead line when provided.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.eval.report import build_markdown_report
  s={"aggregate":{"em":0,"f1":0,"latency":0},"lag_bins":[], "compute":{"b0":{}}, "adapter_latency_overhead_s":0.012}
  md=build_markdown_report(s); assert "Adapter latency overhead" in md
  PY
  ```

### M1-F4 \[CODEX] Test default outdir logic for B1

* **Goal:** Assert that `--mode B1` reroutes default outdir to `reports/m1-episodic-adapter`.
* **Key changes:**

  1. Add `tests/test_harness_cli.py::test_cli_default_outdir_b1`.
* **Tests:**

  * Run CLI with no `--outdir`, `--mode B1`, tiny model; assert files under `reports/m1-episodic-adapter`.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  PYTHONPATH=src pytest -q tests/test_harness_cli.py::test_cli_default_outdir_b1
  ```

### M1-F2H \[HUMAN/ChatGPT] Record A–E tiny B0 vs B1 pairs and attach comparator output

* **Goal:** Provide concrete evidence that B1(empty) ≈ B0 across scenarios.
* **Steps:**

  1. Run:

     * `python -m hei_nw.eval.harness --mode B0 --scenario {A..E} -n 8 --model sshleifer/tiny-gpt2`
     * `python -m hei_nw.eval.harness --mode B1 --scenario {A..E} -n 8 --model sshleifer/tiny-gpt2`
  2. Compare: `python scripts/compare_b0_b1.py reports/baseline/*_B0_metrics.json reports/m1-episodic-adapter/*_B1_metrics.json`
  3. Commit comparator table (or paste) into `reviews/m1-evidence.md`.
* **Acceptance check:** Comparator exit code `0` and evidence committed.

### 8) Final verdict
**Partial** — Core adapter/packer/B1 plumbing and tests are solid and aligned with the plan; reports and CI wiring are present. To fully meet M1’s DoD without execution hazards, fix the import newline and script issues, and add small test coverage for B1’s default outdir and MD overhead line.

**Minimum follow-ups to reach Pass:** **M1-F1**, **M1-F2**, **M1-F4** (and optionally **M1-F3** to improve reporting; **M1-F2H** provides human evidence).
