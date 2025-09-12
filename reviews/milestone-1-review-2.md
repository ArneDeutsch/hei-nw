# M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)— Milestone Review

### 1) Executive summary

**Intent.** M1 was to introduce a *read-only* Episodic Adapter (single cross-attention) and a deterministic Memory-Token packer, plumb an optional `mem_tokens` surface into generation without behavior changes when empty, wire **B1** into the harness, and demonstrate **B1(empty) ≈ B0** with latency overhead recorded. Artifacts include adapter/packer modules, updated harness, tests, and convenience scripts.

**What’s in repo.** The codebase contains a real `EpisodicAdapter` with a strict no-op branch when memory is absent, a deterministic `pack_trace` packer, an extended generation API (`mem_tokens`, `adapter`) that warns/ignores in M1, a B1 mode in the harness that records `adapter_latency_overhead_s` and writes JSON+MD reports, and comprehensive tests & CI plumbing.

**Verdict.** **Partial.** Functionally, all core scope items exist and are wired end-to-end with solid tests. Two items are under-enforced: (1) latency **budget** is *recorded* but not asserted against the design’s target; (2) the plan’s **per-file** (≥95%) coverage target for `adapter.py` is not enforced in CI (diff coverage threshold is 85% repo-wide). CI “green” can’t be verified locally.

---

### 2) Evidence snapshot: repo & planning anchors

* **Repo tree (short)**
  `src/hei_nw/`: `adapter.py`, `pack.py`, `models/base.py`, `eval/harness.py`, `eval/report.py`, `metrics/compute.py|text.py|timing.py`, `utils/cli.py|io.py|seed.py`, datasets `scenario_{a..e}.py`
  `tests/`: `test_adapter.py`, `test_pack.py`, `test_b1_equivalence.py`, `eval/test_harness_unit.py`, `test_harness_b1.py`, `models/test_base_generate.py`, `test_models_base.py`, etc.
  `scripts/`: `compare_b0_b1.py`, `run_b0_small.sh`, `run_b1_empty.sh`, `grep_no_stubs.sh`
  `.github/workflows/ci.yml`, `pyproject.toml`, `.pre-commit-config.yaml`
  Example reports: `reports/baseline/*_B0_*`, `reports/m1-episodic-adapter/*_B1_*`.

* **Planning anchors used**

  * `planning/project-plan.md` → **“## M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)”** (Scope, Verification, DoD/Acceptance, Artifacts).
  * `planning/design.md` → **“Modes: B0–B3”**; **Episodic Adapter (cross-attention)**; **Latency: budget ≤10–20 ms per recall, adapter <5 ms at small token counts.**
  * `planning/validation-plan.md` → **“1) Test harness (four switches)”**, **Scenarios A–E**, metrics EM/F1/latency/recall\@k.
  * `planning/milestone-1-plan.md` → **“M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton) — Executable Task Plan”** with tasks **M1-T1…T8** (Goals, Key changes, Tests, Quality gates, Acceptance checks).

* **Assumptions/limits**

  * Could not run CI or measure real hardware latency budget here; reviewed code, scripts, and local artifacts only.
  * Coverage % not computed; checked CI config gates instead.

---

### 3) DoD / Acceptance verification table

| Item                                                                                                | Evidence (files/funcs/CLI)                                                                                              | Status      | Notes                                                                                          |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| **Adapter no-op when memory empty**                                                                 | `src/hei_nw/adapter.py` — `if M_t is None or M_t.shape[1] == 0: return H_t`                                             | **Pass**    | Byte-for-byte identity verified by tests.                                                      |
| **Deterministic Memory Token Packer**                                                               | `src/hei_nw/pack.py::pack_trace` uses fixed template `"<episodic> … </episodic>"`                                       | **Pass**    | Tests assert determinism and truncation.                                                       |
| **Generation API accepts mem\_tokens; no behavior change when empty; warns if adapter+mem\_tokens** | `src/hei_nw/models/base.py::generate` signature includes `mem_tokens`, `adapter`; emits `UserWarning` and ignores in M1 | **Pass**    | Tests cover identical outputs and warning.                                                     |
| **Harness B1 mode with latency overhead recorded**                                                  | `src/hei_nw/eval/harness.py::_evaluate_mode_b1` computes `adapter_latency_overhead_s` and writes reports                | **Pass**    | `tests/test_harness_b1.py` asserts JSON+MD exist.                                              |
| **Equivalence B1(empty) ≈ B0 across A–E (±0.1 EM/F1)**                                              | `tests/test_b1_equivalence.py` compares predictions; falls back to ±0.1                                                 | **Pass**    | Deterministic seeds handled.                                                                   |
| **Latency budget noted** (adapter <5ms small tokens per design)                                     | `design.md` budget; harness records overhead; **no threshold assert**                                                   | **Partial** | Recorded but not enforced by test/gate.                                                        |
| **Artifacts present**                                                                               | `adapter.py`, `pack.py`, tests, scripts, reports template (`eval/report.py`)                                            | **Pass**    | All items exist and are wired.                                                                 |
| **CI/Quality gates**                                                                                | `.github/workflows/ci.yml` runs black, ruff, mypy, pytest, diff-cover (85%), `scripts/grep_no_stubs.sh`                 | **Partial** | Config present; can’t verify “CI green” here. Per-file coverage ≥95% for adapter not enforced. |

---

### 4) Task-by-task review

#### M1-T1 \[CODEX] Create `EpisodicAdapter` (read-only; no-op when `M_t` empty)

* **Intent.** Real cross-attention; return `H_t` unchanged when no memory.
* **Findings.** `src/hei_nw/adapter.py` implements LN+MHA+residual; **no-op branch**: “`if M_t is None ...: return H_t`”. Tests: `tests/test_adapter.py::test_noop_when_memory_empty`, `::test_shapes_with_memory_tokens`.
* **Gaps/Risks.** Plan asked **≥95% file coverage**; CI doesn’t enforce per-file threshold.
* **Status.** **Partial** (impl/tests solid; coverage enforcement missing).

#### M1-T2 \[CODEX] Memory Token Packer (deterministic, capped)

* **Intent.** Deterministic text template → tokenize → cap.
* **Findings.** `src/hei_nw/pack.py::pack_trace`; template lines `who/what/where/when`; truncation via `[:max_mem_tokens]`. Tests: `tests/test_pack.py` determinism & missing fields.
* **Gaps/Risks.** None observed.
* **Status.** **Pass**.

#### M1-T3 \[CODEX] Extend base generation API with optional mem tokens (plumbing only)

* **Intent.** Stable API; no logic change when empty; warn if adapter+mem\_tokens.
* **Findings.** `src/hei_nw/models/base.py::generate(prompt, ..., mem_tokens=None, adapter=None, **kwargs)`; warning text: *“adapter read path is inactive in M1; memory tokens are ignored.”* Tests: `tests/test_models_base.py::test_generate_signature_accepts_mem_tokens`, `tests/models/test_base_generate.py::test_warning_when_adapter_and_mem_tokens`.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M1-T4 \[CODEX] Wire `--mode B1` in harness (empty memory path) and record latency

* **Intent.** B1 end-to-end; record adapter latency overhead; write reports under `reports/m1-episodic-adapter`.
* **Findings.** `src/hei_nw/eval/harness.py::_evaluate_mode_b1` builds adapter via `build_default_adapter`, runs B0 & B1, computes `adapter_latency_overhead_s`. `_save_reports` writes JSON+MD; default outdir is remapped for B1. Tests: `tests/test_harness_b1.py` runs CLI for `-n 0` and `-n 2` and asserts outputs exist.
* **Gaps/Risks.** Overhead is **recorded**, not **checked** against budget.
* **Status.** **Pass** (recording works; enforcement deferred—see follow-ups).

#### M1-T5 \[CODEX] Equivalence test: `B1(empty)` ≈ `B0`

* **Intent.** Prove equality (or ±0.1 EM/F1 fallback) across A–E.
* **Findings.** `tests/test_b1_equivalence.py` parametrized over A–E with fixed seed; exact prediction equality else EM/F1 deltas ≤0.1.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M1-T6 \[CODEX] CLI scripts + report template

* **Intent.** Reviewer convenience.
* **Findings.** `scripts/compare_b0_b1.py` prints EM/F1 deltas, exits non-zero if exceeding threshold; `scripts/run_b0_small.sh`, `scripts/run_b1_empty.sh`; `eval/report.py::build_markdown_report` includes “Adapter latency overhead” line. Tests: `tests/utils/test_scripts.py::test_compare_b0_b1_runs_help`; `tests/eval/test_report_details.py` asserts MD includes baseline/notes/overhead.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M1-T7 \[CODEX] CI & quality plumbing updates

* **Intent.** Ensure CI runs and gates.
* **Findings.** `.github/workflows/ci.yml` runs black, ruff, mypy, pytest with coverage & **diff-cover fail-under=85**, plus “No stubs” check. `pyproject.toml` configures tools; `.pre-commit-config.yaml` present.
* **Gaps/Risks.** CI result not verifiable here; *per-file* coverage ≥95% (for adapter) not enforced.
* **Status.** **Partial**.

#### M1-T8 \[CODEX] Stub-removal guardrail

* **Intent.** Prevent TODOs/mocks/stubs sneaking in.
* **Findings.** `scripts/grep_no_stubs.sh`; and repository scan shows **no** `TODO|FIXME|pass  # stub|raise NotImplementedError` in `src/`.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M1-H1/H2/H3 \[HUMAN/ChatGPT]

* **Intent.** Light design/API review; tiny runs; PR/CI checks.
* **Findings.** APIs match plan; tiny reports exist under `reports/`. CI config present; can’t attest to PR/green status offline.
* **Status.** **Partial** (execution external to repo).

---

### 5) Design & validation alignment

* **Design mapping.**

  * **Episodic Adapter (cross-attention):** implemented in `src/hei_nw/adapter.py` with pre-norm + residual, read-only semantics; no writes/state.
  * **Memory tokens:** `src/hei_nw/pack.py::pack_trace` provides deterministic tokenization template.
  * **Modes B0–B3:** harness exposes `--mode {B0,B1,...}` with handlers for **B0, B1**; B2/B3 stubs are *not* registered (error path prints “not supported in M1”), consistent with *skeleton* scope.
  * **Latency budget awareness:** overhead recorded and surfaced in MD (`eval/report.py`), aligned with design requirement to track budget early.

* **Validation mapping.**

  * **Scenarios A–E:** generators in `src/hei_nw/datasets/scenario_{a..e}.py` and used in harness; Scenario A includes **hard negatives** (ratio surfaced when scenario==A in reports).
  * **Metrics:** EM/F1 latency per-item → aggregate; `recall@k` propagated when baselines provide it.
  * **B1(empty) ≈ B0:** verified by test suite. Remaining goals (B1≻B0 with actual memory, replay/B2,B3) correctly deferred.

**No architectural drift observed**; M1 moves toward the intended architecture while keeping memory reads inactive.

---

### 6) Quality & CI assessment

* **Tooling.** Black, Ruff, mypy, pytest configured; diff-cover gate at 85% (branch). Pre-commit configured. GitHub Actions workflow present with stub-scan step.
* **Testing depth.** Good unit coverage on adapter/packer, harness utilities, CLI scripts; integration via tiny model and harness CLI (n=0/2). Deterministic seeds handled by `utils/seed.py`.
* **Risks.** Latency assertions absent (only recorded), and **per-file** coverage threshold promised in plan is not enforced. A minor duplication exists between `eval/report.save_reports` and harness’ `_save_reports`.

---

### 7) Gaps and **Follow-up tasks**

#### M1-F1 \[CODEX] Enforce adapter latency budget in tests

* **Goal:** Assert adapter overhead stays within the design target for small token counts on the tiny model.
* **Key changes:**

  1. Add `tests/test_b1_latency_budget.py` to run B0 vs B1 on tiny model (`n=4`, fixed seed) and assert `adapter_latency_overhead_s <= 0.005` (5 ms) *per item* averaged.
  2. Add `--fail-on-overhead` option to `scripts/compare_b0_b1.py` or expose a small Python helper to parse JSON and assert threshold.
* **Tests:**

  * `tests/test_b1_latency_budget.py::test_adapter_overhead_under_budget`
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  pytest -q tests/test_b1_latency_budget.py
  ```

#### M1-F2 \[CODEX] Per-file coverage check for `adapter.py`

* **Goal:** Meet the plan’s ≥95% coverage target for adapter logic.
* **Key changes:**

  1. Add `coverage[toml]` config to enforce per-file minimum for `src/hei_nw/adapter.py` (e.g., `fail_under_file = {"src/hei_nw/adapter.py": 95}` via a small coverage gate script).
  2. Update CI to run that gate after pytest.
* **Tests:**

  * Extend `tests/test_adapter.py` if needed (e.g., mask/shape edge).
* **Quality gates:** unchanged
* **Acceptance check:**

  ```bash
  python scripts/check_file_coverage.py src/hei_nw/adapter.py --min 95
  ```

#### M1-F3 \[CODEX] Add assertion that B1 reports include overhead field

* **Goal:** Guard against regressions in report contents.
* **Key changes:**

  1. Extend `tests/eval/test_harness_unit.py` with a check that `_save_reports` + B1 run includes `"adapter_latency_overhead_s"` in JSON.
* **Tests:**

  * `tests/eval/test_harness_unit.py::test_b1_reports_include_overhead`
* **Quality gates:** unchanged
* **Acceptance check:**

  ```bash
  pytest -q tests/eval/test_harness_unit.py::test_b1_reports_include_overhead
  ```

#### M1-F4 \[CODEX] Remove `_save_reports` duplication

* **Goal:** Use a single report writer to avoid drift.
* **Key changes:**

  1. Replace harness `_save_reports` with calls to `eval.report.save_reports`.
  2. Update imports and tests accordingly.
* **Tests:**

  * Update `tests/eval/test_harness_unit.py::test_save_reports`.
* **Quality gates:** unchanged
* **Acceptance check:**

  ```bash
  pytest -q tests/eval/test_harness_unit.py::test_save_reports
  ```

#### M1-F5H \[HUMAN/ChatGPT] CI “green” confirmation and artifact snapshot

* **Goal:** Verify the PR for M1 passes CI and preserve proof.
* **Steps:** Open PR titled “M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)”; ensure CI green; run `bash scripts/grep_no_stubs.sh`.
* **Acceptance check:** Attach CI run URL and output of `grep_no_stubs.sh` to `reviews/m1-ci-proof.md`.

---

### 8) Final verdict

**Partial** — Core functionality and tests are in place and aligned with design/validation; however, the **latency budget is not asserted** (only recorded) and the **per-file coverage target** for the adapter is not enforced.

**Minimum follow-ups to meet DoD:** **M1-F1**, **M1-F2** (plus **M1-F5H** to confirm CI green externally).
