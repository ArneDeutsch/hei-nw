# M0 — Repo, Baselines, and Harness (B0) — Milestone Review

### 1) Executive summary (2–4 short paragraphs)

**Intent.** M0 was scoped to stand up a reproducible B0 harness for baseline evaluations, provide synthetic datasets (A–E) with Scenario-A hard negatives and lag-binning, lock a base model, implement text + compute metrics, and emit JSON + Markdown reports. DoD also requires CI gates (black/ruff/mypy/pytest) and a no-stubs guard.

**What’s in the repo.** The repo contains a clean `src/hei_nw` package with datasets A–E, a B0 CLI harness that emits JSON+MD, text/compute/timing metrics, a long-context baseline, a functional RAG baseline module (with default `HFEmbedder`), scripts, and a comprehensive test suite including an end-to-end tiny-model run. CI config with QA gates and a “no stubs” grep guard is present. Pre-generated baseline reports are in `reports/baseline/`.

**Verdict.** **Partial.** Core B0 harness, datasets, metrics, reports, CI plumbing, and long-context baseline are implemented and exercised by tests. Two gaps keep this from “Pass”: (1) the CLI advertises `--baseline rag` but the harness ignores it (only long-context is wired), and (2) the Markdown report lacks the Scenario-A “hard negatives/confounders” note that the milestone’s HUMAN check calls for.

---

### 2) Evidence snapshot: repo & planning anchors

* **Repo tree (short).**

  * Top: `.github/`, `codex-env/`, `documentation/`, `models/` (tiny-gpt2 fixture), `planning/`, `prompts/`, `reports/`, `research/`, `scripts/`, `src/`, `tests/`, `pyproject.toml`, `.pre-commit-config.yaml`, `README.md`.
  * Key code: `src/hei_nw/{datasets,eval,metrics,baselines,models,utils}/…`
  * Key tests: `tests/test_*` incl. CLI/e2e, datasets, metrics, baselines.
  * Artifacts: `reports/baseline/*_metrics.json`, `*_report.md`.

* **Planning anchors used.**

  * `planning/milestone-0-plan.md`

    * **H1:** “M0 — Repo, Baselines, and Harness (B0) — **Executable Task Plan**”
    * **§5:** “Deliverables & Artifacts”
    * **§6:** “Definition of Done (DoD) Checklist”
    * **Tasks:** `M0-T1` … `M0-T13` (\[CODEX]); HUMAN items **H1–H3**.
  * `planning/project-plan.md`

    * **H2:** “M0 — Repo, Baselines, and Harness (B0)”; **DoD / Acceptance**.
  * `planning/design.md`

    * **H2:** “2) Executive Summary”; **H2:** “4) System Architecture”.
  * `planning/validation-plan.md`

    * **H1:** “1) Test harness (four switches) — B0/B1/B2/B3”
    * **One-shot phase (Scenario A)** and metrics list (EM/F1/recall\@k, lag bins, compute footprints).

* **Assumptions/limits.**

  * I did not run CI; I verified configs + tests presence.
  * GPU runs: repo already contains B0 reports under `reports/baseline/`; I did not regenerate them here.
  * HF downloads are avoided in tests via the bundled tiny model; RAG default embedder exists in library but is not wired into the CLI baseline.

---

### 3) DoD / Acceptance verification table

| Item                                          | Evidence (files/funcs/CLI)                                                                                                                                                        | Status           | Notes                                                                                                               |                               |          |                                              |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------- | -------- | -------------------------------------------- |
| Harness `--mode B0` works; JSON + MD          | `src/hei_nw/eval/harness.py: parse_args` (supports `B0`), `_save_reports()` writes `*_metrics.json` + `*_report.md`; existing artifacts `reports/baseline/20250911-101939_A_B0_*` | **Pass**         | Example MD shows “Aggregate Metrics / Lag bins / Compute”.                                                          |                               |          |                                              |
| Data generators A–E (≥500 items)              | `src/hei_nw/datasets/scenario_{a..e}.py`; programmatic check produced 512+ items (A yields 2× due to negatives).                                                                  | **Pass**         | A: 1024 records for `n=512` (includes hard negatives); B–E: 512 each.                                               |                               |          |                                              |
| Metrics EM/F1/latency + compute fields exist  | `src/hei_nw/metrics/{text,compute,timing}.py`; MD shows EM/F1/Latency; JSON `compute={"b0":…,"baseline":…}`                                                                       | **Pass**         | Example MD shows EM/F1/Latency; JSON has `compute.b0` and `compute.baseline`.                                       |                               |          |                                              |
| Scenario-A hard negatives                     | `scenario_a.generate()` builds confounders (`neg_*` fields); tests `tests/test_scenario_a.py::test_hard_negative_rate`                                                            | **Pass (logic)** | Logic + tests present. However MD lacks an explicit “hard negatives” note required by HUMAN H3 → **reporting gap**. |                               |          |                                              |
| Lag-bin robustness section                    | `eval/report.py::bin_by_lag`, `build_markdown_report()` adds “## Lag bins”; `tests/test_lag.py`                                                                                   | **Pass**         | Table present in MD with bins (e.g., `0–1`, `1–3`, `3–7`, `7–30`).                                                  |                               |          |                                              |
| Compute footprints: long-context vs no-memory | `baselines/long_context.py` returns `ComputeRecord`; CLI `_run_baseline("long-context", …)`; JSON contains `compute.b0` and `compute.baseline`                                    | **Pass**         | Long-context wired; rag not wired (see gap).                                                                        |                               |          |                                              |
| No stubs/mocks guard                          | `.github/workflows/ci.yml` step “No stubs” uses \`git grep -nE "(TODO                                                                                                             | FIXME            | pass  # stub                                                                                                        | raise NotImplementedError)"\` | **Pass** | Manual grep across `src/` found **no** hits. |
| QA gates green in CI                          | `.github/workflows/ci.yml` has black/ruff/mypy/pytest; `.pre-commit-config.yaml` mirrors                                                                                          | **Partial**      | Config present; not verified as green here. Pre-generated e2e artifacts suggest runs succeed locally.               |                               |          |                                              |

---

### 4) Task-by-task review (mirror the milestone plan order)

#### M0-T1 \[CODEX] Initialize package, configs, and CI plumbing

* **Intent.** Package skeleton, pyproject, seeds/io utils, pre-commit + CI.
* **Findings.** `pyproject.toml` (black/ruff/mypy/pytest cfg), `.pre-commit-config.yaml`, `.github/workflows/ci.yml`; `utils/seed.py::set_global_seed`; `utils/io.py::write_json, write_markdown, timestamp_slug`.
* **Gaps / Risks.** None material.
* **Status.** **Pass.**

#### M0-T2 \[CODEX] Base model loader (Qwen2.5-1.5B-Instruct) + generation defaults

* **Intent.** Load base CausalLM + sensible generation defaults; count tokens.
* **Findings.** `models/base.py::load_base` (handles tokenizer pad/eos; 4-bit optional), `::generate` returns `text`, `prompt_tokens`, `generated_tokens`; tests `tests/test_models_base.py`.
* **Gaps / Risks.** Uses local tiny-gpt2 in tests; Qwen ID support depends on runtime env (acceptable for M0).
* **Status.** **Pass.**

#### M0-T3 \[CODEX] Scenario A generator (+ hard negatives, lag bins)

* **Intent.** Episodic stories with confounders; lag distribution.
* **Findings.** `datasets/scenario_a.py` generates positives + negatives; snippet: “`neg_name = rng.choice([s for s in NAMES if s != name])`” (≤25w). Tests check shapes, negative rate, lag bins.
* **Gaps / Risks.** Reporting in MD lacks explicit “hard negatives” mention (ties to H3).
* **Status.** **Pass (logic)** / **Partial (reporting)**.

#### M0-T4 \[CODEX] Scenario B–E generators (functional, labeled)

* **Intent.** Provide other synthetic scenarios with `should_remember`.
* **Findings.** `datasets/scenario_{b..e}.py`; tests `tests/test_scenarios_rest.py`.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T5 \[CODEX] Text metrics (EM/F1/recall\@k) and timing scaffolds

* **Intent.** Implement metrics + simple timer.
* **Findings.** `metrics/text.py::exact_match, token_f1, recall_at_k`; `metrics/timing.py::time_block`; tests `tests/test_metrics_text.py`.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T6 \[CODEX] Compute metrics scaffolds (FLOPs / KV-cache)

* **Intent.** Estimators + schema with optional fields.
* **Findings.** `metrics/compute.py::estimate_attention_flops, estimate_kv_bytes, ComputeRecord`; tests `tests/test_compute.py`.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T7 \[CODEX] Long-context baseline module

* **Intent.** Pack long context; run baseline; return compute.
* **Findings.** `baselines/long_context.py::build_context, run_long_context`; harness `_run_baseline("long-context", …)`. Tests `tests/test_long_context.py`, `tests/test_run_baseline.py`.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T8 \[CODEX] RAG baseline interface (functional, tiny default)

* **Intent.** Pluggable RAG (`Embedder` Protocol, default `HFEmbedder`), FAISS index; compute + recall\@k.
* **Findings.** `baselines/rag.py` implements `Embedder` Protocol + `HFEmbedder`, FAISS index, `run_rag()`; tests use a **ToyEmbedder** in `tests/test_rag.py`.
* **Gaps / Risks.** **Not wired into the CLI harness**: `harness._run_baseline` early-outs unless `"long-context"`, despite `--baseline rag` being an allowed option.
* **Status.** **Partial.**

#### M0-T9 \[CODEX] Eval harness CLI (`--mode`, `--scenario`, JSON+MD reports)

* **Intent.** CLI driving datasets → model → metrics → reports.
* **Findings.** `eval/harness.py::parse_args` includes `--mode B0|B1|B2|B3` (B0 enforced), `--scenario A..E`, `-n`, `--outdir`, `--baseline`; `_save_reports()` writes expected filenames with timestamp.
* **Gaps / Risks.** Default `--outdir` is `outputs/` (plan examples show `reports/baseline/`); minor drift. `--baseline rag` exposed but ignored (see M0-T8).
* **Status.** **Partial.**

#### M0-T10 \[CODEX] Time-lag robustness scaffolding

* **Intent.** Produce lag-binned table.
* **Findings.** `eval/report.py::bin_by_lag`, included in MD; `tests/test_lag.py`.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T11 \[CODEX] Scripts and README updates

* **Intent.** Provide sample runs + report combiner + README.
* **Findings.** `scripts/run_b0_small.sh` (A..E, long-context for E), `scripts/make_report.sh`; `README.md` project structure and pointers.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T12 \[CODEX] Integration test (imports + end-to-end tiny run)

* **Intent.** E2E sanity with tiny model, CLI, and artifacts.
* **Findings.** `tests/test_e2e.py` imports modules and runs CLI; `tests/test_harness_cli.py` likewise; looks for JSON + MD creation.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

#### M0-T13 \[CODEX] Repo-wide **no-stubs** guard

* **Intent.** CI fails on stub patterns.
* **Findings.** `.github/workflows/ci.yml` includes grep; manual search across `src/` found no hits.
* **Gaps / Risks.** None found.
* **Status.** **Pass.**

---

### 5) Design & validation alignment

* **Design mapping.** The B0 harness, dataset generators, metrics, and baselines correspond to `design.md`’s high-level architecture (pkg layout under `src/hei_nw/*`) and prepare the scaffolding for B1–B3 (e.g., compute record fields, lag bins). Files: `eval/harness.py`, `eval/report.py`, `datasets/scenario_*.py`, `metrics/*`, `baselines/*`, `models/base.py`, `utils/*`.
* **Validation mapping.** Matches `validation-plan.md` “Test harness (four switches)” for **B0** and “One-shot phase” metrics: EM/F1/recall\@k, latency, lag-bin robustness, and compute footprints. Long-context fairness baseline is included; RAG exists but isn’t yet invocable from the harness. The MD report format includes the sections called out in the plan, except for the explicit “hard negatives” note for Scenario A (H3).

---

### 6) Quality & CI assessment

* **Tooling.** `black`, `ruff`, `mypy`, `pytest` configured in `pyproject.toml`; `.pre-commit-config.yaml` mirrors gates; `.github/workflows/ci.yml` runs all plus “No stubs”.
* **Organization & naming.** Clean `src/hei_nw` layout, small modules, clear function boundaries. CLI uses `argparse` per plan.
* **Testing depth.** Unit tests cover datasets, metrics, compute; integration tests cover CLI + artifacts; slow markers used for heavier cases. Determinism via `utils/seed.set_global_seed` and explicit seeds in tests. Risk of flakiness is low given the bundled tiny model and synthetic data.
* **Minor drift.** CLI default `--outdir` is `outputs/` whereas plan examples and scripts use `reports/baseline/`.

---

### 7) Gaps and **Follow-up tasks**

Below are surgical follow-ups to bring M0 to **Pass** without bleeding into later milestones.

````
### M0-F1 [CODEX] Wire RAG baseline into the harness

* **Goal:** Make `--baseline rag` functional so compute + recall@k are emitted when requested.
* **Key changes:**
  1) `src/hei_nw/eval/harness.py`: extend `_run_baseline()` to call `baselines.rag.run_rag()` when `baseline=="rag"`, using default `HFEmbedder` (fallback to ToyEmbedder if offline).
  2) `src/hei_nw/eval/harness.py`: ensure `summary["compute"]["baseline"]` is populated for rag.
* **Tests:**
  - Add `tests/test_harness_cli.py::test_cli_rag_smoke` invoking `--baseline rag --scenario E -n 4` with the tiny model; assert JSON has `compute.baseline`.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  python -m hei_nw.eval.harness --mode B0 --scenario E -n 8 --seed 0 --outdir reports/baseline --baseline rag
  # Expect: reports/baseline/*_metrics.json with "compute":{"b0":...,"baseline":...}
````

```
```

### M0-F2 \[CODEX] Add “Dataset notes” section to MD (hard negatives)

* **Goal:** Make Scenario-A hard-negatives explicitly visible in the Markdown report to satisfy H3.
* **Key changes:**

  1. `src/hei_nw/eval/report.py`: in `build_markdown_report()`, append a “## Dataset notes” section; when scenario == A, include a one-liner like “Hard negatives/confounders included (ratio X)”.
  2. `src/hei_nw/eval/harness.py`: pass scenario ID into the report builder (if not already).
* **Tests:**

  * Extend `tests/test_report_md.py::test_md_contains_required_sections` to assert `"Dataset notes"` and `"Hard negatives"` when fed a summary tagged with scenario A.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  python -m hei_nw.eval.harness --mode B0 --scenario A -n 16 --seed 1 --outdir reports/baseline
  # Expect: *_report.md contains a “Dataset notes” section mentioning hard negatives.
  ```

```
```

### M0-F3 \[CODEX] Align default output directory with plan (optional)

* **Goal:** Reduce friction with docs/scripts by defaulting to `reports/baseline/`.
* **Key changes:**

  1. `src/hei_nw/utils/cli.py`: change `--outdir` default to `Path("reports/baseline")`.
  2. Update `README.md` usage snippet if needed.
* **Tests:**

  * Update `tests/test_harness_cli.py` to not rely on default, or add a new test that asserts default location.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  python -m hei_nw.eval.harness --mode B0 --scenario B -n 4 --seed 0
  # Expect: reports/baseline/*_metrics.json and *_report.md
  ```

```
```

### M0-F3H \[HUMAN/ChatGPT] Re-run B0 sanity after fixes

* **Goal:** Verify the two visible changes (RAG wiring + Dataset notes).
* **Steps:**

  1. `bash codex-env/setup.sh`
  2. Run:

     * `python -m hei_nw.eval.harness --mode B0 --scenario A -n 64 --seed 7 --outdir reports/baseline`
     * `python -m hei_nw.eval.harness --mode B0 --scenario E -n 64 --seed 7 --outdir reports/baseline --baseline rag`
* **Acceptance check:** Commit the two new MD/JSON artifacts and paste one MD excerpt showing “Dataset notes” and one JSON snippet with `"compute":{"baseline":...}` into the PR.

```

---

### 8) Final verdict
**Partial** — core B0 harness, datasets, metrics, long-context baseline, reports, and CI plumbing meet the milestone’s spirit, but **RAG isn’t actually invocable from the harness** and the **Scenario-A “hard negatives” note is missing from the MD** required by H3.

**Minimum follow-ups to reach Pass:** **M0-F1**, **M0-F2**.  
(Optional polish: **M0-F3**, **M0-F3H**.)
```
