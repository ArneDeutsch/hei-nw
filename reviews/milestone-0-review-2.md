# M0 — Repo, Baselines, and Harness (B0) — Milestone Review

### 1) Executive summary (2–4 short paragraphs)

**Intent.** M0 was to stand up a reproducible evaluation harness for **B0 mode** (no memory), implement synthetic **A–E** scenario generators (with **hard negatives** in A), provide **long-context** and **RAG** baselines, wire **EM/F1/latency** metrics with **compute-cost scaffolds**, and emit both **JSON** and **Markdown** reports. Tooling (black/ruff/mypy/pytest/CI), scripts, and a “no-stubs” guard were in scope.

**What exists.** The repo contains a coherent Python package `hei_nw` with modules for **datasets**, **baselines** (long-context, RAG), **metrics** (text, compute, timing), an **eval harness CLI**, minimal **model loader/generation**, and **report** utilities. Tests cover units and an end-to-end CLI flow using a bundled `models/tiny-gpt2`. Scripts generate **reports** under `reports/baseline/` (already present with dated artifacts).

**Verdict.** **Partial → Pass with one fix.** Functionality aligns with the plan and the validation/design documents. The only blocker I found is a truncated **CI “no-stubs” grep line** that looks malformed; everything else meets DoD with real code and passing tests.

---

### 2) Evidence snapshot: repo & planning anchors

* **Repo tree (short).**

  * `.github/workflows/ci.yml` (CI), `.pre-commit-config.yaml`, `pyproject.toml`, `src/hei_nw/{datasets, baselines, metrics, eval, models, utils}`, `tests/`, `scripts/`, `reports/baseline/` (sample outputs).
* **Planning anchors used.**

  * `planning/project-plan.md` → **“M0 — Repo, Baselines, and Harness (B0)”** section (scope/artifacts).
  * `planning/design.md` → harness & data/metrics components; B0 mode semantics.
  * `planning/validation-plan.md` → scenarios **A–E**, metrics (**EM/F1/latency**, recall\@k field), compute scaffolds.
  * `planning/milestone-0-plan.md` → explicit task list **M0-T1 … M0-T13** with goals/tests/acceptance.
* **Assumptions/limits.** Planning docs include some literal `…` redactions; I grounded on concrete **code/tests/paths** when text was abbreviated.

---

### 3) DoD / Acceptance verification table

| Item                                                                        | Evidence (files/funcs/CLI)                                                                                                                          |      Status | Notes                                                                                                                                         |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------: | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Eval harness CLI for **B0** with `--mode/--scenario/--baseline` and reports | `src/hei_nw/eval/harness.py` (`parse_args`, `main`), `python -m hei_nw.eval.harness --mode B0 …`, reports under `reports/baseline/` already present |    **Pass** | Non-B0 modes return code **64** with message “Only B0 mode is supported in M0.” (lines show `if args.mode != "B0": … return 64`).             |
| Scenario generators **A–E** with labels; A has **hard negatives** & **lag** | `src/hei_nw/datasets/scenario_a.py` / `scenario_{b..e}.py`; tests: `tests/test_scenario_a.py`, `tests/test_scenarios_rest.py`                       |    **Pass** | `scenario_a.generate` outputs `{episode_text,cues,answers,should_remember,lag,group_id}`; tests assert hard-negative rate & lag-bin coverage. |
| **EM/F1/latency** metrics and **recall\@k** field                           | `src/hei_nw/metrics/text.py` (exact\_match, token\_f1, recall\_at\_k), `src/hei_nw/metrics/timing.py` (Timer); aggregation in harness               |    **Pass** | `test_metrics_text.py`, `test_lag.py` covered; harness aggregates `em/f1/latency` and includes `recall_at_k` (can be `None` in B0).           |
| **Compute** scaffolds (FLOPs / KV-cache) for B0 + baseline                  | `src/hei_nw/metrics/compute.py` (ComputeRecord, estimators); used in baselines & harness                                                            |    **Pass** | `test_compute.py` checks schema keys always present and monotonicity.                                                                         |
| **Long-context** baseline (functional)                                      | `src/hei_nw/baselines/long_context.py` (`build_context`, `run_long_context`) with compute                                                           |    **Pass** | `test_long_context.py` verifies compute fields set and context packing not empty.                                                             |
| **RAG** baseline (functional; pluggable embedder)                           | `src/hei_nw/baselines/rag.py` (`HFEmbedder`, `FaissIndex`, `run_rag`)                                                                               |    **Pass** | Tests use in-test ToyEmbedder; HF embedder test is `@slow`.                                                                                   |
| Base model loader + small **generate** API                                  | `src/hei_nw/models/base.py` (`load_base`, `generate`)                                                                                               |    **Pass** | `test_models_base.py` checks tokenizer round-trip & token counts with `models/tiny-gpt2`.                                                     |
| JSON + Markdown report with **lag bins** and **compute** sections           | `src/hei_nw/eval/report.py` (`bin_by_lag`, `build_markdown_report`); artifacts present                                                              |    **Pass** | `reports/baseline/*_metrics.json`, `*_report.md` exist; `test_report_md.py` asserts required sections.                                        |
| Scripts and README usage                                                    | `scripts/run_b0_small.sh`, `scripts/make_report.sh`, README usage block                                                                             |    **Pass** | README shows `bash scripts/run_b0_small.sh`.                                                                                                  |
| CI: format/lint/types/tests; **no-stubs guard**                             | `.github/workflows/ci.yml` has steps for black/ruff/mypy/pytest and a grep guard line                                                               | **Partial** | The **grep command appears truncated** in file (literal `…` inside the command). Needs fix.                                                   |
| Integration / CLI smoke tests                                               | `tests/test_e2e.py`, `tests/test_harness_cli.py`                                                                                                    |    **Pass** | E2E run produces JSON+MD; CLI covers `--baseline` branches incl. RAG.                                                                         |

---

### 4) Task-by-task review (mirror the milestone plan order)

#### M0-T1 \[CODEX] Initialize package, configs, and CI plumbing

* **Intent.** Create package + QA tooling.
* **Findings.** `pyproject.toml` (black/tool sections present but some `…` redactions), `.pre-commit-config.yaml` (black/ruff/mypy), CI at `.github/workflows/ci.yml` runs **black/ruff/mypy/pytest**. Utils exist: `utils/seed.py`, `utils/io.py` (`timestamp_slug`), `utils/cli.py` (adds `--seed/--device/--outdir`). Tests: `test_utils.py`, `test_io.py`.
* **Gaps/Risks.** CI “no-stubs” grep line is visibly truncated; `pyproject.toml` shows `…` where ruff/mypy options likely go.
* **Status.** **Partial.**

#### M0-T2 \[CODEX] Base model loader + `generate()`

* **Intent.** Load base LLM and expose simple generation API with token counts.
* **Findings.** `models/base.py` includes `load_base(model_id="Qwen/Qwen2.5-1.5B-Instruct", dtype="auto", quant_4bit=True)` and `generate(prompt, max_new_tokens=…)`. Test `test_models_base.py` validates token counts.
* **Status.** **Pass.**

#### M0-T3 \[CODEX] Scenario A (hard negatives, lag bins)

* **Intent.** Episodic one-shot stories with **hard negatives** and `lag` annotations.
* **Findings.** `datasets/scenario_a.py` outputs `{episode_text,cues,answers,should_remember,lag,group_id}`; tests `test_scenario_a.py::{test_shapes_and_fields,test_hard_negative_rate,test_lag_bins_cover}`.
* **Status.** **Pass.**

#### M0-T4 \[CODEX] Scenarios B–E

* **Intent.** Implement remaining generators with labels.
* **Findings.** `scenario_{b,c,d,e}.py` returning `{context,query,expected,should_remember,…}`; `tests/test_scenarios_rest.py` checks shapes and label presence.
* **Status.** **Pass.**

#### M0-T5 \[CODEX] Text metrics & timing

* **Intent.** Provide `exact_match`, `token_f1`, `recall_at_k` and a timer.
* **Findings.** `metrics/text.py`, `metrics/timing.py`; tests in `test_metrics_text.py`.
* **Status.** **Pass.**

#### M0-T6 \[CODEX] Compute metrics scaffolds

* **Intent.** Emit FLOPs/KV estimates; schema with always-present keys.
* **Findings.** `metrics/compute.py` (`ComputeRecord`, `estimate_attention_flops`, `estimate_kv_bytes`); tests in `test_compute.py`.
* **Status.** **Pass.**

#### M0-T7 \[CODEX] Long-context baseline

* **Intent.** Functional long-context baseline with compute accounting.
* **Findings.** `baselines/long_context.py` (`build_context`, `run_long_context`); tests `test_long_context.py`.
* **Status.** **Pass.**

#### M0-T8 \[CODEX] RAG baseline

* **Intent.** Functional RAG with pluggable embedder; default HF embedder.
* **Findings.** `baselines/rag.py` (`HFEmbedder`, `FaissIndex`, `run_rag`); tests include ToyEmbedder, and an `@slow` HF smoke.
* **Status.** **Pass.**

#### M0-T9 \[CODEX] Eval harness CLI & reports

* **Intent.** End-to-end B0 pipeline; JSON+MD report generation.
* **Findings.** `eval/harness.py` CLI with `--mode {B0..B3}`, `--scenario A..E`, `--baseline {none,long-context,rag}`; non-B0 returns **64** with help text. `eval/report.py` writes MD & JSON. `tests/test_harness_cli.py` validates.
* **Status.** **Pass.**

#### M0-T10 \[CODEX] Time-lag robustness scaffolding

* **Intent.** Bin by lag and report per-bin EM/F1/recall\@k.
* **Findings.** `eval/report.py::bin_by_lag` + table in `build_markdown_report`; tested in `test_lag.py`, `test_report_md.py`.
* **Status.** **Pass.**

#### M0-T11 \[CODEX] Scripts & README

* **Intent.** Repro scripts and usage docs.
* **Findings.** `scripts/run_b0_small.sh` (five runs across A–E, baseline on E), `scripts/make_report.sh`; README shows usage block.
* **Status.** **Pass.**

#### M0-T12 \[CODEX] Integration test (tiny E2E)

* **Intent.** Tiny end-to-end run completing with outputs.
* **Findings.** `tests/test_e2e.py` imports modules and runs B0 A with `-n 4`; asserts JSON/MD written.
* **Status.** **Pass.**

#### M0-T13 \[CODEX] Repo-wide **no-stubs** guard

* **Intent.** Enforce “no stubs/mocks.”
* **Findings.** CI includes a grep step, but the command line in `ci.yml` shows **literal `…`** inside: `git grep -nE "(TODO|FIXME|pass  # stub|raise NotImp…"` (truncated). A raw text search over `src/` found **no** occurrences of `NotImplementedError`/`pass  # stub`/`TODO`/`FIXME`.
* **Status.** **Partial** (guard likely ineffective in CI as written; codebase currently clean).

---

### 5) Design & validation alignment

* **Design mapping.** This milestone implements the **B0 harness** and all **data/metric/report** plumbing described in `design.md`. Concretely:

  * **Data flow:** `datasets/* → eval/harness.py → metrics/{text,timing,compute} → eval/report.py`.
  * **Baselines:** `baselines/{long_context,rag}` produce compute footprints and (for RAG) recall\@k candidates consistent with the design’s “baseline for fairness” notion.
  * **Model layer:** `models/base.py` stays thin, as planned for B0.
* **Validation mapping.** Matches `validation-plan.md`:

  * **Scenarios A–E** implemented with **A hard-negatives** and **lag** fields.
  * **Metrics**: EM/F1/latency reported; `recall@k` **present** (can be `None` in B0); **compute** fields included even if zero/None.
  * **Modes**: CLI exposes `{B0..B3}`; only **B0** enabled now with graceful errors for others → consistent with “not stubs, just unsupported in M0.”

---

### 6) Quality & CI assessment

* **Tooling present.** Black, Ruff, MyPy in CI; `.pre-commit-config.yaml` includes all three. `pyproject.toml` includes Black config; ruff/mypy sections appear abbreviated with `…` but CI uses the tools regardless.
* **CI workflow.** `.github/workflows/ci.yml` runs **install → black → ruff → mypy → pytest**. Final “no-stubs” grep step appears truncated by literal `…` mid-command → **needs correction**.
* **Testing depth.** Good unit coverage + integration (`test_e2e.py`, `test_harness_cli.py`). Determinism via `utils/seed.set_global_seed` and seeded generators. Slow tests gated. Tiny offline model bundled avoids network flakiness.
* **Organization.** Clean `src/hei_nw/*` layout matches plan; clear module boundaries; CLI defaults sane (`reports/baseline`).

---

### 7) Gaps and **Follow-up tasks**

````
### M0-F1 [CODEX] Fix CI “no-stubs” guard

* **Goal:** Ensure CI fails on stubs/TODOs via a working grep.
* **Key changes:**
  1) Edit `.github/workflows/ci.yml` to replace the truncated line with:
     git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" -- ':!tests' ':!prompts' ':!.github/workflows/ci.yml' || echo "No stubs."
  2) Verify YAML quoting so the shell sees the full pattern (wrap in single quotes if needed).
* **Tests:**
  - Add a temporary dummy file in a local branch with `raise NotImplementedError` and confirm CI job fails; then remove it.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  gh workflow run CI && gh run watch && gh run view --log --job "No stubs" | grep "No stubs."
````

```
```

### M0-F2 \[CODEX] Complete tool configs in `pyproject.toml`

* **Goal:** Remove `…` placeholders; make Ruff/MyPy configs explicit and reproducible.
* **Key changes:**

  1. Update `[tool.ruff]` and `[tool.mypy]` sections with the intended options (targets, ignores, strictness).
  2. Ensure configs match those implied in CI and the planning docs’ QA gates.
* **Tests:**

  * `ruff .` and `mypy .` pass locally; introduce a deliberate lint/type issue to confirm they fail appropriately, then fix.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  ruff --show-source . && mypy .
  ```

```

*(No human/GPU follow-ups are required for these fixes.)*

---

### 8) Final verdict
**Partial.** The implementation **meets M0’s functional DoD** with real code, tests, and produced reports. To fully meet the milestone definition, apply **M0-F1** (fix CI no-stubs guard) and **M0-F2** (finalize Ruff/MyPy config). After those two surgical fixes, **M0 is a Pass**.
```
