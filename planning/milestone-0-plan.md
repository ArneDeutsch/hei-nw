# M0 — Repo, Baselines, and Harness (B0) — **Executable Task Plan**

---

## 1) Milestone Summary

* Stand up a **reproducible eval harness** with `--mode {B0,B1,B2,B3}` (B0 fully working) and **scenario generators A–E** producing labeled episodic data with hard negatives (A).
* Lock a **base model** (Qwen2.5-1.5B-Instruct), tokenizer, generation params, and seeds; produce **baseline B0 metrics** and **compute footprints** (no-memory vs **long-context** baseline).
* Emit **machine-readable metrics (JSON)** and a concise **Markdown report**, including EM/F1, latency, recall\@k fields, and FLOPs/KV-cache **scaffolds** (compute fields present, can be 0/None in M0).

---

## 2) Dependencies / Inputs

1. Repository: `hei-nw-main.zip` (unpacked).
2. Planning docs to honor:

   * `planning/project-plan.md` → **M0 — Scope / DoD / Artifacts** (B0 harness, datasets A–E, reports/baseline).
   * `planning/design.md` → modes semantics (**B0 = base only**), base model integration expectations.
   * `planning/validation-plan.md` → **modes B0–B3**, scenarios **A–E**, metrics (**EM/F1, recall\@k, latency**), robustness vs **time-lag**.
3. Toolchain (already listed): `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets`, `faiss-cpu`, `hydra-core`, `pydantic`, `pytest(-cov)`, `black`, `ruff`, `mypy`, `hypothesis`, `psutil`, `matplotlib` (see `codex-env/requirements.txt`).
4. GPU for baseline runs: a single 16GB GPU preferred (for Qwen2.5-1.5B-Instruct in 4-bit). CPU-only smoke works for CI micro-tests.

---

## 3) \[CODEX] Implementation Tasks

> **Branch:** `feat/m0-baselines-harness`
> **Package layout (new):**
>
> ```
> pyproject.toml
> src/hei_nw/__init__.py
> src/hei_nw/models/base.py
> src/hei_nw/datasets/{scenario_a.py,scenario_b.py,scenario_c.py,scenario_d.py,scenario_e.py}
> src/hei_nw/eval/harness.py
> src/hei_nw/eval/report.py
> src/hei_nw/baselines/{long_context.py,rag.py}
> src/hei_nw/metrics/{text.py,compute.py,timing.py}
> src/hei_nw/utils/{seed.py,io.py,cli.py}
> tests/{...}
> scripts/{run_b0_small.sh,make_report.sh}
> reports/baseline/  (generated)
> ```
>
> **Config:** keep simple `argparse`; hydra may be introduced later.

### M0-T1 \[CODEX] Initialize package, configs, and CI plumbing

* **Goal:** Create a clean Python package with QA tooling configured.
* **Key changes:**

  1. `pyproject.toml` (black, ruff, mypy config), `src/hei_nw/__init__.py`.
  2. `src/hei_nw/utils/seed.py` (global seed set for PyTorch/NumPy/random).
  3. `src/hei_nw/utils/io.py` (safe JSON/Markdown writes; `timestamp_slug()`).
  4. `src/hei_nw/utils/cli.py` (common `--seed`, `--device`, `--outdir` parsing).
  5. Add `.github/workflows/ci.yml` to run **format/lint/type/tests** on PR.
* **Tests:**

  * `tests/test_utils.py::test_seed_determinism`
  * `tests/test_io.py::test_write_and_read_json_roundtrip`
  * Coverage ≥85% on new utils.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  python -c "import hei_nw, json; print('ok')"
  ```

### M0-T2 \[CODEX] Base model loader (Qwen2.5-1.5B-Instruct) + generation defaults

* **Goal:** Load the base decoder-only LLM and expose a simple `generate()` API with fixed defaults.
* **Key changes:**

  * `src/hei_nw/models/base.py`

    * `load_base(model_id="Qwen/Qwen2.5-1.5B-Instruct", dtype="auto", quant_4bit=True)`
    * `generate(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9, do_sample=False, stop=None)`
    * Track **prompt\_tokens**, **generated\_tokens**, and return text + token counts.
* **Tests:**

  * `tests/test_models_base.py::test_tokenizer_roundtrip_small`
  * `tests/test_models_base.py::test_generate_count_tokens_smoke` (mark `slow` but keep a CPU tiny call like `max_new_tokens=8`).
* **Quality gates:** same as above.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.models.base import load_base,generate
  tok,model,pipe=load_base(quant_4bit=True)
  out=generate("Hello", max_new_tokens=8)
  print(out["text"][:20], out["prompt_tokens"], out["generated_tokens"])
  PY
  ```

### M0-T3 \[CODEX] Scenario A generator (+ hard negatives, lag bins)

* **Goal:** Implement **episodic one-shot** story generator with **hard negatives** and lag annotations.
* **Key changes:**

  * `src/hei_nw/datasets/scenario_a.py`

    * `generate(n, seed, confounders_ratio=1.0, hard_negative=True, lag_spec={"bins":[0,1,3,7,30]})`
    * Output records: `{episode_text, cues:[...], answers:[...], should_remember:bool, lag:int, group_id}`
      *Hard negatives:* for each episode, synthesize distractors sharing ≥2/3 slots (name/place/item/time).
* **Tests:**

  * `tests/test_scenario_a.py::test_shapes_and_fields`
  * `tests/test_scenario_a.py::test_hard_negative_rate`
  * `tests/test_scenario_a.py::test_lag_bins_cover`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.datasets.scenario_a import generate
  ds=generate(n=64, seed=7)
  print(len(ds), "items", "hardneg?", any('Distractors' in str(x) or True for x in ds))
  PY
  ```

### M0-T4 \[CODEX] Scenario B–E generators (functional, labeled)

* **Goal:** Implement remaining synthetic generators per `validation-plan.md`.
* **Key changes:**

  * `src/hei_nw/datasets/scenario_b.py` (**Hot-patch** contradictions)
  * `src/hei_nw/datasets/scenario_c.py` (**Preference/Config** toggles)
  * `src/hei_nw/datasets/scenario_d.py` (**Stress-interference** mixtures)
  * `src/hei_nw/datasets/scenario_e.py` (**Long-context control**)
    Each returns list of dicts with: `{context, query, expected, should_remember, lag?}` where applicable.
* **Tests:**

  * `tests/test_scenarios_rest.py::test_each_generator_min_sizes`
  * `tests/test_scenarios_rest.py::test_should_remember_present`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.datasets import scenario_b,scenario_c,scenario_d,scenario_e
  sizes=[len(scenario_b.generate(10,0)),len(scenario_c.generate(10,0)),len(scenario_d.generate(10,0)),len(scenario_e.generate(10,0))]
  print("sizes", sizes)
  PY
  ```

### M0-T5 \[CODEX] Text metrics (EM/F1/recall\@k) and timing scaffolds

* **Goal:** Provide metric functions and latency timer; recall\@k present (may be 0/None under B0).
* **Key changes:**

  * `src/hei_nw/metrics/text.py` → `exact_match`, `token_f1` (simple whitespace tokenization), `recall_at_k` (accepts candidate list).
  * `src/hei_nw/metrics/timing.py` → context manager to record wall-clock latency per query.
* **Tests:**

  * `tests/test_metrics_text.py::test_em_and_f1_basic`
  * `tests/test_metrics_text.py::test_recall_at_k_shape`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.metrics.text import exact_match,token_f1
  print(exact_match("a","a"), token_f1("a b","a c"))
  PY
  ```

### M0-T6 \[CODEX] Compute metrics scaffolds (FLOPs / KV-cache)

* **Goal:** Emit compute footprints schema and estimates for **no-memory** and **long-context** baselines.
* **Key changes:**

  * `src/hei_nw/metrics/compute.py`

    * `estimate_attention_flops(prompt_toks, gen_toks, n_layers, d_model, heads)` (big-O constant can be documented)
    * `estimate_kv_bytes(tokens, d_model, dtype="fp16")`
    * `ComputeRecord` (pydantic) with optional fields allowed; **keys always present**.
* **Tests:**

  * `tests/test_compute.py::test_schema_keys_present`
  * `tests/test_compute.py::test_monotonicity_tokens`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.metrics.compute import estimate_attention_flops, estimate_kv_bytes
  print(estimate_attention_flops(100,50,24,1024,16), estimate_kv_bytes(150,1024))
  PY
  ```

### M0-T7 \[CODEX] Long-context baseline module

* **Goal:** Provide a working **long-context** baseline that concatenates relevant episodic snippets to the prompt and returns compute stats.
* **Key changes:**

  * `src/hei_nw/baselines/long_context.py`

    * `build_context(record)` (pack episode/context)
    * `run_long_context(model, tok, records, gen_cfg) -> results`
    * Capture prompt/gen token counts per item and aggregate compute metrics via `metrics.compute`.
* **Tests:**

  * `tests/test_long_context.py::test_pack_context_not_empty`
  * `tests/test_long_context.py::test_returns_compute_fields`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.baselines.long_context import build_context
  print(len(build_context({"context":"X", "query":"Q", "expected":"X"}))>0)
  PY
  ```

### M0-T8 \[CODEX] RAG baseline interface (functional, tiny default)

* **Goal:** Wire a **RAG** baseline interface (search retrieves candidate snippets); default **HFEmbedder** implementation, pluggable.
* **Key changes:**

  * `src/hei_nw/baselines/rag.py`

    * `class Embedder(Protocol)`, `HFEmbedder(model_id="intfloat/e5-small-v2")` (uses `transformers`), `FaissIndex` utility.
    * `run_rag(model, tok, records, embedder=None, k=5)` (returns answers + recall\@k, compute fields present).
    * Tests will use a **ToyEmbedder** defined **inside tests** (not a stub in library) to avoid downloads.
* **Tests:**

  * `tests/test_rag.py::test_index_and_query_toyembedder`
  * Mark actual HF embedder test as `slow`.
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  pytest -q tests/test_rag.py::test_index_and_query_toyembedder
  ```

### M0-T9 \[CODEX] Eval harness CLI (`--mode`, `--scenario`, JSON+MD reports)

* **Goal:** End-to-end **B0** evaluation pipeline with reporting.
* **Key changes:**

  * `src/hei_nw/eval/harness.py`

    * CLI: `--mode {B0,B1,B2,B3} --scenario {A,B,C,D,E} --n INT --seed INT --baseline {none,long-context,rag} --outdir PATH`
    * For **B0**: build prompt from scenario record, call `generate`, compute EM/F1 & latency; include recall\@k field (=None).
    * If `--baseline long-context`, use `baselines.long_context.run_long_context` for compute footprints (also store B0’s).
    * For **non-B0 modes** in M0: **graceful error with help text and exit code 64** (not `NotImplementedError`).
  * `src/hei_nw/eval/report.py`

    * Write **JSON** (per-item + aggregate) and **Markdown** summary: metrics, time-lag binned EM/Recall\@k curves (**scaffolded**).
* **Tests:**

  * `tests/test_harness_cli.py::test_cli_b0_smoke` (use `-n 4` small)
  * `tests/test_report_md.py::test_md_contains_required_sections`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python -m hei_nw.eval.harness --mode B0 --scenario A --n 8 --seed 7 --outdir reports/baseline
  ```

### M0-T10 \[CODEX] Time-lag robustness scaffolding

* **Goal:** Bin results by `lag` and emit per-bin EM/Recall\@k in JSON + include in MD report.
* **Key changes:**

  * Extend `report.py` with `bin_by_lag(records, bins)` and aggregate table.
* **Tests:**

  * `tests/test_lag.py::test_bin_edges_and_counts`
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  python -m hei_nw.eval.harness --mode B0 --scenario A --n 32 --outdir reports/baseline && \
  grep -q "Lag bin" reports/baseline/*A*.md
  ```

### M0-T11 \[CODEX] Scripts and README updates

* **Goal:** Provide one-shot scripts to reproduce M0 and update README.
* **Key changes:**

  * `scripts/run_b0_small.sh` :

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    OUT=reports/baseline
    python -m hei_nw.eval.harness --mode B0 --scenario A --n 64 --seed 7 --outdir "$OUT"
    python -m hei_nw.eval.harness --mode B0 --scenario B --n 64 --seed 7 --outdir "$OUT"
    python -m hei_nw.eval.harness --mode B0 --scenario C --n 64 --seed 7 --outdir "$OUT"
    python -m hei_nw.eval.harness --mode B0 --scenario D --n 64 --seed 7 --outdir "$OUT"
    python -m hei_nw.eval.harness --mode B0 --scenario E --n 64 --seed 7 --outdir "$OUT" --baseline long-context
    ```
  * `scripts/make_report.sh` (optionally concatenate MDs).
  * Update `README.md` with usage examples.
* **Tests:** none beyond smoke; covered by CLI test.
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  bash scripts/run_b0_small.sh && ls reports/baseline | wc -l
  ```

### M0-T12 \[CODEX] Integration test (imports + end-to-end tiny run)

* **Goal:** Ensure all modules import and a tiny E2E B0 run completes.
* **Key changes:**

  * `tests/test_e2e.py::test_b0_scenario_a_tiny` (n=4; asserts JSON+MD exist; modules imported).
* **Tests:** the above.
* **Quality gates:** as above.
* **Acceptance check:**

  ```bash
  pytest -q tests/test_e2e.py::test_b0_scenario_a_tiny
  ```

### M0-T13 \[CODEX] Repo-wide **no-stubs** guard

* **Goal:** Enforce “no stubs/mocks” policy.
* **Key changes:**

  * Add CI step to run:

    ```bash
    git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."
    ```
* **Tests:** none.
* **Quality gates:** CI passes.
* **Acceptance check:**
  CI shows “No stubs.”

---

## 4) \[HUMAN/ChatGPT] Review & GPU Tasks

1. **H1 — Design/plan review**

   * Open PR **“M0 — Repo, Baselines, and Harness (B0)”** and verify tasks M0-T1…M0-T13 present.
   * ✅ Success = All QA gates green in CI and “No stubs.” message.
2. **H2 — GPU baseline runs (B0)** *(single 16GB GPU)*

   1. Env: `bash codex-env/setup.sh`
   2. Run small suite:

      ```bash
      CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_TELEMETRY=1 \
      python -m hei_nw.eval.harness --mode B0 --scenario A --n 512 --seed 42 --outdir reports/baseline
      python -m hei_nw.eval.harness --mode B0 --scenario E --n 512 --seed 42 --outdir reports/baseline --baseline long-context
      ```
   3. ✅ Success signals:

      * JSON + MD created under `reports/baseline/`
      * MD shows **EM/F1, latency** tables; compute fields present; **long-context vs no-memory** footprint numbers visible.
3. **H3 — Sanity inspect Scenario A hard negatives**

   * Open the generated MD for Scenario A; check section mentioning **hard negatives/confounders** present in description.
   * ✅ Success = At least one run indicates distractors were used; EM on B0 **<** EM on the subset with no confounders (spot-check via an extra run with `--n 64` after temporarily disabling hard negatives using generator flag).

---

## 5) Deliverables & Artifacts

* **Code:** under `src/hei_nw/...` as listed.
* **CLI:** `python -m hei_nw.eval.harness --mode B0 --scenario {A..E} --n N --outdir reports/baseline [--baseline long-context|rag]`.
* **Datasets:** in-memory synthetic; optional dumps in `reports/baseline/*_records.jsonl` if `--dump` flag (optional).
* **Reports:**

  * JSON metrics: `reports/baseline/{ts}_{scenario}_B0_metrics.json`
  * Markdown report: `reports/baseline/{ts}_{scenario}_B0_report.md` (includes **time-lag** binned table and **compute** sections).
* **Scripts:** `scripts/run_b0_small.sh`, `scripts/make_report.sh`.
* **CI:** `.github/workflows/ci.yml`.

---

## 6) Definition of Done (DoD) Checklist

1. **Harness:** `--mode B0` works; **metric JSON + Markdown report** produced.

   * Verified by **M0-T9**, **M0-T12**, **H2**.
2. **Data:** Generators for **A–E** exist and can emit **≥500 episodic items per condition** when requested.

   * Verified by **M0-T3/T4**; Human run (**H2**) uses `--n 512`.
3. **Metrics:** **EM/F1 and latency** appear in reports; **compute metrics fields exist** (may be empty/None).

   * Verified by **M0-T5/T6/T9** (reports show keys even if 0/None).
4. **Hard negatives for A:** Generator includes **confounders** semantically close.

   * Verified by **M0-T3** tests and **H3** manual check.
5. **Robustness vs time-lag scaffolding:** Reports include **lag-binned** metric table/section.

   * Verified by **M0-T10** + **H2** (visual presence).
6. **Compute footprints:** Report includes **baseline compute footprints** contrasting **long-context** vs **no-memory**.

   * Verified by **M0-T6/T7/T9** and **H2**.
7. **No stubs/mocks:** Repo passes grep guard.

   * Verified by **M0-T13** (CI step).
8. **QA:** All quality gates pass in CI.

   * See next section.

---

## 7) QA Gates & CI Commands

* **Formatting:** `black .` (no diff)
* **Lint:** `ruff .` (no errors)
* **Types:** `mypy .` (no new issues)
* **Tests (CI fast path):** `pytest -q -m "not slow" --cov=src --cov-report=term-missing`
* **Local slow tests (opt-in):** `pytest -q -m slow`
* **Coverage:** ≥85% on changed lines (pytest-cov configured).
* **No-stubs grep:**

  ```bash
  git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."
  ```

---

## 8) Risks & Mitigations

1. **Model load / GPU memory issues** with Qwen2.5-1.5B-Instruct.

   * *Mitigations:* default to **4-bit** (`bitsandbytes`) and `torch_dtype="bfloat16"`/`float16` fallback; expose `--dtype` and `--quant_4bit` flags; keep `max_new_tokens` small in smoke.
2. **RAG baseline dependency bloat / network download in CI.**

   * *Mitigations:* RAG tests use an in-test **ToyEmbedder**; actual HF embedder test marked `slow`. CI runs `-m "not slow"`.
3. **Hard-negative quality too weak, masking confounding effect.**

   * *Mitigations:* deterministic **slot templates** with overlap constraints (≥2/3 slot match) + unit test asserting hard-negative rate; manual **H3** spot-check.

---

### Notes tying to planning docs

* **Modes:** B0 implemented per `validation-plan.md` (base-only). CLI advertises B1–B3 but exits gracefully until later milestones (not stubs, just unsupported in M0).
* **Scenarios A–E:** All **generators implemented now** with `should_remember` labels and **lag** fields where relevant; Scenario A includes **hard negatives**.
* **Metrics:** EM/F1, latency now; **compute** scaffold present (keys always included); **recall\@k** field present (may be None for B0).
* **Artifacts:** Conform exactly to `project-plan.md` (**eval harness**, **datasets/** generators, **reports/baseline/** with B0 numbers).
