Short answer: you’re right—the **tiny GPT-2** in `tests/models/tiny-gpt2/` is only suitable for unit/smoke checks. It’s not an LLM you can use to **measure** uplift. For the **M2 review & GPU tasks**, the plan should explicitly use a *real* base model—**`Qwen/Qwen2.5-1.5B-Instruct`** (which is already the default in our loader)—and the acceptance gate “**B1 − B0 ≥ +0.30 EM** on the small set” should be evaluated **with Qwen**, not the tiny model.

Here’s the evidence and the fixes:

## What the repo actually says/does

* The **M2 plan** mixes tiny-model smoke commands with a hard acceptance gate: it runs B0/B1 with `--model tests/models/tiny-gpt2` and then checks for “**printed lift ≥ +0.30 EM**” on the small set. See `planning/milestone-2-plan.md` (acceptance #4 and the `scripts/run_m2_retrieval.sh` snippet).
* The same plan **acknowledges** the risk: “**No EM lift with tiny model** (signal too weak).” (M2 Risks & Mitigations §8.)
* Our **base loader** defaults to **`Qwen/Qwen2.5-1.5B-Instruct`** (`src/hei_nw/models/base.py`), and **M0** selects that class of model in the project/validation docs. The baseline script `scripts/run_b0_small.sh` even omits `--model`, so it will use Qwen by default.

## Conclusion

* The milestone text *implicitly* assumed tiny model for fast smoke runs but then used **that same run** to enforce the **+0.30 EM** success gate. That’s inconsistent with our own risk note and with the validation plan’s intent.
* For the **HUMAN/GPU review**, the model must be **Qwen/Qwen2.5-1.5B-Instruct** (4-bit is fine). Use the tiny model **only** for CI/unit/smoke (e.g., checking the retrieval section is present, ablation plot exists, metrics are finite).

## What to change (minimal, concrete)

1. **Update M2 plan wording** (in `planning/milestone-2-plan.md`):

   * Clarify two tracks:

     * **CI/Smoke:** `--model tests/models/tiny-gpt2` → success = files/metrics exist, *no* EM-lift requirement.
     * **HUMAN/GPU Acceptance:** **use Qwen**; success = **B1−B0 ≥ +0.30 EM** on the small set; ablation plot produced; retrieval metrics finite.
   * Keep the existing risk note but make it explicit: “Tiny model is not used for acceptance.”

2. **Make the run scripts model-selectable**:

   * Change `scripts/run_m2_retrieval.sh` to:

     ```bash
     MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
     ```

     (so CI can export `MODEL=tests/models/tiny-gpt2` while human runs don’t need to set anything).
   * Leave `scripts/compare_b0_b1_m2.sh` as the acceptance gate (it already checks `EM lift >= 0.30`).

3. **Where to regenerate reports**

   * **Recreate only the M2 B0/B1 reports** with Qwen:

     ```
     export PYTHONPATH=src
     # optional: bump n from 24 to 64 for a slightly stabler small-set check
     python -m hei_nw.eval.harness --mode B0 --scenario A -n 24 --seed 7 --outdir reports/m2-retrieval-stack
     python -m hei_nw.eval.harness --mode B1 --scenario A -n 24 --seed 7 --outdir reports/m2-retrieval-stack
     bash scripts/compare_b0_b1_m2.sh
     ```

     That satisfies the “exit 0 + printed lift ≥ +0.30 EM” success signal—**with Qwen**.
   * **Do not** recreate the general **baseline** (`reports/baseline/`) unless you changed generation settings; that script already defaults to Qwen.

## Optional niceties (make it hard to misuse)

* Add a second helper script: `scripts/run_m2_retrieval_ci.sh` that pins `MODEL=tests/models/tiny-gpt2`, and keep `scripts/run_m2_retrieval.sh` for GPU/Qwen.
* In the M2 plan’s “Acceptance checks,” remove explicit `--model models/tiny-gpt2` from examples; rely on the default (Qwen) or show `MODEL=...` override.

## Bottom line

* **Model for review/acceptance:** **`Qwen/Qwen2.5-1.5B-Instruct`**.
* **Tiny GPT-2:** unit tests & smoke only; not valid for measuring EM uplift.
* **Yes, update the M2 plan** to state this, and **regenerate the M2 B0/B1 reports** with Qwen to satisfy the +0.30 EM gate. No need to redo unrelated baselines.

---

Here’s a **safe, executable Codex task list** to implement the model-policy fix for M2 (Qwen for acceptance; tiny-gpt2 for CI/smoke), plus tests and docs so this doesn’t regress.

---

# Codex Task List — M2 Model Policy Fix

> Branch: `chore/m2-model-policy-qwen`
> Run before committing: `export PYTHONPATH=src && pytest -q`

## T1 — Parameterize `run_m2_retrieval.sh` to default to Qwen

**Why:** Acceptance (“EM lift ≥ +0.30”) must be measured with a real LLM.
**Edit:** `scripts/run_m2_retrieval.sh`

**Replace file contents with:**

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT="${OUT:-reports/m2-retrieval-stack}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-24}"
SEED="${SEED:-7}"

python -m hei_nw.eval.harness --mode B0 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" --no-hopfield
```

**Also:** `chmod +x scripts/run_m2_retrieval.sh`

**Acceptance (local, CPU only):**

* `bash -n scripts/run_m2_retrieval.sh` exits 0.
* `pytest -q tests/utils/test_scripts.py::test_m2_scripts_present_and_executable` passes.

---

## T2 — Add CI smoke script pinned to tiny model

**Why:** Keep fast, CPU-only smoke run; never use it for EM-lift measurement.
**Add:** `scripts/run_m2_retrieval_ci.sh`

**Create file with:**

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT="${OUT:-reports/m2-retrieval-stack}"
MODEL="tests/models/tiny-gpt2"
N="${N:-12}"
SEED="${SEED:-7}"

python -m hei_nw.eval.harness --mode B0 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" --no-hopfield
```

**Also:** `chmod +x scripts/run_m2_retrieval_ci.sh`

**Acceptance:**

* `bash -n scripts/run_m2_retrieval_ci.sh` exits 0.

---

## T3 — Strengthen tests to enforce the policy

**Why:** Prevent regressions (e.g., hard-coding tiny model in the acceptance script).
**Edit:** `tests/utils/test_scripts.py`

**Change the list in `test_m2_scripts_present_and_executable` to include the new CI script:**

```python
def test_m2_scripts_present_and_executable() -> None:
    scripts = [
        Path("scripts/run_m2_retrieval.sh"),
        Path("scripts/run_m2_retrieval_ci.sh"),  # added
        Path("scripts/compare_b0_b1_m2.sh"),
    ]
    ...
```

**Add:** `tests/utils/test_m2_model_policy.py`

```python
from pathlib import Path

def test_run_m2_defaults_to_qwen() -> None:
    text = Path("scripts/run_m2_retrieval.sh").read_text()
    assert 'MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"' in text

def test_ci_script_pins_tiny() -> None:
    text = Path("scripts/run_m2_retrieval_ci.sh").read_text()
    assert 'MODEL="tests/models/tiny-gpt2"' in text
```

**Acceptance:**

* `pytest -q tests/utils/test_scripts.py tests/utils/test_m2_model_policy.py` passes.

---

## T4 — Update M2 milestone plan to codify the two tracks

> NOTE: `AGENTS.md` currently says “Do not edit … planning …”. This task adds a narrow **exception** (see T5) and then updates the plan.

**Edit:** `planning/milestone-2-plan.md`

1. **Under** `## 4) [HUMAN/ChatGPT] Review & GPU Tasks`, **insert this block just after the “Sanity run (CPU ok with tiny model)” heading:**

```md
### Model policy (IMPORTANT)

- **CI/Smoke (tiny model):** `tests/models/tiny-gpt2` is used only to check that
  reports are produced and retrieval metrics fields are finite. **No EM-lift
  requirement** is evaluated on tiny.
- **HUMAN/GPU Acceptance (real LLM):** Use **Qwen/Qwen2.5-1.5B-Instruct**
  (quantized OK). The acceptance gate **B1 − B0 ≥ +0.30 EM (Scenario A, small set)**
  is evaluated **only** with Qwen.
- The helper scripts implement this:
  - `scripts/run_m2_retrieval_ci.sh` → tiny (smoke)
  - `scripts/run_m2_retrieval.sh` → defaults to Qwen (acceptance)
```

2. **In** `## 6) Definition of Done (DoD) Checklist`, **edit the first bullet to:**

```md
* **Quality lift (measured with Qwen/Qwen2.5-1.5B-Instruct):**
  On Scenario A small set, `B1 − B0 ≥ +30 EM` (≥ +0.30 absolute).
```

**Acceptance:**

* Grep shows both substrings present:

  * `Qwen/Qwen2.5-1.5B-Instruct`
  * `run_m2_retrieval_ci.sh`
* Document still builds as plain Markdown (no broken code blocks/anchors).

---

## T5 — Allow planning errata edits (narrow exception)

**Why:** AGENTS guardrail blocks the plan update; we need a safe, explicit exception.
**Edit:** `AGENTS.md` (append the exception to the bullet list)

**Add after the “Do not edit … planning …” bullet:**

```md
- **Exception:** When a milestone plan contains a factual error that blocks
  acceptance (e.g., using a tiny stub model for quality gates), you **may edit
  files under `planning/`** in a dedicated branch/PR that references the issue
  and includes tests asserting the corrected policy. This PR must be reviewed
  by HUMAN/ChatGPT before merge.
```

**Acceptance:**

* `grep -n "Exception.*planning" AGENTS.md` finds the new paragraph.

---

## T6 — Clarify README model usage so users run the right script

**Edit:** `README.md` (add a short subsection near “Project Structure” or “Running M2”)

**Insert:**

```md
### Model selection for M2

- Use `scripts/run_m2_retrieval.sh` for **acceptance runs**. It defaults to
  **Qwen/Qwen2.5-1.5B-Instruct** and is the only supported way to measure the
  **B1 − B0 EM uplift**.
- Use `scripts/run_m2_retrieval_ci.sh` for **CI/smoke**. It pins a tiny GPT-2
  test model and only checks that reports/metrics exist (no EM-lift check).
```

**Acceptance:**

* `grep -n "Model selection for M2" README.md` finds the heading.

---

## (Optional) T7 — CI wiring (safe stub)

**Only if desired:** add a CI step that **runs** the tiny smoke script without enforcing EM lift. (Skip if CI minutes are tight.)

* Add a job in `.github/workflows/ci.yml` to run:

  ```yaml
  - name: M2 smoke (tiny model)
    run: bash scripts/run_m2_retrieval_ci.sh
  ```
* Do **not** run the Qwen script in CI.

---

# Post-merge HUMAN/GPU checklist (for the reviewer; not for Codex)

1. Run acceptance with real model:

```bash
export PYTHONPATH=src
bash scripts/run_m2_retrieval.sh     # uses Qwen by default
bash scripts/compare_b0_b1_m2.sh     # must print "EM lift ... >= 0.30" and exit 0
```

2. If lift is noisy, you may increase `N`:

```bash
N=64 bash scripts/run_m2_retrieval.sh
bash scripts/compare_b0_b1_m2.sh
```

---

## Definition of Done (for this PR)

* `scripts/run_m2_retrieval.sh` defaults to Qwen via `MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"`.
* New `scripts/run_m2_retrieval_ci.sh` exists and pins `tests/models/tiny-gpt2`.
* Tests enforce the above policies and include the new script.
* `planning/milestone-2-plan.md` explicitly documents the two tracks and states the Qwen requirement in DoD.
* `AGENTS.md` includes the narrow exception allowing planning errata updates.
* `README.md` explains which script to use for acceptance vs. CI.
* `pytest -q` passes locally.

