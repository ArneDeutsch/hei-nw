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
