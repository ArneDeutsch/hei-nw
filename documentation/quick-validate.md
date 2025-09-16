# Quick Validate: Baseline (B0) + M2 Acceptance (B1 vs B0)

```bash
# 0) (one time) install deps
python -m pip install --upgrade pip
pip install -r codex-env/requirements.txt

# 1) clean old artifacts (optional but recommended)
rm -rf reports/baseline reports/m2-retrieval-stack

# 2) baseline: B0 across scenarios A–E (defaults to Qwen in loader)
export PYTHONPATH=src
bash scripts/run_b0_small.sh

# 3) combine the baseline markdowns (optional convenience)
bash scripts/make_report.sh reports/baseline reports/baseline/combined_report.md
# -> outputs:
#    reports/baseline/*_B0_metrics.json
#    reports/baseline/*_B0_report.md
#    reports/baseline/combined_report.md

# 4) M2 acceptance: Scenario A — B0, B1, and B1(no-hopfield) with Qwen
#    (N defaults to 24; you can bump to 64 if lift is noisy).
#    The script pins the short-answer chat prompt, Hopfield (steps=2, T=0.5),
#    removes the explicit newline stop, and gives short answers more headroom.
bash scripts/run_m2_retrieval.sh
# -> outputs in reports/m2-retrieval-stack/ :
#    A_B0_metrics.json, A_B0_report.md
#    A_B1_metrics.json, A_B1_report.md
#    A_B1_no-hopfield_metrics.json, A_B1_no-hopfield_report.md
#    completion_ablation.png

# 5) gate: relaxed-EM uplift must be ≥ +0.30 on the small set (exit 0 on success).
#    The gate also prints the strict-EM values for reference.
bash scripts/compare_b0_b1_m2.sh

# 6) (optional) make a combined file for M2 too
bash scripts/make_report.sh reports/m2-retrieval-stack reports/m2-retrieval-stack/combined_report.md
```

## If the uplift is borderline (stability pass)

```bash
# Re-run M2 with more samples; accepts env vars:
N=64 bash scripts/run_m2_retrieval.sh
bash scripts/compare_b0_b1_m2.sh
```

## Optional: fast smoke (tiny model; no quality gate)

```bash
# CPU-friendly pipeline sanity (no EM-lift requirement, just artifacts exist)
bash scripts/run_m2_retrieval_ci.sh
```

---

## Outputs to check (at a glance)

* **Baseline:** `reports/baseline/…` (A–E)

  * `*_B0_metrics.json`, `*_B0_report.md`, plus `combined_report.md`.
* **M2 (Scenario A):** `reports/m2-retrieval-stack/`

  * `A_B0_metrics.json`, `A_B1_metrics.json`
  * `A_B1_no-hopfield_metrics.json`
  * `completion_ablation.png`
  * `combined_report.md` (if you ran step 6)
* **Gate result:** `bash scripts/compare_b0_b1_m2.sh` prints

  * `EM lift +0.3xx >= 0.30` and **exits 0**.

---

## Notes

* All commands assume **repo root** and `export PYTHONPATH=src`.
* `run_m2_retrieval.sh` now **defaults to `Qwen/Qwen2.5-1.5B-Instruct`** and honors `MODEL`, `N`, `SEED`, and `OUT` while
  forcing Scenario A to use the chat short-answer prompt (`max_new_tokens=16`, no explicit stop).
* Scenario **A** picks the QA defaults automatically; other scenarios remain on the plain prompt unless overridden.
* `run_m2_retrieval_ci.sh` is **pinned to the tiny test model** and is **not** used for acceptance.
* If Hugging Face prompts for auth on first run, log in separately (`huggingface-cli login`) and rerun.
