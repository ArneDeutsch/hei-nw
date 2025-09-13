# HEI-NW (Hippocampal Episodic Index — Neuromodulated Writes)

**HEI-NW** is a *memory add-on* for an LLM. You keep the base Transformer mostly as-is, and bolt on a small adapter plus a persistent “episodic store.” The system decides *when* to write a fresh memory (based on surprise/novelty/reward), *how* to index it (sparse key for low interference), *how* to recall from partial cues (associative lookup/completion), and *when* to distill the best memories back into the model weights via offline replay. Think: **LLM + smart, content-addressable cache with a write policy and nightly consolidation.**&#x20;

# Key artifacts
- [documentation/hei-nw.md](documentation/hei-nw.md) - short introduction to the algorithm
- [documentation/quick-validate.md](documentation/quick-validate.md) - bash command to run validation command on the algorithms (with GPU machine)
- [planning/design.md](planning/design.md) - the design of the implementation
- [planning/project-plan.md](planning/project-plan.md) - The plan we follow to implement and validate the HEI-NW
- [planning/validation-plan.md](planning/validation-plan.md) - the plan how to validate the results of HEI-NW
- [research/experiment-synthesis.md](research/experiment-synthesis.md) - the research document the algorithm originates from

## Project Structure

```text
.
├── codex-env/               - Environment setup scripts and requirements
├── documentation/           - Additional documentation and guides
├── models/                  - Pre-trained model assets (do not edit)
├── planning/                - Design documents and implementation plans
├── prompts/                 - Prompt templates used in development
├── research/                - Research notes and background material
├── src/                     - Source code for HEI-NW
├── tests/                   - Automated test suite
├── AGENTS.md                - Guidelines and project instructions
├── LICENSE                  - Project licensing information
├── pyproject.toml           - Python package and build configuration
└── README.md                - Overview and navigation
```

## Usage

Run a small baseline across scenarios A–E and generate reports:

```bash
bash scripts/run_b0_small.sh
```

The evaluation harness writes reports to `reports/baseline/` by default.

To combine individual Markdown reports into a single file:

```bash
bash scripts/make_report.sh
```

### Model selection for M2

- Use `scripts/run_m2_retrieval.sh` for **acceptance runs**. It defaults to
  **Qwen/Qwen2.5-1.5B-Instruct** and is the only supported way to measure the
  **B1 − B0 EM uplift**.
- Use `scripts/run_m2_retrieval_ci.sh` for **CI/smoke**. It pins a tiny GPT-2
  test model and only checks that reports/metrics exist (no EM-lift check).
