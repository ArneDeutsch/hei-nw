You are a senior engineering planner tasked to produce an **executable, review‑ready task list** for a single HEI‑NW milestone.

**Milestone to plan:** {{MILESTONE_TITLE}}

**Repository ZIP in this chat:** "hei-nw-main.zip"

### Context and expectations
- This project implements and validates **HEI‑NW** (Hippocampal Episodic Index — Neuromodulated Writes) as an augmentation to a decoder‑only LLM.
- The repository includes definitive planning docs:
  - `planning/design.md` — system design & components
  - `planning/validation-plan.md` — scenarios, metrics, acceptance modes (B0–B3)
  - `planning/project-plan.md` — milestone scopes and DoD
- **Codex Web** will implement all code, tests, and refactors. **Human/ChatGPT** will review, run GPU‑only jobs, and verify acceptance criteria.
- Codex tends to leave **stubs/mocks**; do not allow this. All tasks must end with **real implementations** that run end‑to‑end. If a stub is *temporarily* needed, include an explicit task in the same milestone to **replace/remove** it before DoD.

### What you must do
1) **Unzip & read the repo** (assume the ZIP is attached). Skim the tree and open the three planning docs above to ground yourself.
2) From `project-plan.md`, locate **{{MILESTONE_TITLE}}** and parse its *Scope*, *DoD/Acceptance*, and *Artifacts*.
3) From `design.md` and `validation-plan.md`, pull all requirements that constrain this milestone (APIs, modes B0–B3, scenarios A–E, metrics, replay/decay constraints, etc.).
4) Produce a **single Markdown plan** with:
   - A short **Milestone Summary** (1–3 bullets).
   - A **Dependency/Inputs** list (what must exist first; exact files/APIs).
   - Two task sections:
     - **[CODEX] Implementation Tasks** — atomic, ordered, self‑contained tasks with enough context to “just code it”.
     - **[HUMAN/ChatGPT] Review & GPU Tasks** — minimal steps to verify the milestone and run any heavy experiments.
   - A **Definition of Done (DoD) Checklist** mapping directly to `project-plan.md` acceptance bullets.
   - **Deliverables & Artifacts** — concrete file paths, scripts, and report locations to be produced.
   - **QA Gates & CI** — commands and thresholds (formatting, linting, typing, tests, coverage).
   - **Risks & Mitigations** — top 3 risks with pre‑emptive checks.

### Formatting rules for the output
- Use **nested numbered lists** for task breakdowns.
- Prefix each task with a stable identifier: `M{{n}}-T{{k}}` where `{{n}}` is the milestone number you infer from the title.
- Label tasks clearly with `[CODEX]` or `[HUMAN]`.
- For every **[CODEX]** task, include these subsections:
  - **Goal:** one‑sentence intent.
  - **Key changes:** expected files/modules/classes/functions (propose paths if new).
  - **Tests:** unit/integ tests to add (with `pytest` node names), required fixtures, and coverage target.
  - **Quality gates:** exact commands that must pass (see below).
  - **Acceptance check:** how reviewers confirm the task is really done, with a single shell command or CLI invocation where possible.
- For every **[HUMAN]** task, keep it **short and unambiguous** (≤3 sub‑steps), focusing on reviews, GPU runs, and visual checks. Provide the exact command lines and the expected success signal.

### Guardrails to prevent stubs/mocks
- **Never** leave `pass`, `TODO`, or placeholder return values in committed code.
- If a temporary seam is required, add a same‑milestone task “Replace temporary stub XYZ with real implementation” and reference the commit that introduced it.
- Include a repository‑wide grep step in DoD:
  - `git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."`
- All public APIs must have **docstrings**, and all new modules must be imported at least once in an integration test.

### Standard toolchain (assume Python)
- **Runtime & libs:** See `codex-env/requirements.txt` for `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `faiss`, `hydra-core`, `pydantic`, etc.
- **Quality gates (exact):**
  - Format: `black .` (no diff)
  - Lint: `ruff .` (no errors)
  - Types: `mypy .` (no new issues)
  - Tests: `pytest -q` with **≥85%** coverage on changed lines; also run `pytest -m "not slow"` in CI and `pytest -m slow` locally if applicable.
- **Reports:** Store evaluation artifacts under `reports/{{milestone_slug}}/` and baseline metrics under `reports/baseline/` when relevant.

### Branching & PR hygiene
- Work on a feature branch: `feat/m{{n}}-<slug>`.
- Mandate a single PR titled `M{{n}} — {{MILESTONE_TITLE}}` with a checklist mirroring the DoD.
- Require at least one **[HUMAN]** review before merge.

### Expected structure of your final output
1) **Milestone Summary**
2) **Dependencies / Inputs**
3) **[CODEX] Implementation Tasks** (M{{n}}-T1 …)
4) **[HUMAN/ChatGPT] Review & GPU Tasks**
5) **Deliverables & Artifacts**
6) **Definition of Done (DoD) Checklist**
7) **QA Gates & CI Commands**
8) **Risks & Mitigations**

### Hints tying back to the HEI‑NW docs
- If the milestone mentions modes `B0…B3`, ensure the harness supports `--mode {B0,B1,B2,B3}` and that scenario generators A–E exist or are stubbed **only for the exact scope claimed**—and then fully implemented before DoD.
- If the milestone touches memory components (DG keyer, associative store, write gate, replay/decay/consolidation), map each to concrete classes/modules and add micro‑benchmarks or invariants (e.g., pattern separation tests).
- For validation work, always include metrics (EM/F1, recall@k, latency, FLOPs/KV cache stats) and an HTML/Markdown report artifact path.

### Start by inspecting the repo
(You may describe briefly what you read; do not paste large blobs.)
- Unzip: `unzip -o "hei-nw-main.zip" -d .`
- Inspect tree (summarize): `tree -a -I '.git|__pycache__' hei-nw-main | head -n 200`
- Open and summarize: `planning/project-plan.md`, `planning/design.md`, `planning/validation-plan.md`.

Now generate the plan.
