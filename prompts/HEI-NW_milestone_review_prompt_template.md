You are a **senior engineering reviewer**. Your job is to deliver a **critical, evidence‑backed milestone review** for the HEI‑NW project.

**Milestone to review:** {{MILESTONE_TITLE}}
**Repository ZIP in this chat:** "hei-nw-main.zip"

---

## Context you must honor
- HEI‑NW augments a **decoder‑only LLM** with episodic memory and consolidation. The repo contains definitive planning docs:
  - `planning/design.md` — system design & components
  - `planning/validation-plan.md` — scenarios, metrics, acceptance modes (B0–B3)
  - `planning/project-plan.md` — milestone scopes, artifacts, and Definition of Done (DoD)
- For each milestone, the **source of truth** for concrete tasks is its plan in `planning/` (e.g., `planning/milestone-0-plan.md`). If a per‑milestone plan file is missing or incomplete, defer to `project-plan.md` and the design/validation docs.
- **Codex Web** authored code, tests, and refactors. **Human/ChatGPT** performed reviews and any GPU‑only runs. Your review must detect stubs/mocks left behind and any divergence from the design & validation plan.

---

## What you must do (process)
1) **Unzip & read the repo** (assume the ZIP is attached): unarchive `hei-nw-main.zip` into `./hei-nw-main/` and skim the tree. Do not paste large blobs; summarize.
2) Open these files and **ground your review**:
   - `planning/project-plan.md`
   - `planning/design.md`
   - `planning/validation-plan.md`
   - The **milestone plan** for **{{MILESTONE_TITLE}}** in `planning/` (match by H1 title). If not found, use the section from `project-plan.md`.
3) From the milestone plan, collect:
   - **Scope** and explicit **DoD/Acceptance**.
   - The listed **Artifacts/Reports** that must exist at the end.
   - The full list of **[CODEX] Implementation Tasks** and **[HUMAN/ChatGPT] Tasks** (IDs like `M?-T#`), including their **Goal**, **Key changes**, **Tests**, **Quality gates**, and **Acceptance check** fields.
4) **Inspect the codebase** to verify each task’s implementation exists and is real (not a stub). Prefer evidence over opinion:
   - Confirm modules, functions, and CLIs promised by the task titles/Key changes exist in `src/`.
   - Inspect `tests/` for the exact test files and test cases the plan names.
   - Look for **stub anti‑patterns** (`pass`, `raise NotImplementedError`, `TODO`, `FIXME`, mocks left in production code).
   - Check for **wiring completeness** (imports, registration, CLI flags, config defaults).
   - Confirm **reports/metrics outputs** are implemented where the milestone claims they exist (JSON + Markdown, paths match plan).
5) **Cross‑check with the design & validation plan**:
   - Ensure the milestone’s code drives toward the architecture in `design.md` and honors the **mode semantics** B0–B3, scenario definitions, and metric choices in `validation-plan.md`.
   - Call out any **architectural drift** or **metrics mismatch**.
6) **Assess quality gates & CI**:
   - Presence of formatting/linting/type‑checking configs; presence of a CI workflow; tests naming and isolation; minimal public API signatures documented.
7) Produce a **verdict with gaps and follow‑ups**:
   - For each DoD item and each task, mark **Pass / Partial / Fail** with concrete evidence (file paths, function names, short excerpts ≤25 words).
   - If gaps exist, propose **follow‑up [CODEX] tasks** that match the milestone plan’s task format (Goal, Key changes, Tests, Quality gates, Acceptance check). Where GPU‑only checks are required, also propose **[HUMAN/ChatGPT] tasks**.

---

## Output format (use exactly these sections)

# {{MILESTONE_TITLE}} — Milestone Review

### 1) Executive summary (2–4 short paragraphs)
- What the milestone intended to deliver (Scope, DoD, key artifacts).
- What actually exists in the repo (high‑level).
- Overall verdict: **Pass / Partial / Fail** and why.

### 2) Evidence snapshot: repo & planning anchors
- **Repo tree (short)**: top directories and any key files you will reference.
- **Planning anchors used**: the exact headings/sections from `project-plan.md`, `design.md`, `validation-plan.md`, and the per‑milestone plan you matched (include H1/H2 text).
- **Assumptions/limits**: anything you could not validate due to missing files.

### 3) DoD / Acceptance verification table
Provide a compact table with rows = each DoD/Acceptance item, columns = *Item*, *Evidence (files/funcs/CLI)*, *Status (Pass/Partial/Fail)*, *Notes*.

### 4) Task‑by‑task review (mirror the milestone plan order)
For each task (e.g., `### M?-T# [CODEX] <title>` or `[HUMAN/ChatGPT]`):
- **Intent (from plan)**: one‑sentence paraphrase.
- **Findings (evidence)**: file paths, function/class/CLI names, short excerpts (≤25 words), test names.
- **Gaps / Risks**: missing pieces, stubs, mis‑wiring, perf risks.
- **Status**: **Pass / Partial / Fail** (justify briefly).

*(Repeat for all tasks in the plan.)*

### 5) Design & validation alignment
- **Design mapping**: Which `design.md` components this milestone implements or prepares (modules, data flow). Cite file paths.
- **Validation mapping**: Which `validation-plan.md` scenarios/metrics/modes are covered now; what remains deferred; any mismatches.

### 6) Quality & CI assessment
- Tooling present (formatter, linter, type checker), CI workflow(s), pre‑commit, code organization and naming.
- Testing depth: unit vs integration, deterministic seeds, dataset fixtures, flakiness risks.

### 7) Gaps and **Follow‑up tasks**
If anything is **Partial/Fail** (or critical risks), propose follow‑ups in the **same format** as the milestone plan. Use new IDs starting at `{{M?-F1}}` and keep them surgical and implementable **within the scope of this milestone**.

#### Example format (copy for each follow‑up task)
```
### {{M?-F#}} [CODEX] <concise title>

* **Goal:** <what outcome this fixes or proves>
* **Key changes:**
  1) <files/modules to add/modify with brief specifics>
  2) <configs/CLIs to add>
* **Tests:**
  - <exact test files and function names to add or extend>
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  <single command that proves the fix, with expected artifact path(s)>
  ```
```

(If human/GPU work is required, add a sibling task:)
```
### {{M?-F#H}} [HUMAN/ChatGPT] <concise title>

* **Goal:** <what the human run verifies>
* **Steps:** <repro steps on a developer box with GPU>
* **Acceptance check:** <what proof to attach back into the repo (e.g., report, log)>
```

### 8) Final verdict
- **Pass / Partial / Fail** with one‑line justification.
- If Partial/Fail, list the **minimum set of follow‑ups** (by ID) required to meet DoD for **{{MILESTONE_TITLE}}**.

---

## Practical tips for your review
- Prefer **concrete pointers** (paths, function names, test names) over generalities.
- Quote only short excerpts (≤25 words). Summarize otherwise.
- Watch for: leftover stubs/mocks, TODOs/FIXMEs, unreferenced functions, missing CLI flags, orphaned tests, broken report paths, doc/plan drift.
- If a required file or section is **missing**, mark its item **Fail** and include a follow‑up task to create it.
- If the milestone introduces modes (B0–B3) or scenarios (A–E), ensure the harness recognizes them and can emit the required metrics and markdown/JSON reports even if some are scaffolds (but real, not empty).

Now execute the review for **{{MILESTONE_TITLE}}**.
