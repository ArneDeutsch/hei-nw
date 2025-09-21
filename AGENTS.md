- Read [README.md](README.md) for a short overview over the project
- Read [documentation/python-guide.md](documentation/python-guide.md) before any coding and use the checklist for new code.
- Do not edit any files inside "research", "planning", "prompts", "reviews" or "models" folder.
- Our overall goal is
  - to realize the design described in [planning/design.md](planning/design.md)
  - validate the results as described in [planning/validation-plan.md](planning/validation-plan.md)
  - according to the project plan from [planning/project-plan.md](planning/project-plan.md)
- When running tests remember to set PYTHONPATH=src first
- Use the packaged `tests/models/tiny-gpt2` weights only for fast unit tests;
  real evaluation runs must rely on the default Qwen/Qwen2.5-1.5B-Instruct
  checkpoint which is already downloaded locally.
