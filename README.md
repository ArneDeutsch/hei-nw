# HEI-NW (Hippocampal Episodic Index — Neuromodulated Writes)

**HEI-NW** is a *memory add-on* for an LLM. You keep the base Transformer mostly as-is, and bolt on a small adapter plus a persistent “episodic store.” The system decides *when* to write a fresh memory (based on surprise/novelty/reward), *how* to index it (sparse key for low interference), *how* to recall from partial cues (associative lookup/completion), and *when* to distill the best memories back into the model weights via offline replay. Think: **LLM + smart, content-addressable cache with a write policy and nightly consolidation.**&#x20;

---

# Key artifacts
- [documentation/hei-nw.md](documentation/hei-nw.md) - short introduction to the algorithm
- [planning/design.md](planning/design.md) - the design of the implementation
- [planning/validation-plan.md](planning/validation-plan.md) - the plan how to validate the results of HEI-NW
- [research/experiment-synthesis.md](research/experiment-synthesis.md) - the research document the algorithm originates from
