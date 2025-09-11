# M1 Adapter Notes

This milestone introduces a read-only episodic adapter and memory token packer.
The adapter is a slim cross-attention module that returns its input unchanged
when no memory tokens are provided. The packer deterministically formats
simple episode traces into token IDs with a configurable cap.

For M2 the adapter will consume real memory tokens. The generation API already
accepts `mem_tokens` and an optional adapter, so integration will primarily
involve populating these tokens from a persistent store and enabling the
attention path.
