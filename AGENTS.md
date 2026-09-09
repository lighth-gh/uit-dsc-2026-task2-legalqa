# AGENTS.md

## Purpose

This file gives coding agents the project context, workflow rules, and safety boundaries for working in this repository.

The main target agent is GPT-5.6 Sol / Codex-style coding assistance, but the rules are written in normal Markdown so they can also help other coding agents that read `AGENTS.md`.

---

## Core operating principle for GPT-5.6 Sol

**The explicit task is the lake, not the ocean.**

Do the exact task requested by the user. Treat the requested target, allowed files, allowed systems, and acceptance criteria as the boundary.

Words such as “complete”, “full”, “exhaustive”, “every”, “100%”, or “fix all” mean complete **inside the requested boundary**, not permission to widen the job into unrelated cleanup, speculative hardening, broad refactors, or new features.

When something is related but outside the task:

- mention it briefly in the handoff report,
- do not edit it automatically,
- ask before expanding scope.

---

## Communication style

Reply in Vietnamese unless the user asks otherwise.

Use a direct, practical style. Prefer clear diagnosis and concrete next actions over vague advice.

For code/log/debugging tasks, structure the answer like this when useful:

1. Diagnosis
2. Likely cause
3. Smallest safe fix
4. Patch or exact edit
5. Test command
6. Expected result
7. Remaining risk

Do not over-explain obvious things. Do explain trade-offs when they affect score, runtime, cost, or correctness.

---

## Scope discipline

Before editing, identify the exact scope:

- What file or module is in scope?
- What behavior is being changed?
- What output must remain unchanged?
- What test or check proves completion?

Default behavior:

- Fix the smallest relevant area.
- Do not rewrite unrelated modules.
- Do not rename public functions unless requested.
- Do not change file formats or submission schema silently.
- Do not perform broad cleanup unless it is required for the requested fix.

If you notice adjacent issues, report them under `Out-of-scope findings` at the end.

---

## Investigation boundary

Investigate enough evidence to identify the primary cause and its in-scope consequences.

Stop widening the investigation once:

- the primary cause is clear,
- the required fix is clear,
- one relevant verification path exists.

Continue investigating only if:

- a test fails,
- evidence contradicts the current diagnosis,
- the acceptance criteria cannot be verified,
- the user explicitly asks for deeper review.

Avoid scanning the whole repository when one or two files/logs are enough.

---

## Completion rule

A task is complete when:

- the requested artifact or fix is done,
- one clean relevant verification pass has been run or clearly specified,
- the result is reported,
- remaining risks are stated.

After verified completion, stop. Do not keep improving, refactoring, or hardening hypothetical failure modes unless the user asks.

Completeness still matters inside the requested scope. Do not use “bounded scope” as an excuse to skip a required workflow step, safety check, regression test, edge case, or error path.

---

## Project context

This repository is used for Kaggle / LegalQA / RAG-style experiments, especially Vietnamese legal question answering.

Common goals:

- Build a reliable retrieval pipeline.
- Combine BM25, dense retrieval, reranking, and generation.
- Improve answer quality without breaking runtime limits.
- Reduce slow stages such as reranking or generation.
- Produce valid submission files.
- Debug Kaggle notebook/runtime/path/GPU issues.

Common constraints:

- Code should run on Kaggle.
- Avoid hard-coded local machine paths.
- Use `/kaggle/input/...` for read-only input.
- Use `/kaggle/working/...` for generated outputs.
- Keep memory, GPU, and time limits in mind.
- Prefer smoke tests before full runs.
- Preserve submission schema exactly.

---

## Typical RAG pipeline

The expected pipeline is usually:

1. Load questions and context/legal documents.
2. Split or normalize contexts if needed.
3. Retrieve candidates with BM25 and/or dense embeddings.
4. Fuse retrieval candidates.
5. Rerank top candidates.
6. Select top-k contexts.
7. Generate or extract answer.
8. Validate output schema.
9. Write submission.

Keep these stages separable. Avoid mixing retrieval, reranking, generation, and file-writing into one hard-to-debug function.

---

## RAG rules

When working on retrieval:

- Preserve context IDs when available.
- Return both text and score when useful for debugging.
- Log top-k settings.
- Do not reduce top-k silently.
- If changing retrieval settings, explain recall vs speed trade-off.

When working on reranking:

- Log number of candidates sent to the reranker.
- Log max token length or truncation setting.
- If optimizing speed, explain possible quality impact.
- Prefer testing reranker changes on a small sample first.

When working on generation:

- Do not generate unsupported legal claims without context.
- Keep prompts grounded in retrieved context.
- Do not remove anti-hallucination instructions unless requested.
- Log whether the answer came from extractive logic, generative logic, or fallback logic.

When working on long answers:

- Do not over-shorten just to improve runtime.
- Keep enough legal basis and explanation for metrics like ROUGE-L/METEOR when relevant.
- If changing length control, explain metric impact.

---

## Kaggle rules

Before assuming any path exists, check it.

Useful checks:

```bash
find /kaggle/input -maxdepth 3 -type f | head -50
find /kaggle/input -maxdepth 3 -type d | head -50
ls -lah /kaggle/working
