# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

**Bug fix = root cause, not symptom:** a report names a symptom. Grep every caller of the function you touch and fix the shared function once - one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken. But the smallest change in the wrong place isn't lazy, it's a second bug: understand the real flow before choosing where to cut.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
1.以暗猜接口为耻，以认真查阅为荣
2. 以模糊执行为耻，以寻求确认为荣
3.以盲想业务为耻，以人类确认为荣
4.以创造接口为耻，以复用现有为荣
5.以跳过验证为耻，以主动测试为荣
6.以破坏架构为耻，以遵循规范为荣
7.以假装理解为耻，以诚实无知为菜
8.以盲目修改为耻，以谨慎重构为荣
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Physical Fidelity Guardrail (scientific code)

**Simplicity applies to code, never to physics.** This is a DDD/materials-simulation project - the core work is implementing a physical mechanism correctly, not producing the shortest solution.

- Never trade a real physical mechanism for a shorter implementation (e.g. don't fake a strength field with a geometric projection).
- Mark every intentional numerical/physical approximation with a `# PHYS-APPROX:` comment that names its ceiling AND its upgrade path. Example: `# PHYS-APPROX: pure geometric projection; ceiling = no strength field; upgrade = add per-obstacle strength`.
- This keeps a temporary approximation from silently becoming the assumed final physics.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
