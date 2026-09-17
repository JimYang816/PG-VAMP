# Mandatory source-reading gate

This is a reading protocol, not an engineering specification or a substitute
for docs/CODEX_ENGINEERING_SPEC.md.

Trellis validation reports that the original source is 83,463 bytes, exceeding
the current 32,768-byte per-file injection limit. An automatically injected
prefix is incomplete even when context validation exits successfully.

Before any implementation or check:
1. Read the ORIGINAL docs/CODEX_ENGINEERING_SPEC.md from the repository in
   bounded consecutive chunks until every line has been read. Current file:
   2,102 lines; suggested inclusive ranges: 1–350, 351–700, 701–1050,
   1051–1400, 1401–1750, 1751–2102. If any tool truncates output, reduce the
   range and read the missing portion; never treat truncated output as complete.
2. Verify SHA-256:
   A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B.
   If changed, reread the new complete source and reconcile planning against it
   before continuing. Do not overwrite or restore the user's specification.
3. Then read the task PRD, design and implement plan. Scope is §22 WP0;
   §§19–21 and 23 supply engineering checks; §§13–15 govern the two references.
4. Record that the complete source was read, with hash, in work/check evidence.
   Do not rely on this protocol, bootstrap summaries, injected prefixes or
   copied excerpts as the mathematical contract.

This applies equally to implementation and verification agents. The full source
entry remains in both manifests for traceability; its truncation warning is
known and MUST be handled by direct complete source reads.
