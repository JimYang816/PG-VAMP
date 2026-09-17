# Reuse without destroying independent checks

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§0.1, 0.3, 6.6, 11, 19, 23. The source takes precedence.

- Search current project code before adding shared logic. Reuse common QPSK
  labels/decisions, preprocessing, configuration/device handling and counters
  across the three algorithms (§11).
- Do not adopt older workspace implementations without the user's request.
- Independence is intentional: waveform_reference evaluates physical waveforms
  before FFT instead of calling effective H to manufacture verification input.
  Dense PG-VAMP and Cholesky VAMP remain independent checks of production paths.
- Do not deduplicate away the independently checked calculation. Shared data
  definitions do not justify sharing the operator being tested.
- Never change an oracle merely to match production. Explain changes against
  the authoritative equations and validate independently.
- Baselines do not reuse PG-VAMP's learned layer states or checkpoint.
