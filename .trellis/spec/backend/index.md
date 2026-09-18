# Python/PyTorch engineering guidelines

Authority: [CODEX_ENGINEERING_SPEC.md](../../../docs/CODEX_ENGINEERING_SPEC.md),
especially §§0–1. It is the sole, self-contained engineering specification.
These guidelines organize recurring obligations; they never replace, weaken
or reinterpret its mathematics, communication parameters, tests or WP order.
In any conflict, the execution specification wins. Read the relevant source
sections before implementation; resolve ambiguities explicitly rather than
inventing a contract in Trellis.

Current state: WP0 configuration/runtime/CLI and independent references, WP1
modulation/transmit-frame/LFM correlation, and WP2 physical channel/receiver
are implemented; see root VALIDATION.md for actual checks and review status.
WP3 data/replay is implemented and independently verified. WP4 production
MMSE/VAMP interfaces are implemented; see VALIDATION.md for current independent
review evidence. WP5 and later paths remain prescribed future work. The backend
directory is Trellis's discovery location for the Python library/CLI; it does
not imply a web server, database or frontend.

| Guide | Long-term responsibility | Source sections |
| --- | --- | --- |
| [Structure](directory-structure.md) | src layout, modules, interfaces, dependencies | 0.3, 11, 19, 21 |
| [Runtime and configuration](runtime-config.md) | CPU/dtype defaults, strict config, validation | 3, 10.5, 15.2, 20–21 |
| [Reproducibility](reproducibility.md) | random streams, manifests, checkpoint, logs | 10, 15, 18, 21 |
| [Numerical contracts](numerical-contracts.md) | physical model, algorithm invariants, prohibitions | 2–14, 23 |
| [Verification](quality-guidelines.md) | independent oracles, pytest, truthful results | 0.3–0.4, 16–18, 22–24 |
| [WP0 inspection interface](wp0-inspect-contract.md) | strict config, explicit runtime, static CLI and provenance | 15.2, 19–22 |
| [WP1 waveform interface](wp1-waveform-contract.md) | modulation, transmit layout, real waveform, LFM and template-start correlation | 3–5, 7, 19–23 |
| [WP2 physical channel interface](wp2-channel-contract.md) | affine paths, actual-window CP/H, independent continuous reference, FFT, pilot cancellation and AWGN | 5–9, 19–23 |
| [WP3 data interface](wp3-data-contract.md) | compact schema, split/stream identity, deterministic reconstruction, dense export and persisted-data audit | 10–11, 16.2, 21–23 |
| [WP4 detector interface](wp4-detector-contract.md) | raw linear MMSE, exact SVD VAMP, shared production QPSK/messages, independent oracle and label isolation | 11–13, 14.6, 20, 22–23 |

Always also read [workflow and authority](../guides/workflow-and-authority.md).
The complete test list and acceptance gates remain in source §§22–23;
this index is not a reduced acceptance checklist.
