# WP8 delivery and acceptance matrix

Authority: [CODEX_ENGINEERING_SPEC.md](CODEX_ENGINEERING_SPEC.md) §§19–23. This is a coverage index,
not a substitute for execution receipts. Actual WP8 results are recorded in
[VALIDATION.md](../VALIDATION.md); historical evidence remains in each archived task. Main training,
powered sweeps, CUDA numerical measurements and real-world validation are unexecuted.

## Package and delivery gates
| Package | Product responsibility | Verification anchor |
| --- | --- | --- |
| WP0 | config/runtime/CLI and independent references | test_config, test_devices, test_cli, test_reference |
| WP1 | modulation/allocation/frame/LFM | test_allocation, test_qpsk, test_waveform, test_lfm |
| WP2 | affine physical model, receive FFT and pilot cancellation | test_wp2; independent waveform audits |
| WP3 | compact persistence/replay/splits/materialization | test_data_records, test_data_random, test_dataset, test_materialize, test_data_cli |
| WP4 | raw linear MMSE and exact VAMP | test_mmse, test_vamp, test_detector_inputs, test_detector_qpsk |
| WP5 | PG mathematical and gradient contracts | test_pg_vamp, test_pg_vamp_properties, test_pg_vamp_gradients |
| WP6 | train/resume/checkpoint/infer and two smokes | test_training, test_checkpoint, test_inference, test_smoke, test_wp6_review; fresh wp6_acceptance |
| WP7 | paired metrics/CI/failures/timing/persisted reports | test_evaluation, test_statistics, test_benchmark, test_reporting, test_wp7_review; fresh wp7_acceptance |
| WP8 | consolidated documentation, demo-frame and final CPU delivery | new demo tests/CLI/artifact review; full regression; documentation/link checks |

All test module names above are under tests/ with .py suffix. WP0–WP7 archive
directories are linked by the parent PRD; their old receipts are background,
not substitutes for this final execution.

## Source §23.1 communication and generator tests
| Source assertion | Concrete existing coverage |
| --- | --- |
| Resource allocation 400/64/47/1 and invertible indices | test_allocation.py::test_exact_indices_and_resource_partition |
| Frequency identity 96000/8192=6000/512 | test_config.py and test_allocation.py |
| QPSK mapping/energy | test_qpsk.py |
| Unitary FFT/Parseval and CP exact copy | test_waveform.py::test_multibatch_fft_parseval_and_cp |
| Real passband and no-channel recovery | test_waveform.py::test_complete_frame_rf_recovery_absolute_phase |
| LFM length/chirp/lag/noise | test_lfm.py (phase, full_lfm_known_lag, noise_windows) |
| Static multipath diagonal limit | test_wp2.py::test_static_limits |
| Distinct nonzero affine shifts and frequency dependence | test_wp2.py::test_nonzero_physical_equivalence and test_path_sampling |
| Independent full-size waveform/H | test_wp2.py::test_nonzero_physical_equivalence; actual smoke waveform audit |
| CP signs/last block/rejection | test_wp2.py::test_cp_boundaries_and_last_block |
| Strong pilot leakage/cancellation | test_wp2.py::test_pilot_cancellation_and_noise |
| Quadrature/FFT noise covariance | test_wp2.py::test_noise_statistics_and_identity_ber |
| Identity-AWGN Monte Carlo BER | test_wp2.py::test_noise_statistics_and_identity_ber |
| Deterministic replay | test_dataset.py::test_cross_process_replay and test_cache_order_block_identity_and_consumers |
| Physical frame/channel/SNR split isolation | test_data_records.py and test_data_random.py |

## Source §23.2 mathematical tests
| Source assertion | Concrete existing coverage |
| --- | --- |
| MMSE residual/diagonal/zero/rank-deficient | test_mmse.py and test_detector_inputs.py |
| VAMP SVD/Cholesky layer equivalence | test_vamp.py::test_layerwise_cholesky_equivalence |
| Parameter count, ordered thresholds, graph nesting/zero edges | test_pg_vamp.py::test_parameters_thresholds_graph; test_vamp.py::test_single_svd_per_call_no_cache_parameters_or_last_extrinsic |
| Energy diagonal, PSD majorizer and graph-refinement Loewner order | test_pg_vamp_properties.py::test_independent_safety_definition_energy_and_loewner |
| Residual/apply_B, B Hermitian/bounds, one-layer objective, real Jacobian, covariance variance | test_pg_vamp_properties.py::test_operator_objective_covariance_and_jacobian |
| Full-graph exact VAMP and mu irrelevance | test_pg_vamp.py::test_full_graph_exact_vamp |
| Identity/diagonal limits and hard decisions | test_pg_vamp.py::test_limits_and_scale; test_vamp.py::test_identity_matches_scalar_qpsk_enumeration |
| QPSK four-point enumeration | test_detector_qpsk.py; test_reference.py::test_qpsk_against_four_point_enumeration |
| Message rejection/precision cap/fixed mean | test_detector_qpsk.py; test_reference.py::test_message_protection_underflow_reject_and_cap |
| Zero/no-information posterior and zero gradient | test_pg_vamp.py::test_weak_channel_and_zero_backward |
| raw_gaps/raw_mu gradcheck and network backward | test_pg_vamp_gradients.py::test_raw_parameters_gradcheck and test_layerwise_oracle_and_both_parameter_gradients |
| Simultaneous H/y/sigma2 scale invariance | test_pg_vamp.py::test_limits_and_scale; test_vamp.py::test_scale_invariance |

## Source §23.3 engineering and evaluation tests
| Source assertion | Concrete existing coverage |
| --- | --- |
| CPU default regardless of CUDA availability | test_devices.py::test_cpu_never_accesses_cuda and test_cpu_stays_default_with_gpu_available |
| Explicit unavailable CUDA error | test_devices.py::test_requested_cuda_unavailable |
| dtype/device and CUDA parity | test_devices.py, test_pg_vamp_gradients.py, test_wp2.py; hardware skips explicit |
| Checkpoint predictions and resumed 2T updates | test_checkpoint.py::test_prediction_roundtrip_from_live_model; test_training.py::test_strict_resume_epoch_tail_optimizer_rng |
| Target isolation and same inputs | test_evaluation.py::test_preflight_checkpoint_population_and_target_isolation and test_real_evaluate_complete_pairing_counts_and_artifacts |
| Hand metrics/counts | test_statistics.py::test_hand_counts_energy_before_db_and_failures; test_smoke.py::test_hand_counts_and_incomplete_frame |
| Frame bootstrap/paired differences/zero errors | test_statistics.py::test_bootstrap_exact_shared_frame_indices_and_fixed_seed and test_zero_failed_and_unpaired_bootstrap |
| Failed populations remain in planned denominators | test_evaluation.py::test_failure_preserves_denominators; test_reporting.py::test_incomplete_failure_report_retains_denominator |
| Factorization/timing/CUDA synchronization/batch/threads | test_benchmark.py (factorization_boundaries, timing_metadata, explicit_cuda_sync_boundary) |
| Required CLI loop | fresh scripts/wp6_acceptance.py and scripts/wp7_acceptance.py; demo-frame separately |
| No inverse/CG/random trace/detach | test_pg_vamp.py::test_no_forbidden_apis; test_vamp.py::test_forbidden_calls_and_oracle_independence |

## Final execution evidence — 2026-09-18
Fresh full regression: 486 passed / 7 CUDA skipped; product/scripts Ruff and
format checks plus mypy pass. New full-size CPU system smoke and 11 training
CLI commands, 17 evaluation/timing/report commands, and both demo precisions
pass. Independent system/demo NumPy audits, evaluation stdlib recount/hash audit,
figure inspection and delivery-document link review support A1–A6. Exact
commands, stdout/stderr, hashes and limitations are indexed in VALIDATION.md.
Main-scale commands remain documented and unexecuted; inspect-config validates
a profile, not its full workload. These results establish software delivery,
not sufficiently powered performance validation.
