# WP3 compact data and replay contract

Authority: `docs/CODEX_ENGINEERING_SPEC.md` §§5.4, 6.6, 10–11, 16.2,
21.3, 22/WP3 and 23. Source formulas and acceptance requirements take precedence.
This document tracks WP3 interfaces; actual acceptance is recorded in VALIDATION.md.

## 1. Scope / trigger
Read when generating/loading physical frame records, consuming deterministic
detector samples, exporting dense input files, or auditing persisted datasets.
WP4–WP7 actual detector/training/evaluation integration remains separate.

## 2. Signatures
```text
derive_seed(master, stream, *identity) -> int
generate_records(config) -> list[dict]
generate_dataset(config, output, *, allow_large_output=False) -> Path
load_manifest(path) -> (manifest, config, records)
EffectiveDataset(manifest, split, *, device="cpu", cache_entries=None)
EffectiveDataset[index] -> {sample_id, frame_id, channel_id, split, block,
                            H, y, sigma2, x, bits}
detection_inputs(sample) -> {H, y, sigma2}
estimate_size(samples, dimension, dtype, *, labeled=True) -> dict[str, int]
materialize(manifest, split, output, *, max_output_bytes=None,
            allow_large_output=False, labeled=True) -> dict[str, int]
load_materialized(path, *, purpose="inference") -> dict
audit_dataset(manifest_path, output, *, waveform_frames=None) -> dict
```
CLI exposes simulate (generate alias), audit-data and materialize, while
inspect-config remains static. Use fresh output paths. Consult CLI help for flags.

## 3. Contracts
- Versioned manifest and tensor/basic-type `.pt` shards contain every §10.3
  physical record field. Keep IDs/split/scenario, double-precision path tensors,
  uint8 bits, independent QPSK pilots, SNR, quadrature noise seeds, arrival offset,
  waveform/generator versions, CP validity and rejection count.
- Split physical instances before expanding blocks/SNR copies. Test frames are
  paired across SNR, with all blocks in a copy at one SNR and independent noise
  subseeds per copy. Frame/channel identity is not just a split-prefixed name.
- At least split/channel/bits/pilots/noise/arrival_offset have separate stable
  SHA-256 domains. Channel retries cannot consume another stream's random draws.
- Physical identity excludes runtime/backend/storage/detector settings; full
  resolved configuration is still separately hashed and validated.
- Reconstruct in no_grad using full physical H and exact pilot cancellation.
  Noise is seeded in ideal I/Q time samples then transformed with ortho FFT;
  never estimate sigma2 from measured signal power or normalize H in the loader.
- Item tensors are H[400,400], y/x[400], sigma2 scalar and bits[400,2]. Pair
  complex128/float64 or complex64/float32. CPU is default; explicit unavailable
  CUDA errors. Stored tensors load to CPU with weights_only=True first.
- Bounded matrix cache distinguishes physical payload, block/time, mapping,
  model/config, dtype/device; consumer mutation cannot poison future samples.
- Manifest records configuration, frequency/QPSK mapping, generation distribution,
  split lineage, versions/environment, shard checksums, independent-frame and
  expanded-copy/sample counts. Unknown schema and mismatched provenance fail.
- Materialized schema contains H[K,N,N], y[K,N], sigma2[K], optional x/bits and
  metadata. Inference may omit labels; training/误码 evaluation require both.
  Ordinary detector input projection contains only H/y/sigma2.
- Treat externally loaded metadata as an independent boundary: hash values need
  the SHA-256 hexadecimal grammar, sample IDs need configured SNR-copy and block
  identity, and frame/channel IDs must belong to a disjoint split lineage table.
  Producer roundtrip success cannot replace malformed-input rejection tests.
- Schema-v1 frame/channel IDs are lowercase 64-character SHA-256 hexadecimal
  strings in compact records and dense lineage metadata. Enforce that bound:
  accepting arbitrary long IDs would invalidate fixed per-frame metadata reserves.
- Estimate exact tensor payload separately from a conservative serialization
  reserve. Enforce max_output_bytes before constructing dense matrices or output;
  only explicit allow_large_output overrides it, including config-selected export.
- For multiple exports sum per-file estimates. Each selected split's file also
  carries the complete lineage/config metadata; a small sample count does not
  imply small metadata. Static compact estimates must follow the actual schema
  (explicit pilot tensors instead of a seed-only estimate).
- audit-data is explicit and defaults to data.audit_waveform_frames; simulate
  records that policy. A pending manifest is promoted only after requested
  materialized split exports succeed.

## 4. Validation & error matrix
| Condition | Required behavior |
| --- | --- |
| Unknown schema, missing/wrong fields, checksum/config/mapping mismatch | Reject before sample exposure |
| Unsafe/duplicate shard path | Reject; paths stay within dataset root |
| Duplicate sample/copy, cross-split frame/channel reuse | Reject; never rename to hide leakage |
| CP-invalid path/window or exhausted retries | Fail with frame/path context; record actual rejections |
| Nonfinite tensor, wrong shape/dtype, sigma2 <= 0, invalid QPSK | Readable validation error |
| Missing labels for train/evaluation | Reject; inference permits absence but validates present labels |
| Requested unsupported compact lfm_detect | Explicit error; no oracle fallback |
| Output already exists or estimated export exceeds budget | Reject before overwrite/dense generation |

## 5. Good / base / bad cases
- Base: CPU complex128, 512 grid/400 data/8192 FFT/8 blocks; compact records.
- Good: read samples in different orders and through three consumers; IDs and
  payload hashes stay identical. Different SNR copies keep physical bits/paths
  and change noise subseeds, all within the same split.
- Bad: use frame_id alone as matrix cache key, share global noise RNG, compare
  H@X with itself as a physical audit, or label a partially generated output valid.

## 6. Tests required
Schema/tamper/shape/dtype/finite/label/config/index/split rejection; six-stream
isolation; cache eviction and mutation; independent-process replay; precise
payload arithmetic and pre-allocation budget guard; compact/materialized parity;
label-free input projection; CLI small physical closure and WP0–WP2 regression.
Audit saved full-frame tensors with an independent FFT, pilot cancellation,
SNR and hash reconstruction. Nonzero unequal path epsilon must satisfy source
complex128 relative error <1e-9; single precision has its own stated tolerance.
Three-consumer tests are data-contract evidence, not actual three-detector acceptance.

## 7. Wrong vs correct
Wrong: matching seed means reproducible samples even if reading consumes a global RNG.
Correct: stable sample identities derive independent stream seeds; access order and
cache hits cannot change observations.

Wrong: a tensor byte formula gives exact `.pt` file size.
Correct: report matrix and other tensor payload plus a separate metadata/serialization
reserve and the actual file size after writing.

Wrong: only validate what the generator normally writes.
Correct: mutate persisted records/metadata to test the consumer's independent
validation boundary, including lineage, block/SNR identity and hash syntax.
