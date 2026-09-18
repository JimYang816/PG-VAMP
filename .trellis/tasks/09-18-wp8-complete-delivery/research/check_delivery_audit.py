"""WP8 documentation links, frozen source, reused evidence and visual record."""

import hashlib
import json
import re
from pathlib import Path

from pgvamp_ofdm.config import load_config

ROOT = Path(__file__).resolve().parents[4]
RESEARCH = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


baseline = json.loads((RESEARCH / "authority-reference-baseline.json").read_text())
for item in baseline:
    assert sha(ROOT / item["path"]) == item["sha256"]
documents = [
    "README.md",
    "VALIDATION.md",
    "IMPLEMENTATION_STATUS.md",
    "docs/CONFIGURATION.md",
    "docs/ALGORITHMS.md",
    "docs/REPRODUCING.md",
    "docs/DELIVERY_MATRIX.md",
    ".trellis/spec/backend/wp8-delivery-contract.md",
    ".trellis/spec/backend/index.md",
]
links = []
for name in documents:
    source = ROOT / name
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
        if target.startswith(("https:", "http:", "#")):
            continue
        path = source.parent / target.split("#")[0]
        assert path.exists(), (name, target)
        links.append({"source": name, "target": target})
matrix = (ROOT / "docs/DELIVERY_MATRIX.md").read_text(encoding="utf-8")
anchors = []
for module, function in re.findall(r"(test_\w+\.py)::(test_\w+)", matrix):
    assert f"def {function}(" in (ROOT / "tests" / module).read_text(encoding="utf-8"), (
        module,
        function,
    )
    anchors.append(f"{module}::{function}")
overlay = ROOT / "runs/wp8-check-commands/resume-doc.yaml"
overlay.write_text("training:\n  max_steps: 600\n", encoding="utf-8")
original, resumed = load_config(ROOT / "configs/cpu_dev.yaml"), load_config(overlay)
assert resumed.values["training"].pop("max_steps") == 600
assert original.values["training"].pop("max_steps") == 300
assert original.values == resumed.values
reused = []
for name, expected in [
    ("training-cli-receipts.json", 11),
    ("evaluation-receipts.json", 15),
    ("evaluation-report-receipts.json", 2),
]:
    receipts = json.loads((RESEARCH / name).read_text())
    assert len(receipts) == expected
    assert all(item["exit_code"] == 0 for item in receipts)
    reused.append({"file": name, "sha256": sha(RESEARCH / name), "successful_commands": expected})
system = json.loads((RESEARCH / "system-artifact-audit.json").read_text())
for name, digest in system["artifacts_sha256"].items():
    assert sha(ROOT / "runs/wp8-training-acceptance" / name) == digest
figures = []
for dtype in ("complex128", "complex64"):
    for figure in sorted((ROOT / f"runs/wp8-check-demo-{dtype}/figures").glob("*.png")):
        figures.append({"path": str(figure.relative_to(ROOT)), "sha256": sha(figure)})
result = {
    "status": "passed",
    "authority_reference_hashes": baseline,
    "document_hashes": {name: sha(ROOT / name) for name in documents},
    "resolved_local_links": links,
    "exact_test_anchors": anchors,
    "resume_overlay": "600 cumulative updates, otherwise identical cpu_dev defaults",
    "reused_cli_evidence": reused,
    "system_artifacts_rehashed": len(system["artifacts_sha256"]),
    "reused_audit_hashes": {
        name: sha(RESEARCH / name)
        for name in (
            "system-artifact-audit.json",
            "evaluation-artifact-audit.json",
            "report-visual-review.json",
            "evaluation-report-receipts.json",
        )
    },
    "demo_visual_review": {
        "status": "passed",
        "method": "Reviewer directly viewed all four native PNGs",
        "checks": [
            "No blank or clipped panels",
            "Complete TX/RX and selected LFM peak displayed",
            "Noiseless real RF and oracle noisy IQ kept distinct",
            "400x400 channel and unequallized receive constellation correctly labeled",
            "Reference QPSK labeled audit only; no performance claims",
        ],
        "figures": figures,
    },
}
(RESEARCH / "check-delivery-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "status": "passed",
            "links": len(links),
            "test_anchors": len(anchors),
            "authority_reference_files": len(baseline),
            "system_artifacts_rehashed": len(system["artifacts_sha256"]),
        }
    )
)
