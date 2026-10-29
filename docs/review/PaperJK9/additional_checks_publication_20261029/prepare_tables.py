"""Recover the publication precision of the missing Task 4 mean difference.

Uses retained numerical records, not new estimation or prediction. Confidence
interval endpoints are not reconstructed: the table uses the recorded final
95% interval classifications uniformly for all comparisons.
"""

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
REVIEW = ROOT / "docs/review/PaperJK9"
OUT = Path(__file__).resolve().parent
worklog_path = REVIEW / "WORKLOG.md"
worklog = worklog_path.read_text()
lines = worklog.splitlines()


def evidence(phrase):
    matches = [
        {"line": number, "text": line}
        for number, line in enumerate(lines, 1)
        if phrase in line
    ]
    assert matches, phrase
    return matches[0]


ab = evidence("B−A 等权平均分别 −.175047/−.174871")
primary = evidence("D−C平均+.081218105，D−A平均−.060447295，D−B平均+.114599333")
date = evidence("日期D−C+.083629989")
date_interval = evidence("D−B日期区间跨零")
source_audit_path = REVIEW / "completion_evidence_audit_20261025.json"
source_audit = json.loads(source_audit_path.read_text())
checks = source_audit.get("checks", {})
if isinstance(checks, list):
    checks = {c.get("name", c.get("check", "")): c for c in checks}
final = checks["task4_complete"]
assert final["all_four_models_both_folds_73"]
assert final["all8_comparisons_reportable"]
assert final["all_hour_count_signatures_verified"]
assert final["no_unresolved_tail_challenges"]
assert final["replicates"] == 500000

# D-A and D-B are available directly for the primary split. The identity
# (D-A)-(D-B)=B-A checks the split/order and rounding of the retained A/B record.
primary_da = Decimal("-0.060447295")
primary_db = Decimal("0.114599333")
primary_ba = primary_da - primary_db
assert primary_ba.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) == Decimal("-0.175047")

# Date-aligned split: D-B=(D-A)-(B-A). Use a conservative full last displayed
# digit on each input, covering both rounding and truncation of the log values.
# The interval below is a reporting-precision bound, not a confidence interval.
date_da = Decimal("-0.056970332")
date_ba = Decimal("-0.174871")
date_db_from_printed_inputs = date_da - date_ba
input_reporting_radius = Decimal("0.000001") + Decimal("0.000000001")
lower = date_db_from_printed_inputs - input_reporting_radius
upper = date_db_from_printed_inputs + input_reporting_radius
quantum = Decimal("0.0001")
date_db_4dp = date_db_from_printed_inputs.quantize(quantum, rounding=ROUND_HALF_UP)
assert lower.quantize(quantum, rounding=ROUND_HALF_UP) == date_db_4dp
assert upper.quantize(quantum, rounding=ROUND_HALF_UP) == date_db_4dp

rows = [
    {"benchmark": "Homogeneous Poisson", "primary": "-0.0604", "date_aligned": "-0.0570",
     "primary_95_interval": "strictly_negative", "date_aligned_95_interval": "strictly_negative"},
    {"benchmark": "Nonhomogeneous Poisson", "primary": "+0.1146", "date_aligned": f"+{date_db_4dp}",
     "primary_95_interval": "contains_zero", "date_aligned_95_interval": "contains_zero"},
    {"benchmark": "Unrestricted filtered activity kernel", "primary": "+0.0812", "date_aligned": "+0.0836",
     "primary_95_interval": "strictly_positive", "date_aligned_95_interval": "strictly_positive"},
]
assert len(rows) == 3
record = {
    "date": "2026-10-29",
    "scope": "Publication table completion from retained historical results",
    "new_parameter_estimation": False,
    "new_prediction_calculation": False,
    "rows": rows,
    "missing_cell_recovery": {
        "identity": "(D-B)=(D-A)-(B-A), date-aligned split",
        "D_minus_A_printed": str(date_da),
        "B_minus_A_printed": str(date_ba),
        "reporting_precision_interval": [str(lower), str(upper)],
        "publication_value_4dp": str(date_db_4dp),
        "unrounded_original_value_recovered": False,
        "primary_split_identity_cross_check": str(primary_ba),
        "rounding_and_truncation_check_passed": True,
    },
    "inference_presentation": {
        "method": "Final paired contest-bootstrap 95% interval classifications",
        "replicates": 500000,
        "asterisk": "interval excludes zero",
        "missing_interval_endpoints_imputed": False,
        "raw_prediction_artifacts_restored": False,
    },
    "evidence": {"A_B": ab, "primary_final": primary, "date_final": date,
                 "date_deadline_interval": date_interval, "completion_audit": final},
    "source_sha256_at_generation": {
        "WORKLOG.md": hashlib.sha256(worklog_path.read_bytes()).hexdigest(),
        source_audit_path.name: hashlib.sha256(source_audit_path.read_bytes()).hexdigest(),
    },
    "sensitivity_case_characteristics": [
        {"contest": 8219, "delta": 2, "characteristic": "Very small cost saving"},
        {"contest": 2464, "delta": 100, "characteristic": "Large prize increase"},
        {"contest": 5144, "delta": 1, "characteristic": "Lower feedback precision"},
        {"contest": 2551, "delta": 0.1, "characteristic": "Longer deadline and lower prize"},
    ],
}
(OUT / "table_completion_audit.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"date_deadline_score": str(date_db_4dp), "publication_precision_verified": True}))
