import json
from pathlib import Path


def test_operator_catalog_contains_the_29_operator_vocabulary():
    reference_dir = Path(__file__).resolve().parents[3] / "benchmark" / "reference" / "methods"
    rows = []
    for filename in ("encoding_operators.json", "byte_codec_operators.json"):
        payload = json.loads((reference_dir / filename).read_text(encoding="utf-8"))
        rows.extend(payload["operators"])

    ids = [row["id"] for row in rows]
    assert len(ids) == 29
    assert len(set(ids)) == 29
    assert "grouping" in ids
