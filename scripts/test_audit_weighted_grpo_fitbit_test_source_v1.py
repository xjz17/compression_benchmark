#!/usr/bin/env python3
import unittest

from audit_weighted_grpo_fitbit_test_source_v1 import (
    SOURCE_GROUP,
    source_group_overlaps_frozen_cohort,
    validate_readme_evidence,
)


class FitbitEvidenceTests(unittest.TestCase):
    def test_explicit_utc_card_claim_passes(self):
        text = (
            "Time | TIME | INT64 (ms) | timestamp, parsed as UTC\n"
            "One train file with 291 observations.\n"
            "2022-12-23T09:20:30Z through 2023-11-18T07:59:00Z.\n"
            "All five measurements and all 291 rows are retained and sorted by Time."
        )
        result = validate_readme_evidence(text)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["timestamp_policy"],
            "explicit_utc_epoch_milliseconds_from_source_card",
        )

    def test_missing_timezone_claim_does_not_pass(self):
        text = (
            "Time | TIME | INT64 (ms)\n"
            "One train file with 291 observations.\n"
            "All five measurements and all 291 rows are retained and sorted by Time."
        )
        result = validate_readme_evidence(text)
        self.assertFalse(result["passed"])
        self.assertFalse(result["claims"]["utc_timestamp"])

    def test_candidate_placeholder_is_not_frozen_overlap(self):
        manifest = {
            "datasets": [
                {"role": "test_candidate", "source_group": SOURCE_GROUP},
            ]
        }
        self.assertFalse(source_group_overlaps_frozen_cohort(manifest))

    def test_train_validation_group_is_frozen_overlap(self):
        manifest = {
            "datasets": [
                {"role": "train_validation", "source_group": SOURCE_GROUP},
            ]
        }
        self.assertTrue(source_group_overlaps_frozen_cohort(manifest))


if __name__ == "__main__":
    unittest.main()
