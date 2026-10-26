#!/usr/bin/env python3
"""Focused tests for blocked-source timestamp remediation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_weighted_grpo_blocked_sources_v1 import (  # noqa: E402
    AuditError,
    METR_ROW,
    calendar_midnight_unix_ms,
    metr_overlap_matches,
)


class CalendarTimestampEncodingTest(unittest.TestCase):
    def test_date_only_and_midnight_labels_have_exact_unix_encoding(self):
        self.assertEqual(
            calendar_midnight_unix_ms("1970-01-01", ("%Y-%m-%d",)),
            0,
        )
        self.assertEqual(
            calendar_midnight_unix_ms(
                "1990/1/1 0:00", ("%Y/%m/%d %H:%M",)
            ),
            631152000000,
        )
        self.assertEqual(
            calendar_midnight_unix_ms(
                "2002-01-01 00:00:00", ("%Y-%m-%d %H:%M:%S",)
            ),
            1009843200000,
        )

    def test_subdaily_naive_timestamp_is_not_relabelled_as_utc(self):
        with self.assertRaisesRegex(AuditError, "not a calendar-date midnight"):
            calendar_midnight_unix_ms(
                "2024-01-01 00:15:00", ("%Y-%m-%d %H:%M:%S",)
            )

    def test_unexpected_format_is_rejected(self):
        with self.assertRaisesRegex(AuditError, "does not match"):
            calendar_midnight_unix_ms("01/02/2003", ("%Y-%m-%d",))

    def test_metr_adjacent_windows_share_exactly_23_steps(self):
        previous = METR_ROW.pack(*[float(value) for value in range(48)])
        current_values = [float(value) for value in range(2, 48)] + [100.0, 101.0]
        current = METR_ROW.pack(*current_values)
        self.assertTrue(metr_overlap_matches(previous, current))
        changed = bytearray(current)
        changed[12] ^= 1
        self.assertFalse(metr_overlap_matches(previous, bytes(changed)))


if __name__ == "__main__":
    unittest.main()
