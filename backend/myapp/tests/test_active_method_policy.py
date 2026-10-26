from __future__ import annotations

import json
import os
from unittest import TestCase
from unittest.mock import patch

from rest_framework.test import APIRequestFactory

from myapp.active_method_policy import (
    active_method_policy_binding,
    active_method_policy_summary,
    is_active_algorithm_key,
    is_active_method_name,
)
from myapp.algorithm_allow_list import should_run
from myapp.home_leaderboard_boxplot_data import (
    is_home_leaderboard_algorithm_enabled,
    is_home_leaderboard_label_enabled,
)
from myapp.views import MethodCatalogView


class ActiveMethodPolicyTests(TestCase):
    def test_binding_records_frozen_policy_sha(self) -> None:
        binding = active_method_policy_binding()
        self.assertEqual(
            binding["policy_sha256"],
            "f48677a529247a165898bce319f3e20ef6ba9c305552676344f3ead39d0bfc5d",
        )

    def test_glza_is_retired_from_active_scopes(self) -> None:
        self.assertFalse(is_active_method_name("GLZA"))
        self.assertFalse(is_active_method_name("glza"))
        self.assertFalse(is_active_algorithm_key("glza"))
        self.assertTrue(is_active_method_name("LZ4"))
        self.assertTrue(is_active_algorithm_key("lz4-framed"))

    def test_explicit_allowlist_cannot_reenable_retired_algorithm(self) -> None:
        with patch.dict(os.environ, {"COMPRESSION_BENCHMARK_RUN_ALGORITHMS": "glza,lz4-framed"}):
            self.assertFalse(should_run(None, "glza"))
            self.assertTrue(should_run(None, "lz4-framed"))

    def test_home_leaderboard_uses_active_policy(self) -> None:
        self.assertFalse(is_home_leaderboard_algorithm_enabled("glza"))
        self.assertFalse(is_home_leaderboard_label_enabled("GLZA"))
        self.assertTrue(is_home_leaderboard_algorithm_enabled("lz4-framed"))

    def test_policy_count_contract(self) -> None:
        summary = active_method_policy_summary()
        self.assertEqual(summary["catalog_method_count"], 86)
        self.assertEqual(summary["active_method_count"], 85)
        self.assertEqual([row["canonical_name"] for row in summary["retired_methods"]], ["GLZA"])

    def test_methods_api_returns_active_catalog_and_policy(self) -> None:
        response = MethodCatalogView.as_view()(APIRequestFactory().get("/api/methods/"))
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content)
        self.assertNotIn("GLZA", {row["name"] for row in payload["items"]})
        self.assertEqual(payload["active_method_policy"]["active_method_count"], 85)
