#!/usr/bin/env python3
"""Frozen user-requested experiment exclusions; catalog IDs stay unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 'native-experiment-action-scope-v1'
COMPONENT = Path(__file__).resolve().parents[1] / 'cpp/native_pipeline_v1'
DEFAULT_CATALOG = COMPONENT / 'frozen_complete_action_catalog_v1.json'
DEFAULT_PROFILE = COMPONENT / 'experiment_no_cmix_nncp_v1.json'
REGISTRY = '50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(catalog: Path = DEFAULT_CATALOG, profile: Path = DEFAULT_PROFILE) -> dict:
    catalog, profile = Path(catalog).resolve(strict=True), Path(profile).resolve(strict=True)
    raw = catalog.read_bytes()
    table = json.loads(raw)
    profile_raw = profile.read_bytes()
    scope = json.loads(profile_raw)
    catalog_sha = hashlib.sha256(raw).hexdigest()
    require(table['action_count'] == 6341 and table['registry_fingerprint'] == REGISTRY,
            'experiment catalog identity drift')
    require(scope.get('schema') == SCHEMA and scope.get('registry_fingerprint') == REGISTRY
            and scope.get('catalog_sha256') == catalog_sha, 'experiment profile identity drift')
    actions = table['registry_payload']['actions']
    require([a['action_id'] for a in actions] == list(range(6341)), 'experiment catalog ID drift')
    excluded = [a['action_id'] for a in actions if a.get('implementation_id') in ('A109', 'A110')]
    require(len(excluded) == 64 and 0 not in excluded, 'experiment exclusion source drift')
    require(scope.get('excluded_implementation_ids') == ['A109', 'A110']
            and scope.get('excluded_action_ids') == excluded, 'experiment exclusion IDs drift')
    counts = dict(catalog_action_count=6341, excluded_action_count=64, enabled_action_count=6277,
                  catalog_fixed_configuration_count=193, excluded_fixed_configuration_count=2,
                  enabled_fixed_configuration_count=191)
    require(all(scope.get(k) == v for k, v in counts.items()), 'experiment profile count drift')
    configs = [a for a in actions if a['category'] == 'mechanism' and a['target_slot'] == 0]
    require(len(configs) == 193 and sum(a['action_id'] in excluded for a in configs) == 2,
            'experiment fixed configuration drift')
    return dict(schema=SCHEMA, profile_path=str(profile),
                profile_sha256=hashlib.sha256(profile_raw).hexdigest(), catalog_sha256=catalog_sha,
                excluded_action_ids=excluded, excluded_action_count=64, enabled_action_count=6277,
                excluded_fixed_configuration_count=2, enabled_fixed_configuration_count=191)


def verify_native(catalog_response: dict, expected: dict) -> None:
    require(catalog_response.get('experiment_scope') == expected,
            'native experiment scope differs from frozen CMIX/NNCP exclusion profile')


def verify_recipe(recipe, scope: dict) -> None:
    require(not set(recipe).intersection(scope['excluded_action_ids']),
            'recipe contains a user-excluded CMIX/NNCP action')
