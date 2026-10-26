#!/usr/bin/env python3
"""Independent-process online-v2 framing and cold-decode contract test.

The C++ online contract must have run first so its synthetic WGPN fixtures exist
under BUILD_DIR/online-contract-fixtures.  This test uses only the standard
library and does not import training, dataset, or compression Python modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def little_i64(values: list[int]) -> str:
    return struct.pack('<' + 'q' * len(values), *values).hex()


def numeric_root(values: list[int]) -> dict:
    return {
        'kind': 'numeric',
        'width': 8,
        'data_hex': little_i64(values),
        'original_numeric_payload': True,
        'logical_view': 'int64',
    }


PRIMARY_VALUES = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
SECONDARY_VALUES = [5, 8, 13, 21, 34, 55, 89, 144]


def retained_leaf_roots() -> list[dict]:
    return [numeric_root(PRIMARY_VALUES), numeric_root(SECONDARY_VALUES)]


def lower_slot_reindex_roots() -> list[dict]:
    return [{
        'kind': 'serial',
        'width': 1,
        'data_hex': bytes([4, 3, 2, 1]).hex(),
        'original_numeric_payload': False,
        'logical_view': 'bytes',
    }, numeric_root(PRIMARY_VALUES)]


def run(command: list[Path | str], *, stdin: str | None = None,
        expected_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(value) for value in command], input=stdin, text=True,
        capture_output=True, timeout=60, check=False,
    )
    if expected_success:
        require(result.returncode == 0,
                f'command failed ({result.returncode}): {result.stderr[-2000:]}')
    else:
        require(result.returncode != 0, 'invalid native request unexpectedly succeeded')
    return result


def online(server: Path, model: Path, catalog: Path, profile: Path,
           roots: list[dict], max_policy_steps: int) -> dict:
    request = {
        'request_id': model.stem,
        'roots': roots,
        'weights': [1.0, 0.0, 0.0],
        'max_policy_steps': max_policy_steps,
        'warmups': 0,
        'repeats': 1,
    }
    result = run([server, model, catalog, profile],
                 stdin=json.dumps(request, sort_keys=True, separators=(',', ':')) + '\n')
    lines = [line for line in result.stdout.splitlines() if line]
    require(len(lines) == 1, 'online server did not return exactly one JSON response')
    response = json.loads(lines[0])
    require(response.get('ok') is True,
            f'online request failed: {response.get("error", "missing error")!r}')
    require(response.get('diagnostic_only') is True and
            response.get('matched_training') is False,
            'synthetic online fixture lost its diagnostic claim boundary')
    require(response.get('candidate_evaluations') == 0 and
            response.get('replayed_prefix_stages') == 0,
            'online path performed candidate evaluation or prefix replay')
    require(response.get('action_count') == 6373,
            'online response has the wrong action dimension')
    require(response.get('registry_fingerprint') ==
            'cbd7756de4ebaf544a4eac138dc92c9eac85c701d09e8855a1b624e456170542',
            'online response has the wrong flow-STOP registry')
    return response


def cold_decode(graph_cli: Path, frame_hex: str, roots: list[dict],
                max_active_streams: int | None = None,
                *, expected_success: bool = True) -> dict | None:
    request = {'schema': 'native-graph-request-v1', 'frame_hex': frame_hex}
    if max_active_streams is not None:
        request['limits'] = {'max_active_streams': max_active_streams}
    with tempfile.TemporaryDirectory(prefix='online-v2-cold-decode-') as directory:
        path = Path(directory) / 'request.json'
        path.write_text(json.dumps(request, sort_keys=True, separators=(',', ':')))
        result = run([graph_cli, 'decode', path], expected_success=expected_success)
    if not expected_success:
        return None
    response = json.loads(result.stdout)
    require(response.get('schema') == 'native-graph-response-v1',
            'cold decoder returned the wrong schema')
    require(response.get('roots') == roots,
            'independent cold decoder changed root bytes or metadata')
    expected_raw = sum(len(bytes.fromhex(root['data_hex'])) for root in roots
                       if root['original_numeric_payload'])
    require(response.get('original_numeric_bytes') == expected_raw,
            'independent cold decoder changed B_raw')
    return response


def resolve_source(value: Path) -> Path:
    value = value.resolve()
    direct = value / 'frozen_complete_action_catalog_v1.json'
    nested = value / 'cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json'
    if direct.is_file():
        return value
    if nested.is_file():
        return nested.parent
    raise FileNotFoundError('source directory does not contain native_pipeline_v1 inputs')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()

    build = args.build_dir.resolve()
    source = resolve_source(args.source_dir)
    server = build / 'native_online_policy_server'
    graph_cli = build / 'native_graph/native_graph_request_cli'
    fixture_dir = build / 'online-contract-fixtures'
    catalog = source / 'frozen_complete_action_catalog_v1.json'
    profile = source / 'experiment_no_cmix_nncp_v1.json'
    required = [server, graph_cli, catalog, profile,
                fixture_dir / 'retained_leaf_split.wgpn',
                fixture_dir / 'lower_slot_reindex.wgpn']
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, 'missing build/source prerequisite: ' + ', '.join(missing))

    retained_roots = retained_leaf_roots()
    retained = online(server, fixture_dir / 'retained_leaf_split.wgpn',
                      catalog, profile, retained_roots, 2)
    require(retained.get('recipe_action_ids') == [6342, 194, 6342, 6341],
            'retained-leaf trace changed')
    require(retained.get('policy_calls') == 2 and
            retained.get('selected_policy_actions') == 2 and
            retained.get('closed_flow_count') == 3 and
            retained.get('forced_flow_stops') == 2 and
            retained.get('forward_stage_calls') == 0 and
            retained.get('termination') == 'budget_closed',
            'retained-leaf functional counters changed')
    require(retained.get('decode_limits', {}).get('max_active_streams', 0) >= 3,
            'online runtime did not publish capacity for three retained leaves')
    cold_decode(graph_cli, retained['frame_hex'], retained_roots,
                max_active_streams=3)
    cold_decode(graph_cli, retained['frame_hex'], retained_roots,
                max_active_streams=2, expected_success=False)
    for invalid_limit in (-1, 0, 1025):
        cold_decode(graph_cli, retained['frame_hex'], retained_roots,
                    max_active_streams=invalid_limit, expected_success=False)

    reindex_roots = lower_slot_reindex_roots()
    reindex = online(server, fixture_dir / 'lower_slot_reindex.wgpn',
                     catalog, profile, reindex_roots, 3)
    require(reindex.get('recipe_action_ids') == [6341, 101, 6341],
            'lower-slot reindex trace changed')
    require(reindex.get('policy_calls') == 3 and
            reindex.get('selected_policy_actions') == 3 and
            reindex.get('closed_flow_count') == 2 and
            reindex.get('forced_flow_stops') == 0 and
            reindex.get('forward_stage_calls') == 1 and
            reindex.get('termination') == 'all_flows_closed',
            'lower-slot reindex did not execute exactly one selected stage')
    cold_decode(graph_cli, reindex['frame_hex'], reindex_roots)

    print(json.dumps({
        'schema': 'native-online-cold-decode-v2-test',
        'functional_cases': 2,
        'independent_decode_successes': 2,
        'independent_decode_rejections': 4,
        'retained_leaf_trace': retained['recipe_action_ids'],
        'lower_slot_reindex_trace': reindex['recipe_action_ids'],
        'selected_native_stage_calls': reindex['forward_stage_calls'],
        'status': 'passed',
    }, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
