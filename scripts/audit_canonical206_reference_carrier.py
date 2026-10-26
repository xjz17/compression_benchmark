#!/usr/bin/env python3
"""Audit reference action carriers without changing existing experiments.

Reads three saved frames and the original CSV columns. Only the two explicitly
listed NMHC encodings (identity[1], ZigZag[11]) and their inverse calls execute
the existing reference CLI. Outputs are created in a new, non-overwritten dir.
No search or timing benchmark is performed. --check-only never executes codecs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from zoneinfo import ZoneInfo
import zlib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'analysis_outputs/grpo_recipe_search_20260906_v3'
DEFAULT_OUTPUT = ROOT / 'analysis_outputs/reference_carrier_audit_20260906_v1'
HEADER = struct.Struct('<4sBBQI')
MASK64 = (1 << 64) - 1
CASES = [('Air-Quality', 'NMHC(GT)', 11), ('Air-Quality', 'timestamp', 7),
         ('Metro-Traffic', 'clouds_all', 1)]
SOURCES = {
    'cpp/src/reference_operator_actions.cpp': [
        ('I zigzag(I value)', 4), ('Bytes compact(const Values& values)', 14),
        ('case 1: out = values;', 1), ('case 7: case 8: {', 7),
        ('case 11: for (I value : values)', 1), ('Bytes state_bytes(', 6),
        ('Bytes raw_encode(', 5), ('Bytes decode(', 4)],
    'cpp/src/reference_operator_actions_cli.cpp': [('if (command == "encode")', 8)],
    'scripts/primitive_operator_action_runtime.py': [('def encode_operator_pipeline(', 14)],
    'scripts/unified_action_runtime.py': [
        ('def _pack_result(', 37), ('if all(name in primitive.ACTION_BY_NAME for name in action_names):', 4)],
    'scripts/canonical206_search_env.py': [
        ("HEADER = struct.Struct", 1), ('ref_pair = len(child)<=2', 10),
        ('def encode_cold(', 17)],
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def signed64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def source_evidence():
    hashes, evidence = {}, []
    for relative, anchors in SOURCES.items():
        path = ROOT / relative
        raw = path.read_bytes()
        hashes[str(path)] = digest(raw)
        lines = raw.decode('utf-8').splitlines()
        for anchor, count in anchors:
            start = next(i for i, line in enumerate(lines) if anchor in line)
            evidence.append({'path': str(path), 'line_start': start + 1,
                             'line_end': start + count,
                             'excerpt': '\n'.join(lines[start:start + count])})
    return hashes, evidence


def original_column(column):
    """Read actual CSV ground truth; do not derive it from codec output."""
    path = Path(column['source_path'])
    source_hash = digest(path.read_bytes())
    if source_hash != column['source_sha256']:
        raise AssertionError(f'CSV source changed: {path}')
    if column['kind'] != 'int64' or column['nan_removed'] != 0:
        raise AssertionError('this diagnostic requires unchanged complete int64 columns')
    values = []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header[column['source_index']] != column['name']:
            raise AssertionError('CSV source column name differs')
        for row in reader:
            number = Decimal(row[column['source_index']])
            value = int(number)
            if number != value:
                raise AssertionError('non-integral value in frozen int64 column')
            values.append(value)
    raw = b''.join(struct.pack('<q', value) for value in values)
    if len(values) != column['count'] or digest(raw) != column['input_sha256']:
        raise AssertionError('CSV numerical column differs from frozen input')
    return raw, values, source_hash


def parse_frame(frame: bytes, action_id: int, original: bytes):
    magic, kind, layers, raw_count, checksum = HEADER.unpack_from(frame)
    if (magic, kind, layers) != (b'C206', 1, 1):
        raise AssertionError('expected one-layer int64 C206 frame')
    outer = frame[HEADER.size:]
    if outer[:4] != b'UAR1':
        raise AssertionError('missing UAR1 carrier')
    header_size = struct.unpack_from('<I', outer, 4)[0]
    metadata_bytes = outer[8:8 + header_size]
    metadata = json.loads(metadata_bytes)
    compressed = outer[8 + header_size:]
    opa = zlib.decompress(compressed)
    if opa[:6] != b'OPA1\x01\x01' or opa[6] != action_id:
        raise AssertionError('expected exact one-action OPA1 stream')
    pos = 7
    tag = opa[pos:pos + 12].rstrip(b'\0').decode()
    pos += 12
    array_count, extra_size = struct.unpack_from('<HI', opa, pos)
    pos += 6 + extra_size
    arrays = []
    for _ in range(array_count):
        count, size = struct.unpack_from('<II', opa, pos)
        pos += 8
        if size != count * 8:
            raise AssertionError('OPA1 metadata array size mismatch')
        arrays.append(list(struct.unpack_from('<' + 'q' * count, opa, pos)))
        pos += size
    meta_end = pos
    compact_tag = opa[pos:pos + 2].decode()
    count = struct.unpack_from('<I', opa, pos + 2)[0]
    pos += 6
    width = {'U1': 1, 'I1': 1, 'I2': 2, 'I4': 4, 'I8': 8}[compact_tag]
    if len(opa) - pos != count * width:
        raise AssertionError('compact payload size differs from count * width')
    transformed = [int.from_bytes(opa[i:i + width], 'little', signed=compact_tag != 'U1')
                   for i in range(pos, len(opa), width)]
    if action_id == 11:
        values = [((value & MASK64) >> 1) ^ -((value & MASK64) & 1) for value in transformed]
    elif action_id == 1:
        values = transformed
    elif action_id == 7:
        values = [arrays[0][0]]
        for value in transformed:
            values.append(signed64(values[-1] + value))
    else:
        raise AssertionError('diagnostic only supports actions 1, 7, 11')
    restored = b''.join(struct.pack('<q', value) for value in values)
    if (restored != original or raw_count * 8 != len(original)
            or checksum != zlib.crc32(original)):
        raise AssertionError('independent carrier decode differs from original CSV bytes')
    layout = {
        'C206_header_bytes': HEADER.size, 'C206_layers': layers,
        'UAR1_magic_and_length_bytes': 8, 'UAR1_json_bytes': header_size,
        'UAR1_metadata': metadata, 'zlib6_payload_bytes': len(compressed),
        'zlib_payload_sha256': digest(compressed), 'OPA1_uncompressed_bytes': len(opa),
        'OPA1_uncompressed_sha256': digest(opa), 'OPA1_header_bytes': 7,
        'transform_metadata_bytes': meta_end - 7, 'transform_tag': tag,
        'transform_metadata_arrays': arrays, 'compact_tag': compact_tag,
        'compact_header_bytes': 6, 'compact_width_bytes': width,
        'compact_value_count': count, 'compact_values_bytes': len(opa) - pos,
        'transformed_fixed64_value_bytes': len(transformed) * 8,
        'complete_frame_bytes': len(frame), 'complete_frame_sha256': digest(frame),
        'original_bytes': len(original), 'original_sha256': digest(original),
        'compression_ratio': len(original) / len(frame),
        'independent_decode_equals_source_csv': True,
    }
    if sum(layout[k] for k in ('C206_header_bytes', 'UAR1_magic_and_length_bytes',
                              'UAR1_json_bytes', 'zlib6_payload_bytes')) != len(frame):
        raise AssertionError('outer layer byte accounting mismatch')
    if 7 + layout['transform_metadata_bytes'] + 6 + layout['compact_values_bytes'] != len(opa):
        raise AssertionError('OPA1 byte accounting mismatch')
    return layout, metadata, compressed, opa


def report(result):
    cases = result['cases']
    ablation = result['nmhc_identity_vs_zigzag']
    lines = ['# Reference 算子实际包含 compact + zlib6：可复现审计', '',
             f"北京时间：{result['created_at_beijing']}。原实验和结果未修改。", '',
             '## 结论', '',
             '44.905× 是 ZigZag → 自适应整字节宽度 compact → OPA1 元数据 → zlib6，再加 UAR1/C206 的完整文件结果；不是裸 ZigZag 的压缩比。',
             '编码和回放正确；原表只写“ZigZag”隐藏了实际执行的载荷压缩，名称/归因需要纠正。', '',
             '| 已存结果 | 原始 B | 变换后定宽64位数值 B | compact数值 B | 未压缩OPA1 B | zlib6载荷 B | 完整文件 B | CR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = {11: 'NMHC：ZigZag + compact + zlib6',
              7: 'timestamp：Delta + compact + zlib6',
              1: 'clouds_all：Identity（scaling-ref）+ compact + zlib6'}
    for case in cases:
        a = case['layout']
        lines.append(f"| {labels[case['action_id']]} | {a['original_bytes']} | {a['transformed_fixed64_value_bytes']} | {a['compact_values_bytes']} | {a['OPA1_uncompressed_bytes']} | {a['zlib6_payload_bytes']} | {a['complete_frame_bytes']} | {a['compression_ratio']:.6f} |")
    lines += ['', 'Delta 的定宽数值少一个首值，但首值完整保存在变换元数据中，不是删除原始值。', '',
              '## NMHC 的具体层级', '',
              '74,856 B 原始 int64 → ZigZag 后仍 74,856 B → compact I2 数值 18,714 B → OPA1 18,745 B → zlib6 1,511 B。',
              '最终：C206 18 + UAR1 magic/length 8 + JSON 130 + zlib6 1,511 = 1,667 B。', '',
              '## 同源轻量消融', '',
              '| 算法的真实名称 | 完整文件 B | CR | 原始CSV逐字节回放 |',
              '|---|---:|---:|---|']
    for a in ablation['rows']:
        lines.append(f"| {a['display_name']} | {a['complete_frame_bytes']} | {a['compression_ratio']:.6f} | PASS |")
    lines += ['', f"两条完整文件仅相差 **{ablation['identity_minus_zigzag_bytes']} B**；compact 载荷都为 I2，均没有仅靠 ZigZag 减少每值字节。该7B包含变换后符号分布、OPA1标签以及UAR1动作名长度的共同差异，不应全当成裸变换贡献。", '',
              f"NMHC 中 -200 占 {cases[0]['minus200_count']}/{cases[0]['source_count']}（{cases[0]['minus200_fraction']:.4%}）；保留这些原始值，未为提高压缩比删除数据。UCI 明确把 -200 定义为缺失标记：[官方数据说明](https://archive.ics.uci.edu/dataset/360/air%2Bquality)。", '',
              '## 纯 ZigZag 小例子', '',
              '`[0,-1,1,-2,2] → [0,1,2,3,4]`。若输入输出都按64位存储，均为40 B；它改变数字表示，不自行减少定宽字节。', '',
              '## 深度10必须说明计数对象', '',
              '- 延续原206库时，建议名称为“最多10个注册动作ID”；硬检查展开后的ID数，不把10个完整recipe误当成10个ID。',
              '- 对单个reference动作，“1个注册ID”实际还含compact与zlib6；不能称其只有1个纯物理变换。两个连续reference动作由原运行器合并后只做一次compact/zlib6。',
              '- 若改成严格的10个物理阶段，须把隐藏compact/zlib作为显式阶段并另立版本与基线；不能悄悄改现有ID语义或覆盖旧结果。',
              '- 保留原始ID及实现名，展示名改为“ZigZag-ref + compact + zlib6”等；scaling[1]为identity，不是float定标。', '',
              '## 复验', '',
              f"`python3 -B {ROOT / 'scripts/audit_canonical206_reference_carrier.py'} --output /另一个不存在的新目录`", '',
              f"只核对本次文件而不运行codec：`python3 -B {ROOT / 'scripts/audit_canonical206_reference_carrier.py'} --check-only --output {result['output']}`", '',
              f"本次只调用既有CLI {len(result['codec_calls'])} 次（2次encode、2次decode），不是搜索或性能基准。全部source、输入和frame哈希、分层字节、调用参数保存于 AUDIT.json。", '',
              '## 精确源码证据', '']
    for e in result['source_evidence']:
        lines += [f"- `{e['path']}:{e['line_start']}-{e['line_end']}`", '```', e['excerpt'], '```']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run, output = args.source_run.resolve(), args.output.resolve()
    if args.check_only:
        audit = read_json(output / 'AUDIT.json')
        for name, expected in {**audit['input_artifact_hashes'], **audit['generated_artifact_hashes'],
                               **audit['source_hashes']}.items():
            if digest(Path(name).read_bytes()) != expected:
                raise AssertionError(f'hash changed: {name}')
        print(json.dumps({'status': 'PASS', 'mode': 'read_only_hash_check', 'codec_calls': 0,
                          'output': str(output)}, ensure_ascii=False))
        return
    if output.exists():
        raise FileExistsError(f'refuse to overwrite existing audit: {output}')
    manifest = read_json(run / 'manifest.json')
    columns = {(c['dataset'], c['name']): c for c in manifest['columns']}
    source_hashes, evidence = source_evidence()
    source_hashes[str(Path(__file__).resolve())] = digest(Path(__file__).read_bytes())
    cli = ROOT / 'cpp/build/reference_operator_actions_cli'
    source_hashes[str(cli)] = digest(cli.read_bytes())
    input_hashes = {str(run / 'manifest.json'): digest((run / 'manifest.json').read_bytes())}
    selected, payloads = [], {}
    for dataset, name, action_id in CASES:
        pool = []
        for path in (run / 'columns').glob('grpo_recipe__*.json'):
            row = read_json(path)
            if row['dataset'] == dataset and row['column'] == name and row['selected_ids'] == [action_id]:
                pool.append((row, path))
        row, path = min(pool, key=lambda pair: (pair[0]['compressed_bytes'], pair[0]['seed']))
        column = columns[dataset, name]
        raw, values, csv_hash = original_column(column)
        frame_path = (run / row['frame_path']).resolve()
        frame = frame_path.read_bytes()
        if len(frame) != row['compressed_bytes'] or digest(frame) != row['frame_sha256']:
            raise AssertionError('saved frame differs from result row')
        layout, metadata, compressed, opa = parse_frame(frame, action_id, raw)
        input_hashes.update({str(path): digest(path.read_bytes()), str(frame_path): digest(frame),
                             str(Path(column['source_path'])): csv_hash})
        selected.append({'dataset': dataset, 'column': name, 'action_id': action_id,
                         'source_row': str(path), 'source_frame': str(frame_path),
                         'seed': row['seed'], 'source_count': len(values),
                         'source_csv': column['source_path'], 'source_csv_sha256': csv_hash,
                         'minus200_count': values.count(-200),
                         'minus200_fraction': values.count(-200) / len(values),
                         'most_common_values': Counter(values).most_common(5), 'layout': layout})
        payloads[action_id] = (raw, frame, metadata, compressed, opa)
    toy = [0, -1, 1, -2, 2]
    mapped = [((x & MASK64) << 1 ^ (MASK64 if x < 0 else 0)) & MASK64 for x in toy]
    if mapped != [0, 1, 2, 3, 4]:
        raise AssertionError('pure ZigZag toy mismatch')
    ablations, codec_calls, generated = [], [], {}
    raw, saved_frame, metadata, _, _ = payloads[11]
    for action_id, name, label in [(1, 'op.scaling.reference_v1', 'Identity[1] + compact + zlib6'),
                                    (11, 'op.zigzag.reference_v1', 'ZigZag[11] + compact + zlib6')]:
        command = [str(cli), 'encode', name]
        encoded = subprocess.run(command, input=raw, capture_output=True, check=True, timeout=10).stdout
        codec_calls.append({'argv': command, 'input_sha256': digest(raw), 'output_sha256': digest(encoded)})
        command = [str(cli), 'decode']
        recovered = subprocess.run(command, input=encoded, capture_output=True, check=True, timeout=10).stdout
        codec_calls.append({'argv': command, 'input_sha256': digest(encoded), 'output_sha256': digest(recovered)})
        if recovered != raw:
            raise AssertionError('light ablation failed independent inverse check')
        header = json.dumps(dict(metadata, actions=[name]), sort_keys=True, separators=(',', ':')).encode()
        complete = saved_frame[:HEADER.size] + b'UAR1' + struct.pack('<I', len(header)) + header + encoded
        layout, _, _, _ = parse_frame(complete, action_id, raw)
        if action_id == 11 and complete != saved_frame:
            raise AssertionError('replayed ZigZag frame differs from original saved frame')
        relative = f'frames/NMHC_action{action_id}_compact_zlib6.c206'
        generated[relative] = complete
        ablations.append({'action_id': action_id, 'implementation_name': name, 'display_name': label,
                          **layout, 'frame_path': relative})
    gap = ablations[0]['complete_frame_bytes'] - ablations[1]['complete_frame_bytes']
    if gap != 7:
        raise AssertionError(f'frozen NMHC identity/ZigZag result unexpectedly differs: {gap}')
    result = {'status': 'PASS', 'scope': 'three stored full int64 columns; two explicit lightweight encodings only',
              'semantic_verdict': 'carrier naming/attribution correction required; no bitwise mismatch observed',
              'created_at_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'output': str(output), 'source_run': str(run), 'cases': selected,
              'nmhc_identity_vs_zigzag': {'identity_minus_zigzag_bytes': gap, 'rows': ablations},
              'pure_zigzag_toy': {'input': toy, 'output': mapped, 'input_fixed64_bytes': 40,
                                  'output_fixed64_bytes': 40, 'codec_executed': False},
              'codec_calls': codec_calls, 'source_hashes': source_hashes,
              'source_evidence': evidence, 'input_artifact_hashes': input_hashes,
              'missing_value_semantics_source': 'https://archive.ics.uci.edu/dataset/360/air%2Bquality'}
    output.mkdir(parents=True)
    for relative, content in generated.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    report_path = output / 'REPORT_compact_zlib6.md'
    report_path.write_text(report(result), encoding='utf-8')
    generated_hashes = {str(output / relative): digest(content) for relative, content in generated.items()}
    generated_hashes[str(report_path)] = digest(report_path.read_bytes())
    result['generated_artifact_hashes'] = generated_hashes
    for name, expected in {**source_hashes, **input_hashes}.items():
        if digest(Path(name).read_bytes()) != expected:
            raise AssertionError(f'source/input changed during audit: {name}')
    (output / 'AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'created_at_beijing': result['created_at_beijing'],
                      'cases': len(selected), 'codec_calls': len(codec_calls),
                      'identity_minus_zigzag_bytes': gap, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
