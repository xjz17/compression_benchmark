"""Versioned canonical identities; historical selectors are replay-only aliases.

Native matching advances by eight bytes. Same minimum (also the wire bias)
and floor(maximum/8)*8 therefore give identical tokens and final codec wires.
NLS1's informational maximum differs; aliases retain their original metadata
on historical replay. Different minima are deliberately NOT merged.
"""
from __future__ import annotations
import re

REGISTRY_VERSION = "cpp-canonical-206-v1"
DICTIONARY = "op.dictionary.project_words64.cpp_v1"
_EXPLICIT = {
    "op.entropy_huffman.project_huffman_bytes.v1": "op.entropy_huffman.project_bytes.cpp_v1",
    "op.entropy_range.project_arithmetic_bytes.v1": "op.entropy_range.project_bytes.cpp_v1",
    "op.entropy_fse.project_rans_bytes.v1": "op.entropy_fse.zstd_rans.cpp_reference_v1",
    "op.dictionary.project_int64.v1": DICTIONARY,
    "op.dictionary.project_float64.v1": DICTIONARY,
}
_NATIVE = re.compile(r"op\.lz_match\.native_lz\.aligned8\.min(\d+)\.max(\d+)$")


def build_aliases(names):
    names = tuple(names)
    groups = {}
    for name in names:
        match = _NATIVE.fullmatch(name)
        if match:
            minimum, maximum = map(int, match.groups())
            groups.setdefault((minimum, maximum // 8 * 8), []).append(name)
    aliases = dict(_EXPLICIT)
    for members in groups.values():
        canonical = min(members, key=lambda x: int(x.rsplit(".max", 1)[1]))
        aliases.update({name: canonical for name in members if name != canonical})
    for name in names:
        target = aliases.get(name, name)
        aliases[name] = target.replace(".python_v", ".cpp_v")
    return aliases


def cpp_source(name, previous):
    if ".reference_v1" in name:
        return "cpp/src/reference_operator_actions.cpp"
    if ".native_lz." in name:
        return "cpp/src/native_lz_action_stages.cpp"
    if name in _EXPLICIT:
        if ".dictionary." in name:
            return "cpp/src/dictionary64_action_cli.cpp"
        return "cpp/src/general_codec_reference_stages.cpp"
    for fragment, source in (
        ("legacy_numeric_stages", "legacy_numeric_action_stages.cpp"),
        ("snappy_legacy_stages", "snappy_exact_stages.cpp"),
        ("reger_legacy_stages", "reger_legacy_cpp.cpp"),
        ("blosc2_legacy_stages", "blosc2_binding_cpp.cpp"),
        ("corad_lossless_stages", "corad_lossless_cpp.cpp"),
        ("tsfile_encoding.p02", "rle_int64_action_cli.cpp"),
    ):
        if fragment in previous:
            return "cpp/src/" + source
    return previous
