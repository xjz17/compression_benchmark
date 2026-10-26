"""Versioned accounting for mechanisms, profiles, recipes and compatibility I/O.

Historical A IDs remain addressable. A retained mechanism is not a claim that
all its internals are atomic or that a complete benchmark path is available.
"""
from collections import Counter

VERSION = "operator-implementation-layers-v1"
LAYER_NAMES = {
    "mechanism": "保留机制",
    "profile": "参数与配置",
    "recipe": "组合配方与完整后端",
    "adapter": "封包与兼容适配",
}

PROFILES = {
    "A001": ("A104", "widths=[4,60]；高60位按原合同符号扩展"),
    "A041": ("A042", "dictionary_order=value；保持原字及有符号排序规则"),
    "A051": ("A104", "widths=[32,32]；保留分块和raw旁路"),
    "A053": ("A042", "dictionary_order=frequency；并列按原有signed value顺序"),
    "A057": ("A060", "minimum/maximum配置；原NLS1 role校验与封包保留"),
    "A059": ("A058", "search_policy=greedy；A058为lazy，共用同一LZ搜索函数"),
    "A065": ("A060", "window配置；原NLS1 role校验与封包保留"),
}

RECIPES = {
    "A022": {"mechanism_ids": ["A020", "A055", "A031", "A038"],
             "kind": "branched_block_recipe",
             "description": "9值块：anchor旁路，8残差交A020；整数尾部RLE，浮点尾部Gorilla。保留当前/历史RLE格式选择。"},
    "A043": {"mechanism_ids": ["A042", "A020"], "kind": "branched_dictionary_recipe",
             "description": "首见字典分离为dictionary和ids；只将ids交A020，dictionary旁路，保留ADI1。"},
    "A047": {"mechanism_ids": ["A049", "A028"], "kind": "ordered_stateful_recipe",
             "description": "字节分列后按列顺序Concat，再做uint8模差分；previous初值0且跨列连续。"},
    "A066": {"mechanism_ids": ["A104", "A020"], "kind": "branched_bitfield_recipe",
             "description": "widths=[4,60]分流；低4位交A020(width=4,MSB)，高位符号扩展后旁路；保存N及奇数尾填充。"},
}
for _id in ("A084", "A085", "A087", "A088", "A089", "A090", "A091", "A092", "A093", "A094"):
    RECIPES[_id] = {
        "mechanism_ids": [], "kind": "opaque_composite_backend",
        "description": "保留完整/剩余流程作为兼容后端；未证明可完全展开为当前独立机制，不能因移出机制计数而宣称86方法已全部原子化。",
    }

ADAPTERS = {
    "A027": "BUFF已有planes/base/scale/width序列化；不重复执行位打包",
    "A056": "历史LZ标签下的阈值RLE代理；不冒充LZ，也不等同A055最优RLE",
    "A062": "历史parse标签下的参考代理；保留旧接口与逆过程",
    "A063": "A086封包schema：magic=ZPP1，payload_validation=LZS1；没有parse搜索",
    "A064": "位置常量XOR参考代理；不冒充窗口匹配",
    "A086": "通用magic/length封包schema",
    "A095": "历史参考头尾字段代理；非真实gzip容器",
    "A096": "LZ4 frame schema，含descriptor checksum和块边界",
    "A097": "DEFLATE容器header/core/trailer组装；保留gzip/zlib/XZ各自格式合同",
    "A098": "兼容数值方法封包schema",
    "A099": "兼容数值方法封包schema",
    "A100": "REGER兼容封包schema",
    "A101": "BUFF固定头与payload序列化",
    "A102": "Blosc2 frame schema及logical/physical offset映射",
}

CANDIDATES = [
    {"ids": ["A006", "A007", "A008", "A009", "A010"],
     "decision": "保留5项尺度选择机制；统一选择策略接口仍是候选。尺度搜索、舍入、精度、异常规则不同，本轮未按同核参数重复扣数。"},
    {"ids": ["A067", "A070"],
     "decision": "保留2项token格式机制；offset/length位宽及偏置参数化仍是候选，当前A067不覆盖A070完整范围。"},
    {"ids": ["A061", "A082"],
     "decision": "保留不同Snappy匹配实现；输出记录同型不能证明搜索与切token相同。"},
    {"ids": ["A035", "A038", "A039"],
     "decision": "保留Elf+/Gorilla/Chimp各自窗口规则；共享位写入不消除控制分支差异。"},
]


def accounting(implementation_id):
    if implementation_id in PROFILES:
        target, description = PROFILES[implementation_id]
        return {"layer": "profile", "mechanism_ids": [target], "description": description}
    if implementation_id in RECIPES:
        return {"layer": "recipe", **RECIPES[implementation_id]}
    if implementation_id in ADAPTERS:
        return {"layer": "adapter", "mechanism_ids": [], "description": ADAPTERS[implementation_id]}
    return {"layer": "mechanism", "mechanism_ids": [implementation_id],
            "description": "保留一个不同机制条目；不等于已证明不可继续分解，也不自动代表已接入图策略。"}


def validate_layers(implementations):
    ids = {item["implementation_id"] for item in implementations}
    classified = [*PROFILES, *RECIPES, *ADAPTERS]
    assert len(classified) == len(set(classified))
    assert set(classified) <= ids
    counts = Counter()
    for item in implementations:
        record = accounting(item["implementation_id"])
        if "accounting" in item:
            assert item["accounting"] == record, item["implementation_id"]
        assert set(record["mechanism_ids"]) <= ids
        for target in record["mechanism_ids"]:
            assert accounting(target)["layer"] == "mechanism", (item["implementation_id"], target)
        counts[record["layer"]] += 1
    assert sum(counts.values()) == len(ids)
    assert len({family for item in implementations for family in item["family_numbers"]}) == 30
    return dict(counts)
