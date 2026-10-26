# 140项实现：按最大输入数值位宽定宽打包

当前注册表为 `cpp-canonical-140-v1`：28类、140项实现、168个I/O配置；Policy ID 0–139，STOP=140。本轮将 TS_2DIFF、LECO、ALP、REGER 的4项 bit-packing合为1项，143−3=140，Bit-packing类12→9。原窗口与LZ参数化继续有效；FOR算子和其变换函数未改动。

新实现名为 `op.bitpack_int.max_value_width.cpp_v1`，Policy ID 25，文档P026。它对输入的一组非负整数计算 `b = bit_width(max(input))`，每个数保存b位。空组和全零组的b均为0。输入不限于残差，不隐式减去最小值，也不隐式做Delta或ZigZag。例如 `[100,101]` 直接使用7位，而不是先减100后用1位。

四种旧输入数据包只作为I/O适配器，保留原有分组、基准、模型、首值、排列、缩放、异常/旁路信息。它们实际调用同一个 [C++核心](../cpp/src/max_value_bitpack.hpp)，而非只在清单中改名。核心支持MSB/LSB和字节/8字节对齐格式参数；当前四旧适配器均使用原有MSB字节位流。

| I/O配置 | 输入约束 | 输出与旁路信息 |
| --- | --- | --- |
| `op.bitpack_int.max_value_width.uint64_le.cpp_v1` | 非负int64，或8字节对齐的raw little-endian uint64数组bytes；值域0–2^64−1，允许空数组 | MVW1 bytes：count:uint64 LE、width:uint8、MSB packed bytes；一个输入数组为一组 |
| `op.bitpack_int.ts_2diff.cpp_v1` | S2F1包中的uint64数值组 | 保留TS_2DIFF块边界、first、minimum、浮点缩放信息与原公共位流 |
| `op.bitpack_int.leco.cpp_v1` | LAR1包中的uint64数值组 | 保留LECO分块、截距/斜率、长度和原位流 |
| `op.bitpack_int.alp.cpp_v1` | AFO1包中的uint64数值组 | 保留ALP分块、base、scale、异常位模式和原位流 |
| `op.bitpack_int.reger.cpp_v1` | RFO1包中的uint64数值组 | 保留REGER原排列、首值、base和原位流 |

通用profile拒绝负int64与float64；若需要压缩带符号整数，调用者先选择适合的独立有符号变换。uint64完整范围通过原始LE字节数组输入，不能把负int64静默重解释为无符号值。

```python
import struct
import primitive_operator_action_runtime as p
import unified_action_runtime as u

name = "op.bitpack_int.max_value_width.cpp_v1"
source = u.ActionData.ints([0, 1, 7, 2])
encoded = u.run_policy_action(name, source, registry_version=p.ACTION_REGISTRY_VERSION)
restored = u.run_policy_action(name, encoded,
    registry_version=p.ACTION_REGISTRY_VERSION, reverse=True)
assert restored == source

# uint64满域：明确使用little-endian无符号数组bytes。
unsigned = u.ActionData.bytes(struct.pack("<3Q", 0, 1 << 63, (1 << 64) - 1))
encoded_unsigned = u.run_policy_action(name, unsigned,
    registry_version=p.ACTION_REGISTRY_VERSION,
    io_variant="op.bitpack_int.max_value_width.uint64_le.cpp_v1")
```

通用MVW1码流为 `magic[4] + count:u64le + width:u8 + packed:bytes`，头部13字节。`[0,1,7,2]` 的数值payload为12位、按字节补齐为2字节，完整MVW1为15字节；公共运行时另有UAR类型/接口元数据，不能把2字节当作完整压缩结果。`output="streams"` 暴露一个 `packed:bytes` stream，count/width作为可重建的旁路元数据保存，逆向恢复原来的int64或bytes输入。

裸bytes默认按通用uint64 LE数组解释，即使前4字节碰巧为S2F1/LAR1/AFO1/RFO1也不会误判成私有包。私有包可通过旧实现名字或显式`io_variant`选定；若输入是已有正确上游算子的UAR结果，则按producer与包magic一起自动选择对应适配器。逆向使用结果中保存的I/O配置，不猜测来源。

## 兼容、搜索与验证

[canonical140.json](../benchmark/reference/actions/canonical140.json)保留逐项`previous_143_policy_ids`、`previous_156_policy_ids`与`previous_182_policy_ids`。历史143/156/182清单不变；旧143/156/182/206/225 ID必须带版本显式迁移。四旧名称/固定压缩配方继续保留各自I/O配置与位流，并不会都转换成TS_2DIFF格式。

路径、缓存、冷重放、PPO/GRPO、完整recipe与Qwen binding使用140项算法空间和168种I/O配置。通用profile可作为单独数值根算子；同一算法的五个I/O分支在路径和缓存中保持不同。Qwen的通用uint64 profile以原始bytes解释无符号LE数值，保留所有输入位模式。新算法头为141维（含STOP），旧模型权重需重训，不能仅改标签；既有Window/LZ条件参数候选和冻结默认值不变。

验证包括：四旧适配器45个改前二进制冻结样例字节一致；共享核心0–64位宽、4种格式共3575次往返；CLI空/零/满域与坏输入拒绝；四个C++测试目标；公共运行时一般数值、0位/64位、四magic冲突、类型与元数据校验、typed streams、五个I/O分支、历史版本迁移、固定配方、搜索/RL/checkpoint及Qwen往返。14个FOR及相关序列化函数经改前后逐字符SHA核对未变。

对应的 C++、Python 与注册表验证由测试代码覆盖；代码发布版不包含开发阶段的验证回执。
