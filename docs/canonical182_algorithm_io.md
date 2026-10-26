# 182 项算法与输入输出变体

> 本文保留 206→182 的历史迁移说明。LZ 的 min/max 改为输入参数后，当前版本在进一步合并Window参数后已更新为 `cpp-canonical-143-v1`；使用方法与当前编号见 [143项实现与Window/LZ参数](canonical143_parameters.md)。下文的“新编号”均指历史 182 版本。

该版本为 `cpp-canonical-182-v1`：28 类、182 项算法，编号 `0–181`，`STOP=182`。原子策略网络输出 183 维（182 算法 + STOP）。编号按《28类算子_按算法合并后的182项实现.docx》的 A001–A182 顺序排列，文档编号减一即代码 policy ID。

此次将 15 组合并为一个算法身份，共减少 24 项。206 个原接口仍作为 `io_variant` 保留输入类型、位宽、包布局和逆向恢复规则，不再占用独立算法 ID。已有 C++ 接口承担格式适配与兼容解码；兼容函数/descriptor 的数量不用于统计算法数量。源清单为 [`canonical182.json`](../benchmark/reference/actions/canonical182.json)，包含原文档 SHA-256、每项算法及原 206 编号。

## 调用与类型约束

使用 `unified_action_runtime.run_policy_action`，传入当前 `registry_version`。同一算法下有多种可用格式时传入 `io_variant`；只有输入类型与阶段包上下文能唯一确定格式时才自动选择。不会通过尝试压缩或随意取第一个变体来解决歧义。

输入为 `ActionData` 或 `ActionPorts`。原始列的 `ActionData.kind` 为 `int64`、`float64` 或 `bytes`。无符号残差和分肢流通过 `ActionPorts` 中的 `uint64`、`uint32`、`uint8` 表达，保留实际位模式，不转换成浮点数，也不把大于 INT64_MAX 的值当作 signed 数值。

| 参数/结果 | 含义 |
|---|---|
| `output="wire"`（默认） | 完整 UAR1 字节封包，含算法版本、编号、变体、源类型/数量和上游恢复信息 |
| `output="streams"` | `ActionPorts(streams, metadata, header)`；可直接作为后续调用或逆向调用的输入 |
| `TypedStream` | `name, kind, payload, count`；数值流使用小端，严格验证字节数等于数量乘位宽 |
| `metadata` | 各流的布局、形状及重建约束；修改流后按新内容重建阶段包 |
| `header` | UAR1 的恢复上下文；保留算法、I/O 变体和上游链 |
| `reverse=True` | 从封包记录的变体执行逆向，恢复原始类型和完整上游封包 |

`registry()` 的 `output_kinds` 沿用描述器口径，包含逆向恢复类型；**不表示编码位流变成了整数或浮点数**。实际前向 stream 类型请查 `semantic_ports`、每个变体的 `semantic_output` 或结果中的 `streams`。

合并涉及的 39 个变体均有明确输出 schema：24 个数值布局、10 个结构化字节布局、5 个编码位流。Gorilla/Pongo 的窗口打包、FPC/pFPC 的残差打包及单路 native-LZ ctrl8 结果保持 bytes。其他已识别阶段也可解析；尚未识别的输出保留完整 bytes，不虚构数值字段。gzip/zlib/xz 的 header、core、trailer 可拆分重建，但修改 core 后不会自动重算原有校验和。

### 例：一个 Delta 算法，两个 uint32 stream

从仓库根目录运行：

```python
import sys
sys.path.insert(0, "scripts")
import primitive_operator_action_runtime as p
import unified_action_runtime as u

source = u.ActionData.ints([100, 103, 106])
algorithm = "op.delta.previous_modular.cpp_v2"
result = u.run_policy_action(
    algorithm, source,
    registry_version=p.ACTION_REGISTRY_VERSION,
    io_variant="op.delta.stream_vbyte.cpp_v1",
    output="streams",
)
assert [(s.name, s.kind, s.count) for s in result.streams] == [
    ("low_deltas", "uint32", 3),
    ("high_deltas", "uint32", 3),
]
restored = u.run_policy_action(
    algorithm, result,
    registry_version=p.ACTION_REGISTRY_VERSION,
    reverse=True,
)
assert restored == source
```

同一算法的 `op.delta.reference_v1`、`op.delta.delta_binary_packed.cpp_v1` 和 `op.delta.bos.cpp_v1` 分别保留文档规定的单流或阶段包输入，不新增策略 ID。传入同 kind 的数据仍可能存在格式歧义，此时调用必须指定变体。

## 合并编号对照

| 文档编号 | 新 policy ID | 原 206 policy IDs | 算法 |
|---|---:|---|---|
| A008 | 7 | 80, 86 | Exact decimal scaling |
| A015 | 14 | 2, 112, 146 | Whole-stream minimum FOR |
| A019 | 18 | 3, 144 | Stable value sort |
| A037 | 36 | 7, 84, 94, 111 | Previous-value modular delta |
| A044 | 43 | 8, 97, 102 | Previous-word XOR |
| A045 | 44 | 28, 99 | Chimp history-128 XOR |
| A054 | 53 | 98, 103 | XOR window packing |
| A056 | 55 | 108, 110 | Nibble residual packing |
| A062 | 61 | 11, 91, 95 | Signed-64 ZigZag |
| A066 | 65 | 158, 159, 160 | Byte shuffle |
| A072 | 71 | 14, 167 | Value/count RLE |
| A131 | 130 | 59, 188 | Native-LZ ctrl8, bias 5 |
| A158 | 157 | 163, 164 | Block magic/length envelope |
| A176 | 175 | 133, 134, 135 | Stored header/core/trailer assembly |
| A177 | 176 | 170, 171, 172, 205 | Container magic/length envelope |

除这些组外的 167 项保留独立算法身份。没有合并模型、重新分块、重置/回退规则或搜索范围不同的实现。

## 旧数据、路径与模型

```python
step = p.migrate_policy_action(84, registry_version="cpp-canonical-206-v1")
assert step == {
    "policy_action_id": 36,
    "io_variant": "op.delta.stream_vbyte.cpp_v1",
}
```

旧 206 和更早的 225 编号必须给出源注册版本，迁移同时产生新 ID 与准确变体。单独重编号会丢失格式选择。已有 UAR1 包可按记录的旧阶段名解码；新版封包会检查版本、算法、编号与变体一致性。`run_action(int)` 的历史 descriptor 编号保持兼容；新编号只用于 `run_policy_action`。

搜索路径使用 `canonical182_policy_paths.PolicyPath`，同 ID 的不同变体在缓存中是不同路径。存盘使用 `path_record()`，或同时记录 `selected_ids`、`selected_io_variants` 和 `policy_registry_version`。切片、拼接、冷编码与完整帧解码均保留变体。

旧 207 维策略权重不能直接视为新的 183 维权重。新 checkpoint 绑定注册版本、菜单顺序、环境类型和特征配置；不匹配时拒绝加载。可迁移旧候选路径后重测，模型需要按新空间重新训练。Recipe/depth 的宏菜单是另一层控制选项，不等于算法数量。

Beam 可枚举显式变体；目前原子 RL 仅在输入上下文唯一确定变体时执行，歧义时拒绝。二者尚非完全相同的可达搜索空间，不能据此直接宣称新的 RL–Beam 公平比较。保留 `canonical206_*` 脚本名与 C206 外层帧 magic 以兼容既有入口；当前语义版本以注册表及帧内元数据为准。原有仅面向冻结实验的 C206 报表脚本不作为新 182 实验结果。

## 验证

- 39 个旧接口的 195 个二进制 fixture 在修改前冻结；逐项比较旧载荷、解码旧封包、新封包往返和 ActionPorts 往返。
- 17 项新兼容/边界测试，11 项语义端口测试，35 项搜索与策略集成测试通过。
- 原始 primitive 自检 387 次往返、unified 全类型/阶段自检通过；lossless64 的 48 个真实数据和 74 个边界检查通过；参考 C++ 算子回归通过。
- 修复两个缩放实现的 +2^63 上界判断，并重建 TS_2DIFF/Simple8b CLI 与单元目标；两个 C++ 单元通过。其他新增检查覆盖 NaN/Inf、负零、尺度 0–18、signed/unsigned FOR 极值、错误 magic 和封包身份篡改。
- 已验证裸 GXF1/CXF1 浮点阶段包的后续打包逆向，以及 BOS64 按阶段包格式选择整数/缩放分支。

没有重新运行 182 空间的压缩性能实验；旧压缩率与计时不作为本次迁移后的结果。UAR1 增加变体和版本元数据，完整帧大小应重新测量。

可复跑的关键命令：

```sh
backend/.venv/bin/python scripts/test_cpp_action_migration.py --inventory-only
backend/.venv/bin/python scripts/test_canonical182_io_runtime.py
backend/.venv/bin/python scripts/test_action_semantic_ports.py
backend/.venv/bin/python scripts/test_canonical182_search_integration.py
```

导出清单、完整旧编号映射与验证日志位于 `analysis_outputs/canonical182_migration_20260908_v1/`。
