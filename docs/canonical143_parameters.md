# 143 项实现：Window 与 LZ 作为输入参数

> 历史143项版本说明。四种最大值位宽bitpack接口合并后，当前为140项；见 [140项实现与最大输入数值位宽打包](canonical140_bitpack.md)。下文的Window/LZ参数语义仍适用，编号以当前清单为准。

当前代码注册表是 `cpp-canonical-143-v1`：28 类、143 项实现、167 个 I/O 配置，Policy ID 为 0–142，STOP 为 143。

本轮只合并 Match Window Model 中 14 个仅窗口大小不同的 native 配置：156 − (14 − 1) = 143。新的实现名为 `op.match_window_model.native_lz.window.cpp_v2`，当前 Policy ID 87（文档 P088）。`op.match_window_model.reference_v1` 仍独立：它执行位置相关的 `XOR(i×131)` 可逆变换，不维护真实 LZ 历史窗口，因此不能合并为同一算法的窗口参数。

完整实现表见 [canonical143.json](../benchmark/reference/actions/canonical143.json)。历史 [156 表](../benchmark/reference/actions/canonical156.json) 和 [182 表](../benchmark/reference/actions/canonical182.json) 保持不变；当前每行记录 `previous_156_policy_ids` 与 `previous_182_policy_ids`。

## 输入、输出与参数

| 算子 | 输入 | 参数 | 输出 |
| --- | --- | --- | --- |
| Native Match Window Model | 原始 bytes（可为空） | `window_bytes`：整数 1–65535，默认 65535 | NLS1 role=1 字节包；语义为原始 byte stream 加窗口大小、长度等元数据 |
| Native LZ Match | NLS1 role=1 bytes | `min_match`、`max_match`，既有范围与约束不变，默认 8/256 | NLS1 role=2 bytes，保留窗口并写入匹配上下界 |
| Native Match Parse Policy | NLS1 role=2 bytes | 从包中读取上述配置 | NLS1 role=3 的 literal/offset/length 记录 |

Window 阶段只写入上下文，本身没有压缩收益；后续 `aligned8_greedy` 执行搜索，要求候选向后距离不超过 `window_bytes`。上限 65535 来自当前 offset16 token 格式。此实现保存每个 8 字节键的最近位置；增大窗口不等于增加哈希链搜索候选数，也不保证对所有数据单调改善压缩率。更大窗口的实际收益和时间开销需按完整链测量。

公共 `run_policy_action` 默认输出带元数据的 UAR bytes；`output="streams"` 可取得可重建的 typed streams。参数不是额外数据 stream，完整 UAR/NLS1 元数据必须保存以支持逆变换。参数缺省时使用冻结默认值，逆向从已编码结果读取参数；历史名字、旧 ID 和显式参数不一致时直接报错。

```python
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import native_window_parameters as window

source = u.ActionData.bytes(b"ABCDEFGH" * 128)
encoded = u.run_policy_action(
    window.ALGORITHM_NAME,
    source,
    registry_version=p.ACTION_REGISTRY_VERSION,
    parameters={"window_bytes": 32768},
)
restored = u.run_policy_action(
    window.ALGORITHM_NAME,
    encoded,
    registry_version=p.ACTION_REGISTRY_VERSION,
    reverse=True,
)
assert restored == source
```

## 默认值与有限校准

默认 `window_bytes=65535` 是 [冻结校准记录](../benchmark/reference/actions/native_window_defaults.json) 中 14 个历史候选的最优聚合值。候选为 3072、4096、5120、8192、12288、16384、19456、24576、28672、32768、40960、53248、57344、65535。

校准固定 min=8、max=256、aligned8 greedy 和 token length bias=3。Air-Quality 与 Metro-Traffic 的 86 个样本块共 2,424,512 原始字节，最佳完整 NLC1 帧聚合为 1,436,749 字节。14 候选 × 86 块 × 3 次，共 3,612 次逐位无损往返均通过。压缩与解压时间另有记录，不参与默认值选择。该结论只适用于记录的样本、候选和固定其余参数；不是任意数据的最优保证，也不是三参数联合全局最优。

## 搜索、学习与兼容

原子决策保持独立：Window 动作有 14 个学习候选，LZ 动作有 147 个有效 min/max 配置。原子算法数为 143，参数值不增加算法数量。整条 native recipe 同时包含两个参数阶段，因此提供 14 × 147 = 2,058 个联合配置，支持同时学习 window/min/max。

当前条件参数头共用 2,219 个配置字典：147 个 LZ、14 个 Window、2,058 个联合 recipe 配置。对应的合法掩码只允许当前动作的 147、14 或 2,058 个候选。PPO/GRPO 同时使用算法与参数的联合概率比，参数头也获得压缩奖励梯度。路径、缓存、冷重放、固定配方与 checkpoint identity 均包含参数；状态特征单独保留窗口与匹配上下界。

旧 156/182/206/225 ID 必须带注册表版本迁移；旧窗口名字仍表示原来的窗口值。固定压缩方法的原始 window/min/max 配置继续保留。旧训练权重不能直接改名为新 144 维（143+STOP）算法头或当前条件参数头，应迁移历史路径、重新测量并训练。

```python
from canonical182_policy_paths import path_from_names, path_record

# 旧名字精确保留 window=32768 和 min=3/max=258。
path = path_from_names([
    "op.match_window_model.native_lz.window32768",
    "op.lz_match.native_lz.aligned8.min3.max258",
    "op.match_parse_policy.native_lz.aligned8_greedy",
    "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3",
])
record = path_record(path)  # 当前 ID + io_variants + parameters + registry version
```

验证覆盖历史注册表迁移、冻结 206 I/O payload、不同比例窗的完整压缩/解压、窗口及 LZ 参数独立缓存、旧窗口包逆向、冲突参数拒绝、typed streams、联合 recipe、条件 mask/梯度、checkpoint 与 Qwen binding。
