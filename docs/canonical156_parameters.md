# 156 项算法与可学习的 LZ 输入参数

> 历史156项版本说明。Match Window的窗口大小也作为输入参数后，当前为143项；见 [143项实现与Window/LZ参数](canonical143_parameters.md)。

当前注册版本为 `cpp-canonical-156-v1`：28 类算子、156 项算法实现，算法 ID 为 `0–155`，`STOP=156`。原子策略网络的算法输出为 157 维。

原 182 项中的 27 个 `native_lz.aligned8.min*.max*` 配置使用同一套匹配算法，现在合为一个实现：`op.lz_match.native_lz.aligned8.cpp_v2`（ID 75）。`min_match`、`max_match` 是每次调用的输入参数，不再占算法 ID。因此总数为 `182 − 27 + 1 = 156`。窗口模型、匹配解析策略和 token 编码格式仍是各自的算子。

当前保留 180 个 I/O 配置。历史 206 个接口中的 27 个 LZ 配置映射为一个 I/O 配置加显式参数，原有输入输出能力继续保留。算法和来源对应关系见 [`canonical156.json`](../benchmark/reference/actions/canonical156.json)；历史 182 清单保持原样，供迁移核对。

## 参数、默认值和约束

| 项目 | 当前规则 |
|---|---|
| `min_match` | 最短可接受匹配长度，整数，3–256 字节 |
| `max_match` | 匹配长度上限，整数，8–258 字节 |
| 联合约束 | `min_match <= max_match`，且按 8 字节对齐后的最小长度不大于最大长度 |
| RL 的 min 候选 | 8、16、24、32、64 |
| RL 的 max 候选 | 8、16、…、256 |
| RL 的合法参数组合 | 满足 `min <= max` 的 147 组，独立于算法 ID |
| 默认值 | `min_match=8, max_match=256` |
| 未指定参数 | 读取已固定的默认值，不在待压缩输入上临时搜索 |

当前匹配器按 8 字节比较，因此有效 min 为 `max(8, ceil(min/8)*8)`，有效 max 为 `floor(max/8)*8`。RL 使用有效长度避免重复动作；兼容接口仍接受历史未取整的原始数值，并在阶段包中保留它们。搜索 min 与 token 中用于长度编码的 `length_bias` 已分开，改变搜索 min 不会顺带改变解码格式。

输入为窗口算子生成的阶段包（bytes），输出为携带窗口、min/max、原始长度等恢复信息的 LZ 阶段包（bytes）；后续解析算子生成匹配 token，token 格式算子输出编码字节。`min/max` 不改变原数据的整数或浮点语义，逆向恢复保留原始位模式。通过 `output="streams"` 调用时仍沿用 typed ports 的输入输出约束，参数写入恢复 header。

默认值依据固定校准集选定：Air-Quality 14 列和 Metro-Traffic 6 列，每列前、中、后各取一块，共 60 块、327,360 原始字节。在固定 32 KiB 窗口、aligned8 greedy 解析及 length bias 3 下，对 147 组参数比较包含恢复头的 NLC1 完整校准帧总大小。8/256 得到 271,833 字节，为本次候选最小值；全部 26,460 次逐位往返通过。代码发布版保留 [默认值记录](../benchmark/reference/actions/native_lz_defaults.json)，不包含校准结果目录。

“压缩比最好”在这里指**该校准集与候选范围内最好**。换数据、窗口、token 格式或下游编码后，最优值可能变化。NLC1 用于本次默认参数比较，其大小和计时不代表完整 UAR/RL 链的性能。校准样本不是保留测试集；正式训练与测试分离时，应只用训练侧数据重新校准并固定默认值。

## 调用例子

从仓库根目录运行：

```python
import sys
sys.path.insert(0, "scripts")
import unified_action_runtime as u

source = u.ActionData.bytes(b"ABCDEFGH" * 65)
actions = (
    "op.match_window_model.native_lz.window32768",
    "op.lz_match.native_lz.aligned8.cpp_v2",
    "op.match_parse_policy.native_lz.aligned8_greedy",
    "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3",
)
parameters = (None, {"min_match": 16, "max_match": 256}, None, None)
encoded = u.run_actions(actions, source, parameters=parameters)
assert u.run_actions(actions, encoded, parameters=parameters, reverse=True) == source

# 省略 parameters，LZ 使用固定的默认值 8/256。
encoded_default = u.run_actions(actions, source)
assert u.run_actions(actions, encoded_default, reverse=True) == source
```

`run_policy_action(..., parameters={...})` 可直接调用单个 LZ 算子。一个参数对象对应一个算子；多算子调用传等长的参数列表。逆向默认读取封包记录的参数，显式传入的参数必须一致。非 LZ 算子拒绝不属于自己的参数。旧名称中的 min/max 与显式参数冲突时会报错。

## 强化学习与重放

策略先选择算法，仅当选中 LZ（或含 LZ 的 recipe）时再从 147 个合法组合中选择参数。条件参数头使用状态特征，联合概率为 `P(算法|状态) × P(参数|状态, LZ)`；PPO/GRPO 的奖励、优势和裁剪概率比同时更新算法头与参数头。参数组合不扩充算法 ID 表。

路径记录同时保存 `ids`、`io_variants`、`parameters` 和注册版本。参数参与路径相等比较、哈希、缓存、切片、序列化和冷重放；Beam 在同一个 LZ 算法下枚举参数选择。checkpoint 保存两个策略头及参数目录/默认值指纹。旧网络维度不同，应重新初始化训练，不能把旧权重直接当作当前策略加载。

历史 182/206/225 ID 和名称可显式迁移，恢复其原始 min/max；历史固定算法配方也保留原参数，不用默认值覆盖。迁移逻辑由当前运行时代码维护。部分脚本文件名仍含 canonical182/canonical206 作为历史入口名称；运行时身份以版本字段为准。

开发时的验证覆盖参数化真实编解码、旧接口迁移、完整帧逆向、策略梯度、参数训练和 checkpoint 重载；代码发布版不包含验证回执。这里只完成代码接入和有限校准，没有据此声称经过大规模 RL 训练或优于已有压缩算法。
