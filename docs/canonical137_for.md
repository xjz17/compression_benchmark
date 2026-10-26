# canonical137：统一 minimum + offsets FOR

当前原子注册表为 `cpp-canonical-137-v1`：137 项算法、169 个 I/O 配置，FOR 由 4 项合并为 1 项。历史 `canonical140/143/156/182/206/225` 的 ID 不能作为新 ID 直接使用；保存的配方必须带注册表版本，迁移时保留原来的 I/O 配置和 LZ 参数。

项目仍为 **30 类算子与 3 种特殊动作**。原子表覆盖历史的 28 类；项目完整图目录采用独立 `operator-taxonomy-30-v2`：137 个原子算法、3 个额外列拆分实现、5 个概率模型提供器及 3 种特殊动作，共 148 条描述。概率模型条目不等于已接入熵编码的压缩算法。STOP / Concat / split_segments 不计为算子类别或压缩算法。原图目录 v1 保持冻结，可以回放；v1 的 ID 143 是 FloatDeconstruct，而 canonical143 的 ID 143 是 STOP。

## 通用契约

- 算法名：`op.frame_of_reference.minimum.cpp_v1`，当前 Policy ID `14`，文档编号 `P015`。
- 通用 I/O：`op.frame_of_reference.minimum.int64_uint64_le.cpp_v1`。
- 输入：signed int64 数值流（允许负数），或长度为 8 的整数倍的 uint64 little-endian 原始字节。uint64 不作 signed int64 重解释。float64 被拒绝，不执行可能丢失原始位模式的浮点减法。
- 核心：`minimum = min(values)`；`offsets[i] = values[i] - minimum`。输入可以是原始数值，也可以是上游生成的整数；不要求是残差。
- 输出：`minimum[1]` 与 `offsets[n]` 两个真正的数值流。minimum 类型沿用输入的 int64 或 uint64；offsets 类型恒为 uint64。`[-2^63, 2^63−1]` 合法，对应 offsets `[0, 2^64−1]`。空输入输出 minimum `[0]` 与空 offsets。
- 逆向：`values[i] = minimum + offsets[i]`；按输入类型检查恢复范围、offsets 最小值必须为 0（非空时）、类型与数量一致性。截断、冗余字节及非法范围被拒绝。

例如输入 `[1000, 1001, 1003]` 得到 minimum `[1000]`、offsets `[0, 1, 3]`。后续若使用统一最大值位宽打包，offsets 需要 2 bits/value；直接打包原输入需要 10 bits/value。minimum 和所有恢复元数据必须一起计入最终压缩大小。

## 真实实现与 I/O 适配

本轮注册的六个 FOR 接口及纳入的原压缩算法分支，其最低值与偏移计算使用 C++ 共享核心 `cpp/src/minimum_for.hpp`。Python 仅负责注册、调用 C++、类型检查、协议解析与图重建。

六个旧接口仍可显式选择：

| I/O 适配器 | 输入和外层结构 |
|---|---|
| `op.frame_of_reference.reference_v1` | 原参考 int64 接口及 OPA1 侧信息包装 |
| `op.frame_of_reference.bos.cpp_v1` | BDL1 差分包，保留首值与数量 |
| `op.frame_of_reference.reger.cpp_v1` | RAR1 预测值包，保留重排索引与首值 |
| `op.frame_of_reference.ts_2diff.cpp_v1` | S2D1 包；外层保留 128 项分块及首值 |
| `op.frame_of_reference.bos.post_outlier.cpp_v1` | BOS1 包；异常掩码、异常值和正常子集 gather/scatter 属于适配器 |
| `op.frame_of_reference.alp.cpp_v1` | ASC1 包；外层保留缩放、异常旁路及 1024 项分块 |

原来源名称决定原 I/O，不会因算法合并而把 BOS/REGER 路由到 reference。BOS 配方可以两次使用同一个 FOR 算法 ID，每次使用各自不同的 I/O 配置。分块、掩码和旁路信息均需保留，但不再作为不同的 FOR 算法计数。

未声明来源的原始 bytes 按 uint64 LE 数值解释，不因恰巧出现 S2D1、BDL1、BOS1、ASC1、RAR1 或 MFR1 前缀而选择私有格式。来源包可显式指定旧 I/O，或由合法上游 UAR 元数据中的生产者与格式共同识别。显式通用 I/O 按实际传入的原始字节解释，避免与传输头发生歧义。

## 两流执行与完整格式

`run_policy_action(..., output="streams")` 返回可重建的 ActionPorts。minimum 与 offsets 是独立 TypedStream；布局 metadata 仅保存类型、数量及必要字段，不隐藏原始输入。编辑 minimum 或 offsets 会影响逆向恢复。

在图 API 中，通用 FOR 生成两个**活动叶流**，分别可选后续压缩方法，例如 minimum → Huffman、offsets → 最大值位宽打包 → range。完整图帧保存必要的端口、类型与逆向边信息，能够在独立进程从叶流冷解码。旧来源接口保持历史包边界，旧配方继续回放。此处验证的是图执行接口，不代表现有 PPO/GRPO 已训练这种多分支图策略。

C++ 通用 CLI：

```text
minimum_for_cli encode-int64  < values_i64_le.bin > result.mfr1
minimum_for_cli encode-uint64 < values_u64_le.bin > result.mfr1
minimum_for_cli decode < result.mfr1 > restored_values.bin
```

MFR1 为 `magic[4] + kind[1] + count_u64le[8] + minimum_word[8] + offsets_u64le[8*n]`；kind 0 表示 signed int64，kind 1 表示 uint64。该 21-byte 头加 offsets 是变换的传输格式，不能把 offsets 大小当作完整压缩率。旧压缩配方继续输出原有格式，不强制改用 MFR1。

```python
import unified_action_runtime as u
import minimum_for as f

source = u.ActionData.ints([-9, -5, -1])
ports = u.run_policy_action(
    f.ALGORITHM_NAME, source,
    registry_version="cpp-canonical-137-v1", output="streams")
assert [(s.name, s.kind, s.count) for s in ports.streams] == [
    ("minimum", "int64", 1), ("offsets", "uint64", 3)]
restored = u.run_policy_action(
    f.ALGORITHM_NAME, ports,
    registry_version="cpp-canonical-137-v1", reverse=True)
assert restored == source
```

## 迁移和学习边界

`canonical137.json` 每行记录 `previous_140_policy_ids`，并保留此前 143/156/182 映射。旧 FOR 的 14、15、16、17 合并到 14；其余后续 ID 减 3。旧 merged-FOR 配方缺少 I/O 配置时，路径迁移明确拒绝歧义，不静默换成新格式。

当前线性策略动作头为 138 维（137 算法 + STOP），原 min_match/max_match 与 window_bytes 的独立条件参数选择保持原语义。算法、I/O、参数共同进入路径和缓存身份、搜索语法与 checkpoint 指纹。旧模型权重不会自动扩缩或被当成新模型。Qwen 保存记录更新为 `qwen-canonical137-bitviews-parameters-v6`；历史注册表配方可迁移，但旧绑定测得的压缩大小不得当作新完整格式的结果。

验证结果及完整命令见 `analysis_outputs/minimum_for_20260909_v1/CODE_VALIDATION.json`；旧压缩输出逐字节一致的 C++ 证据单独列出。有限测试集合上的字节一致性证明对应样本的压缩大小与压缩比保持相同，不声称已经重新测量全数据集。

本轮验证：156 项不同 Python 测试通过（包含 15 项图测试），9 项 CTest 通过；131 份冻结 stage/完整码流逐字节一致，其中 76 份完整压缩输出合计均为 13,465 bytes。额外复测不重复计入不同测试数。独立 ablation-only FOR 与 BUFF column-split 内 FOR 不在本轮 C++ 重构范围内。
