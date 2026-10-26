# Canonical127：统一 Run Length

第 15 类 Run Length 从 3 个实现替换为 1 个共享 C++ 实现：
`op.run_length.byte_optimal.cpp_v1`，当前基础 ID 为 65（文档编号 P066）。
这是编码实现的替换，不是声称原来的三个算法等价。

## 原来三项的区别

| 原实现 | 对什么做游程编码 | 还必须保存什么 |
|---|---|---|
| Value/Count | 连续相同的输入值；整数/字节是不同 I/O 形式 | 值和重复次数 |
| TsFile Hybrid RLE | 同值段与定宽打包段混合，采用固定阈值和分组规则 | 全局位宽、段类型、长度、普通值 |
| RLBE | 连续相同的残差有效位宽 | 位宽序列之外，还必须保存每个残差的有效位 |

例如输入残差 `[2, 3, 2, 3]` 并无同值长游程，但位宽是 `[2, 2, 2, 2]`。
RLBE 可以压缩该位宽序列，仍需保存四个残差。不能只留下“2 重复 4 次”。

## 统一内核与“最优”的范围

`cpp/src/optimal_rle.cpp` 接收 1–8 字节定宽原始符号，比较以下完整 URL1 表示：

1. 原样保存；
2. 整列为同一值；
3. 全列按最大有效位宽打包；
4. 位宽打包普通段与重复段混合，用动态规划求最小字节分段。

普通段最长 128 个符号；重复段不受该长度限制。所有长度码、位宽、类型宽度、
模式和帧头都计入大小。相同大小使用固定优先级，输出可重复。

**保证是这四种表示及上述分段规则内的完整字节数最小。** 不保证对任意输入、
任意其他格式或任意后续压缩算法全局最优，也不保证比历史格式逐例更小。
动态规划为此付出编码时间和 O(N) 辅助空间，不能把压缩率目标当作速度最优。

格式、复杂度、最优性论证和独立穷举验证见
`analysis_outputs/unified_rle_20260909_v1/OPTIMAL_RLE_FORMAT.md` 与
`OPTIMAL_RLE_CORE_VALIDATION.md`。

## 输入和输出约束

| 当前 I/O 形式 | 输入 | 输出 |
|---|---|---|
| `words64` | int64、float64 原始位型，或显式选择的 8 字节对齐 bytes | URL1 bytes |
| `uint8` | 任意 bytes，每字节为一个符号 | URL1 bytes |
| `tsfile_scaled` | 合法 RLS1：scale、count、scaled int64 值 | URS1 bytes，保留 scale 和 URL1 |
| `rlbe_lengths` | 合法 RLD1：count 和全部 int64 残差 | URB1 bytes：位宽的 URL1 加残差有效位载荷 |

这些是一个内核的输入适配，不增加算法项。固定宽度整数包括有符号数的原始位型；
float64 不经过数值转换，保留 ±0、NaN payload、无穷和 subnormal。
原生核心另支持宽度 1–8，完整数值 codec 支持 int32/int64/float32/float64。
缩放步骤必须单独满足其可逆约束；不能把不合法 RLS1 当作原始数值猜测。

完整浮点数值入口继续接收 `max_point_number`（0–18）。其 URF1 帧保留原始
浮点宽度、请求/实际 scale 和模式：缩放可逐位恢复时压缩 scaled integers；
否则显式改为压缩浮点原始位型。两种模式均调用同一 URL1 内核，并计入全部帧头。
URS1 是既有合法缩放包的算子输出；URF1 是包含缩放选择的完整数值 codec，不能混用。

默认解码输出上限为 256 MiB，可通过显式 API 参数调整。截断、额外尾字节、
计数溢出、类型宽度冲突、非规范长度码和非零 padding 均拒绝。

URL1 是完整编码 byte stream，具体模式不一定存在固定 values/counts 两列。
URB1 的残差载荷是真正的数据流，不能计为免费元数据或丢弃。

## 接线与第三方边界

- 项目 RLE 完整数值编码器、Python 数值入口与当前算子调用使用共享内核。
- RLBE 保留 delta 与块重置规则，只将位宽序列的游程编码交给共享内核。
- Sprintz 保留预测、ZigZag 和打包主体，只替换原本使用 RLE 的整数尾组；
  原本的 raw 或 Gorilla 尾组不凭空增加 RLE。
- Bzip2 的项目参考前缀使用新内核；其 residual 适配器还原该前缀后调用
  原第三方 Bzip2，因此该前缀替换不等于 Bzip2 内部的 RLE 被替换。

对 86 项逐一追踪的源码记录在 `RLE_86_SOURCE_AUDIT.json`/`.md`。
逐项最终替换范围在 `RLE_86_REPLACEMENT_SCOPE.json`/`.md`。
86 个名称中有 47 项实际上使用项目 native-LZ 参考内核，不能根据同名上游
算法的内部结构声称替换了本地并未调用的代码。

Bzip2、DEFLATE、Brotli 等第三方格式内部的零游程、码长游程或常量块
与格式上下文绑定。本次保留这些格式；改成 URL1 会形成新的自定义编码格式，
该范围仍需用户确定。LZ 历史匹配、固定 Simple8b 零组、单符号熵编码等
也不应一概当成相同的通用 RLE 来替换。

## 版本和计数

| 项目 | 当前值 |
|---|---:|
| v1 保留算法槽位 | 127 |
| v2 活动算法实现 | 126 |
| v2 活动 I/O 形式 | 164 |
| 基础 STOP ID | 127 |
| 30 类图目录的活动变换/codec | 129 |
| 概率模型提供器 | 5 |
| 退役兼容描述符 | 1 |
| 图目录描述符，不含特殊操作 | 135 |
| 特殊操作 STOP / Concat / 连续 Split | 3 |
| 图描述总数 | 138 |

当前活动版本为 `cpp-canonical-127-v2` 和 `operator-taxonomy-30-v5`；
`cpp-canonical-127-v1` 和图 v4 是冻结兼容版本。
图 STOP/Concat/连续 Split ID 分别为 135/136/137；特殊操作不计入 30 类。
五个概率提供器仍按原能力状态列示，目录计数不表示它们全部已接入压缩运行时。

旧 canonical129、图 v3 和更早版本保留原 ID 与码流语义。
迁移旧配方明确标记 `implementation_replacement`、`requires_remeasurement`；
不能沿用旧压缩结果或重标旧模型 logits。
旧 TsFile RLE 前面若是缩放步骤，配方迁移选择 `tsfile_scaled`；原始整数入口
选择 `words64`。旧 Sprintz 的尾组替换采用独立的当前 I/O 名称，仍只计一个打包实现。

当前策略头是 128 维（127 个保留动作槽加 STOP，其中槽位 5 不可选）。Qwen binding 为
`qwen-canonical127-bitviews-parameters-v9`，检查点保留严格版本、特征和目录指纹校验。

验证结果与完整码流的大小、编码/解码时间由
`analysis_outputs/unified_rle_20260909_v1/FINAL_VALIDATION.json` 汇总。
有限样本实验不代表全部 86 方法的性能基准。

## 有限实测结果（2026-09-10）

选取 10 组输入，每组 4099 个值，比较 RLE、RLBE、Sprintz；包含 Air-Quality 的两个原始列和合成列。
下表为原生 C++ 完整数值 codec 的实际字节数。时间是同一 `-O2` harness 的进程内 11 次中位数，包含分配，不含 CLI 启动。

| 输入 / 方法 | 旧 → 新字节 | 编码 ms，旧 → 新 | 解码 ms，旧 → 新 |
|---|---:|---:|---:|
| air_first4099_NO2(GT) / RLE | 28988 → 5314 | 0.0390 → 1.4811 | 0.0341 → 0.0326 |
| air_first4099_NO2(GT) / RLBE | 17720 → 18160 | 0.2914 → 2.0169 | 0.0982 → 0.1440 |
| air_first4099_T / RLE | 34702 → 30916 | 0.0291 → 1.5368 | 0.0167 → 0.0969 |
| air_first4099_T / RLBE | 30117 → 30736 | 0.5020 → 2.0710 | 0.1570 → 0.1983 |

30 组原生及 30 组实际 Python benchmark 入口均逐位往返通过。原生 RLE 为 8 组更小、2 组更大；RLBE 为 1 组更小、9 组更大。原生 64 位 Sprintz 原本为 raw 尾组，10 组字节不变；Python benchmark 的整数尾组使用 RLE，其新旧结果另列在 CSV 中。

这些数据明确表明：统一实现不等于全部压缩算法都获得更高压缩比；动态规划也增加了 RLE 编码耗时。Python 新入口的墙钟时间含进程创建、IPC 和数值 marshaling，不能直接当成 C++ 内核耗时。

完整结果：`analysis_outputs/unified_rle_20260909_v1/numeric_audit/numeric_metrics.csv`、`numeric_metrics.json`。全部外层图/UAR元数据在集成测试中另作完整 frame 大小校验，未与这里的完整数值 codec 字节数混为同一个指标。
