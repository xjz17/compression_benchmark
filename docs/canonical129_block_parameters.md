# Canonical129: Block State Control 参数归并

本页记录冻结的 canonical129 版本。后续 RLE 实现替换及当前计数见
[Canonical127 统一 Run Length](canonical127_unified_rle.md)。

当前基础注册表为 `cpp-canonical-129-v1`：129 项算法、165 种 I/O 形式。30 类图目录为 `operator-taxonomy-30-v3`：132 项变换/编码、5 项概率模型，共137个条目。STOP、Concat、连续 Split 单独计数，图描述总数140。

## 合并结果

| 合并前条目（旧145条目Word编号） | 当前算法 | 参数或兼容形式 |
| --- | --- | --- |
| P116–P118 Blosc1 t1/t4/t8 | P115 `blosc1_residual_core.cpp_v2` | `typesize=1/4/8` |
| P119–P121 Blosc2 t1/t4/t8 | P116 `blosc2_residual_core.cpp_v2` | `typesize=1/4/8` |
| P123、P124、P128 gzip/zlib/CORAD DEFLATE | P118 `deflate_residual_core.cpp_v2` | `level=0..9`；3种显式 I/O |
| P108 block reference、P081 window reference | P081 `op.match_window_model.reference_v1` | 同一个 XOR(index×131) 分支；保留两种旧名称 |
| P113、P132 magic/length envelope | P112 `op.container_frame.magic_length.cpp_v3` | 同一封装函数；6种 I/O |

第27类由21项变为13项，第28类仍为9项。前三行算法全名均以 `op.block_state_control.` 开头。两版 REGER、不同版本的 Blosc2 宏及其他操作不同的编码后端仍分别保留。

## 输入、输出与参数

Blosc1/2 输入合法参考前缀 bytes，先撤销 token、LZ、shuffle 前缀，再调用对应原生完整编码后端；输出完整 Blosc bytes。`typesize` 必须等于输入 SHF1 前缀的元素字宽，逆向还检查编码头字宽。不匹配直接拒绝。改变它需要同时选择相容的上游 shuffle 形式。其余配置沿用旧值：clevel=5、shuffle启用、线程数1。

DEFLATE 输入输出均为 bytes，必须选择具体来源协议，不能仅提供算法名称或通过 magic 猜测：

| `io_variant` 名称中段 | 合法输入 | 输出 | 兼容默认 level |
| --- | --- | --- | --- |
| `gzip_prefix` | gzip参考前缀 | DFC1 G型，含gzip头/core/尾 | 6 |
| `zlib_prefix` | zlib参考前缀 | DFC1 Z型，含zlib头/core/尾 | 9 |
| `corad_ctk2` | 有效 CTK2 包 | CZL2，含原长度与zlib流 | 9 |

完整形式如 `op.block_state_control.deflate_residual_core.gzip_prefix.cpp_v2`。三者使用 `cpp/src/parameterized_deflate.hpp` 的同一个 zlib DEFLATE 内核；保留各自输入恢复与输出封装步骤。相同内核不意味着不同封装的总大小相同。

Magic/length 接受既有 bytes 载荷，输出4字节标识、uint64LE长度及原载荷，增加12字节。B1F1/B2F1/GZF1/ZLF1/XZF1/CRD2 归入六种 I/O；CRD2 仍要求合法 CZL2 输入。

## 运行时与强化学习接口

单阶段参数使用 `{"typesize": 4}` 或 `{"level": 7}`。配方级参数分别使用 `blosc1_typesize`、`blosc2_typesize`、`deflate_level`，经 `current_action_parameters.for_stage` / `for_option` 转换，避免两个 Blosc 阶段的同名键互相覆盖。

新 Blosc 单阶段名称默认 typesize=1；当前原子搜索根据上游前缀选择唯一合法字宽，已有配方保留其上游固定字宽。历史 t1/t4/t8 名称始终恢复1/4/8，显式冲突拒绝。DEFLATE默认值取决于 I/O。上述默认用于兼容旧配置，不宣称对任意数据都是压缩比最优。

算法头现为130维（129算法+STOP）。条件参数接口包含5项参数化算法：LZ、Window、Blosc1、Blosc2、DEFLATE。当前候选目录为2235项，包含旧联合LZ/Window域和新增的3+3+10个候选；当前配方目录中的参数组合全部覆盖，不声称是任意五阶段组合的笛卡尔积。输入合法性掩码约束 Blosc 字宽。

当前路径、帧、缓存及特征保存明确的参数与 I/O。旧 checkpoint 需匹配注册表、动作数、特征和参数目录，不能仅改ID或扩缩输出头。本次没有训练新模型。

## 版本与码流兼容

旧 canonical137 及更早注册表、旧图 v1/v2 保持各自ID和协议；通过明确旧版本迁移。基础新版本 STOP=129，图 v3 的129是 FloatDeconstruct，STOP=137、Concat=138、连续 Split=139。旧数值ID不能直接用于新注册表。

保留原名称、原输入输出形式和原默认参数。旧169种 I/O 中4个仅 typesize 不同的配置转为参数，因此当前165种；旧调用仍可解析。原生默认编码载荷用修改前冻结样本逐字节核对。公共版本化帧包含当前注册表等元数据，完整帧大小仍须实测，不能把元数据变化算成内核压缩收益。

## 验证与代码入口

2026-09-09：147项 Python 回归测试全部通过；运行时自检通过；59项原生旧结果（53份码流、6项原有拒绝）保持一致，150项DEFLATE级别冷进程往返、15项新默认与旧码流对比、48项Blosc字宽不匹配拒绝检查通过。范围为有限回归，没有新的全量压缩率或性能基准。

- 当前注册表：`benchmark/reference/actions/canonical129.json`、`scripts/canonical129_registry.py`。
- 参数与公开运行时：`scripts/block_state_parameters.py`、`scripts/current_action_parameters.py`、`scripts/unified_action_runtime.py`。
- 图及迁移：`scripts/operator_taxonomy_v30_v3.py`、`scripts/graph_registry_migration.py`、`scripts/typed_graph_runtime.py`。
- 原生共享实现：`cpp/src/parameterized_deflate.hpp`、`cpp/src/general_codec_residual_stages.cpp`、`cpp/src/corad_lossless_cpp.cpp`。
- 策略及配方：`scripts/canonical182_policy_paths.py`、`scripts/canonical206_search_env.py`、`scripts/canonical206_recipe_env.py`、`scripts/qwen182_action_binding.py`。
- 可复核结果：`analysis_outputs/block_state_merge_20260909_v1/FINAL_VALIDATION.json`、`CPP_VALIDATION.json`、`PYTHON_VALIDATION.json`、`RUNTIME_SELF_CHECK.json`；原生重放脚本为同目录的 `validate_cpp.py`。
