# Pure ZigZag v1：变换、打包、压缩、封装分开

## 版本边界

旧 canonical206 的 `ZigZag-reference[11]` / `op.zigzag.reference_v1` 保留原有
`ZigZag → signed compact → OPA1 metadata → zlib6 → UAR1/C206` 行为，方便复现旧结果。
本次未修改旧 runtime、旧 C++、旧 CLI、旧 CMake 或任何冻结实验。

新的独立命名空间是 `pure-numeric-v1`，其中 namespace-local ID **11** 为
`op.zigzag.int64_to_uint64.pure_v1`。相同数字不表示可以跨版本替换：新 API 用明确名称调用，
禁止将裸 `11` 无版本地传入旧 canonical206 搜索。

| 显式步骤 | 输入 | 输出 | 是否压缩/封装 |
|---|---|---|---|
| `zigzag` | little-endian int64，8N B | little-endian uint64，8N B | 只变换，无打包、压缩、元数据 |
| `compact_uint64` | uint64，8N B | `UCV1` + count(u64) + width(u8) + 定宽数据 | 独立打包器；头部13 B；每值1/2/4/8 B |
| `zlib6` | bytes | 标准 zlib level-6 stream | 独立压缩后端 |
| `frame_wrap` | 明确的输入、输出和步骤列表 | `PNV1` + JSON长度 + JSON + payload | 独立外层存储封装，不压缩 |

注意是“**非负整数**”，不是严格正整数：0仍然映射为0。
`x >= 0 ? 2*x : -2*x-1`，例如 `[0,-1,1,-2,2] → [0,1,2,3,4]`。
`INT64_MIN → UINT64_MAX`，`INT64_MAX → UINT64_MAX-1`；独立逆变换完整恢复整个int64域。
用uint64存储输出，不能再解释成int64，否则大值会显示为负数。

## 最小用法：默认只做 ZigZag

```sh
python3 scripts/pure_numeric_operator_runtime_v1.py build
python3 scripts/pure_numeric_operator_runtime_v1.py transform --input signed.i64 --output nonnegative.u64
python3 scripts/pure_numeric_operator_runtime_v1.py transform --reverse --input nonnegative.u64 --output restored.i64
```

上述两个 transform 调用**不会自动调用compact、zlib或frame**，输入和输出长度完全相同。
也可以直接调用新增 C++ CLI，它从stdin读取、向stdout写出原始字节：

```sh
cpp/build/pure_numeric_v1/pure_numeric_operator_v1 zigzag-forward < signed.i64 > nonnegative.u64
cpp/build/pure_numeric_v1/pure_numeric_operator_v1 zigzag-inverse < nonnegative.u64 > restored.i64
```

只有显式列出组合才会压缩；只有加 `--framed` 才会加外层frame：

```sh
python3 scripts/pure_numeric_operator_runtime_v1.py pipeline --input-kind int64 --actions zigzag,compact_uint64,zlib6 --framed --input signed.i64 --output compressed.pnv1
python3 scripts/pure_numeric_operator_runtime_v1.py decode-frame --input compressed.pnv1 --output restored.i64
python3 scripts/test_pure_numeric_operator_runtime_v1.py
```

Python API 可分别调用 `zigzag_forward`、`zigzag_inverse`、`run_action("compact_uint64", ...)`、
`run_action("zlib6", ...)`、`frame_wrap`。每个步骤的输入输出类型均检查，不能将int64直接交给uint64打包器。
`registry` 子命令给出机器可读schema；编译清单位于新目录 `cpp/build/pure_numeric_v1/build_manifest.json`。

## 与已有实验和本次PPT的关系

本次是算子实现拆分和小型无损测试，**没有重跑RL**。旧 depth10、Beam29、PPO、GRPO数据均仍是旧版本实测，
不可将旧 `ZigZag-reference[11]` 压缩比归给新的纯映射。
旧环境仅支持int64/float64/bytes，且将首两个reference步骤重新合并执行。要让纯uint64步骤进入RL，
下一轮必须版本化迁移typed环境、合法类型转移、序列封装与长度计费，并在相同预算重新测量所有对照。

新 `compact_uint64` 是明确的无符号1/2/4/8字节打包，不等于旧 signed compact/OPA1布局；
因此显式组合的最终字节数不保证与旧ID11逐字节一致，不能将它们混入同一结果版本。
