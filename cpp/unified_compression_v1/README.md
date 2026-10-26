# 统一数值列压缩接口 v1

当前接口支持算法/算子名称、30个组合名称、显式算子树，以及推荐—实测—反馈预算闭环。执行与计时均为C++20。新增契约、示例、预算语义和192隔离验证见 [TREE_SELECTION.zh.md](TREE_SELECTION.zh.md)。普通 `compress` 仍执行指定方法，不隐式选型。

新增BOS实现及兼容、审计规则见 [BOS_REPLACEMENT.zh.md](BOS_REPLACEMENT.zh.md)。

## C++ 调用

```cpp
#include "api.hpp"

using namespace unified_compression_v1;
std::vector<double> values{1.0, 1.5, 2.0, 3.0};
auto result = compress_column(std::span<const double>(values), "Zstd");

// result.compressed_data       std::vector<uint8_t>，完整 UCC1 二进制
// result.compression_ratio     std::optional<double>
// result.compression_time_ns   uint64_t
// result.decompression_time_ns uint64_t
auto restored = decompress_column(result.compressed_data);
```

同样支持 `span<const int32_t>`、`span<const int64_t>`、`span<const float>`。原始 dtype 保留，不先转为 double；NaN 的 payload、正负零和 int64 边界都按原位字节恢复。

已有 compact 数值列可直接传 `Column`。它复用现有 C++ `native_baseline86_v1::Block`，无需在算法适配层复制整列：

```cpp
Column column{
  .dtype = "int64",
  .row_count = 3,
  .values = /* 两个非空 int64 的 little-endian 字节，按原行顺序 */ {},
  .missing = {0x02} // 第2行是 NULL；values 中省略这一行
};
// 填好 column.values 后：
Registry registry; // 常驻进程复用；只在初始化时加载算子目录
auto result = registry.compress(column, "algorithm:Zstd", {0, 3});
```

`Column` 必须满足：dtype 为 int32/int64/float32/float64，bitmap 长度为 `ceil(row_count/8)`，最低位对应第一行，置 1 表示 NULL，尾部未用 bit 为 0；`values.size() == 非空行数 × dtype宽度`。算法自身的 dtype/值域限制仍然生效。

## 名称、全部目录与输入输出

注册原规范目录的188项（86算法、102算子），另加30个固定组合，共 **218** 项。`methods()` 不删除禁用或缺依赖条目，逐项给出 `enabled`、`available`、原因、dtype、固定配置及原生 recipe/输入视图。成功执行必须同时满足当前构建可用、策略允许和本次输入域适用，注册本身不表示全域可执行。

- 算法：`"Zstd"`、`"gzip"`、`"LZ4"` 等，规范名称为 `"algorithm:Zstd"`。Zstd 的固定配置为 level 3，也接受 `"zstd3"`。同一个算法只暴露现有 baseline 固定 profile，配置记录在目录中；不会从输入、oracle 或历史最优参数中选 level。
- 算子：`"A013"`、`"A048"` 等，规范名称为 `"operator:A013"`。显示名称在没有歧义时也可用；若同名，要求使用规范名称。
- 名称忽略 ASCII 大小写及首尾空白。未知名称、歧义、禁用、缺依赖、dtype/值域不适用均抛出 `Error{code,message}`，不会替换为另一个算法。
- 102 项算子中 A106–A110 及 PPMd/DMC/PAQ8/CMIX/LSTM/NNCP 继续禁用；算法 PPMD8 继续禁用，GLZA 按当前 active policy 退役。没有覆盖开关。

所有按名称调用的方法的外部输入均为 `Column + string`，输出均为 `CompressionResult`；二进制均为可独立恢复原始 dtype、row count、数值字节和 null bitmap 的完整 UCC1 帧。算法内部复用 native baseline NGF1 帧；算子内部复用目录绑定的 NGF1 算子树。

算子的目录 `initial_kind / initial_width / input_view / recipe` 也完整保留在 inventory 的 `contract` 中。较宽源 dtype 可能按目录解释为原位视图或较窄 raw lane；只允许宽度整除的原位拆分，不进行数值转换、扩展或截断。A048 的输出仍是纯 typed-plane 多端口，所有端口及逆变换元数据都装入同一完整帧。

原子变换不一定减少字节。本接口不会在算子后面悄悄加 Zstd，也不把它当成已测的算子×算法联合最优；算子调用的大小就是它原有目录 recipe 加完整恢复封装的大小。目录既有的无损旁路仍可能执行，`bit_exact` 只表示本次逐位恢复成功，不能据此声称严格核心路径执行。需要研究比较时应继续区分严格路径与无损旁路。

完整机器目录和逐项输入输出表可由同一个可执行文件导出，避免维护另一份会漂移的名称表：

```sh
cpp/unified_compression_v1/build-local/unified_compression_cli inventory
cpp/unified_compression_v1/build-local/unified_compression_cli inventory --markdown
```

已导出的 [METHOD_IO.md](METHOD_IO.md) 含当前218项的输入、输出、固定配置与可用状态，并记录本机 macOS/arm64 构建的新BOS路由。其他机器应重新导出 inventory，不能直接沿用本机的依赖可用性结论；192本轮完整目录随部署验收记录保留。

## 压缩率与计时

| 字段 | 定义 |
|---|---|
| `compressed_data` | 完整 UCC1 帧，包括方法标识、NGF1、dtype/行数/null 信息及完整性校验；解码不依赖原列 |
| `original_numeric_bytes` | 原 dtype 的非空数值有效载荷字节数 |
| `compressed_bytes` | `compressed_data.size()`，不只算 codec payload |
| `compression_ratio` | `compressed_bytes / original_numeric_bytes`，越低越好，可大于 1；空列或全 NULL 为 `nullopt` / JSON `null` |
| `compression_time_ns` | 编码 API、输入检查、分配、native kernel、原生完整帧、统一封装和校验的实测耗时 |
| `decompression_time_ns` | UCC1 检查、方法绑定检查、原生帧解析、逆运算和恢复列结构的实测耗时 |
| `bit_exact` | 每次编码后，仅用结果帧解码，再与源列的 dtype、行数、数值及 bitmap 逐位比较通过 |

默认 0 次预热、1 次测量。指定多次重复时返回中位数（偶数次取排序后上中位）与全部原始 ns 样本。正确性比较、名称解析、registry 初始化、输入数组转成 `Column`、文件 I/O、JSON/Base64 和进程启动均不计入编解码时间。两个适配器均复用现有输入校验，不额外增加统一编码层的重复全列扫描；算法原生内部的检查仍按原实现执行。

本接口的计时是新统一 API 的完整调用边界，新增 UCC1 封装也计入；不能直接替代或覆盖冻结 oracle 中历史算法/算子的计时，尤其不能给原来为 null 的算子排名时间补值。小输入的时间容易受系统噪声影响，单次测量只作为调用结果；性能研究应增加重复并单独控制 CPU/线程环境。

## JSON / 二进制文件接口

`unified_compression_cli serve` 每行接收一个 JSON，请求间独立，错误后继续处理下一行：

```json
{"command":"compress","method":"Zstd","dtype":"float64","values":[1.0,1.5,null,3.0]}
```

返回 `compressed_base64` 和上述指标（另提供 ms 单位）。需要原始二进制文件时，增加绝对 `output_path`，输出写文件后响应给出路径、字节数和 SHA-256。文件写入在计时区外。

失败响应为 `{"status":"error","error":{"code":"...","message":"..."}}`，不返回伪造的压缩指标；JSONL 服务继续处理下一行。

```json
{"command":"compress","method":"A013","dtype":"int64","values":[10,11,12],"output_path":"/tmp/column.ucc1"}
```

单请求文件及独立解码：

```sh
unified_compression_cli compress request.json
unified_compression_cli decompress /tmp/column.ucc1
```

也可用 JSON 请求 `{"command":"decompress","compressed_base64":"..."}` 或 `frame_path`。恢复输出为 `dtype`、`row_count`、`values_le_hex` 与 `missing_bitmap_hex`，以免 JSON 浮点表示破坏 int64、NaN payload 或负零。输入也支持 `values_le_hex`、`row_count`、`missing_bitmap_hex` 通道。压缩请求不接受 `values` 与 raw-hex 通道同时出现。

独立 `decompress_column(frame)` 和 standalone CLI `decompress FRAME` 不构造推荐器或算子目录，不读取原数值文件或编码时的 catalog；只需要完整帧和同一可用 C++ kernel 实现。

## 独立构建与验证

```sh
cmake -S cpp/unified_compression_v1 -B cpp/unified_compression_v1/build-local \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/unified_compression_v1/build-local \
  --target unified_compression_cli unified_compression_api_test \
           unified_algorithm_adapter_test unified_operator_adapter_test \
           unified_tree_selection_test unified_recommender_test \
           bos_harness_integration_test -j2
ctest --test-dir cpp/unified_compression_v1/build-local -R '^(unified_|bos_harness)' \
  --output-on-failure
```

默认链接现有 residual 算子依赖；可用 `-DUNIFIED_COMPRESSION_ENABLE_OPERATOR_SUPPORT_DEPS=OFF` 构建较小闭包，缺依赖项保留在 inventory 并明确 unavailable。NNCP 在 CMake 中强制关闭。所有新代码及构建放在独立目录，不修改现有实验二进制，192验证使用独立目录和只读旧依赖；不改动训练或confirm组数据。

若 `external/compression/c-blosc2-3.3.3` 是历史空目录，CMake 优先复用项目留存的 pinned 3.3.3 源；也可指定 `-DOPERATOR102_BLOSC333_SOURCE=/absolute/source`。缺少本地源码时，仅在新构建目录下载[官方 v3.3.3 源码](https://github.com/Blosc/c-blosc2/releases/tag/v3.3.3)，校验 archive SHA-256 `3ab395116eae3ce0b7488e994224f85cbc858a7a71663af1f938cc5997a7dddd`。可用 `-DUNIFIED_COMPRESSION_FETCH_MISSING_DEPENDENCIES=OFF` 禁止此下载。

本版沿用底层 runtime 的资源限额（单 stream 默认不超过 256 MiB、完整帧约 1 GiB，另有工作内存和算法自身限额）；超限显式失败，不截断、抽样或只压缩部分输入。大于限额的列应先按应用的固定块规则分块调用。
