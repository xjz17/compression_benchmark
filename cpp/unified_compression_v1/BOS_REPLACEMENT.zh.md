# 当前 harness 的 BOS 实现替换

`BOS` / `algorithm:BOS` 保持同一公共名称，输入输出接口不变：完整UCC1二进制、完整字节比、压缩与解压ns、逐位恢复标记和执行审计。

- int64：使用隔离命名空间的 Project-150 BOS2，配置 `delta_zigzag_split_partition_for_bitpack_v1`；这是新实现的整数内核，算法体没有改动。它替代当前旧 `project_bos64_adapter` 编码路径。
- float32 / float64：使用 `bos-bfe-ngf-hybrid-v2`。先在不超过10000有效值的分块上扫描并取得精确整列最大精度P；每块共享该P，实际构建 DSC3+BOS2 和 BFE1+BOS2 两个完整NGF，严格取更小者，平局取DSC3。BFE自身精度pilot选择规则保持原实现。
- 浮点整列超过10000有效值：完整NGF成员存于带CRC的BHC1容器，外层仍是UCC1；NULL位图和IEEE位模式全部保留。
- int32：不做隐式扩宽，本BOS profile仍明确拒绝。

整数、浮点帧身份与已独立验证的 BOS v2 部署一致。旧BOS64整数UCC1/NGF仍通过原native baseline解码；旧baseline没有浮点BOS可执行路径。新bridge不会接管其他算法、算子、固定30配方或冻结oracle。

## 审计和选型

编码计时包含精度扫描、BFE内部选择、两个完整候选的真实编码、BHC1/UCC1封装和审计。解码计时仅包含已选帧的恢复及检查。它与150集成对实验、旧UCC1部署保留各自边界，不能直接替换历史时间。

`execution_audit` 返回 `implementation_route/selected_frontend/block_count`、BOS2/BFE/DSC3块数、`native_raw_block_count` 和 `harness_raw_fallback_stage_count`。BFE原生Raw是合法无损表示，公开压缩正常返回；为保持本harness的严格研究准入规则，该路径标记为 `native_raw_selected`、`strict=false`，不进入严格候选排名。它不等于输入域拒绝后的harness旁路；后者单独记为0。DSC3是完整编码的候选，不能因为选中它就称发生了raw旁路。

已有固定30候选池保持原样。使用新BOS参与预算闭环时，显式把 `algorithm:BOS` 放进 `selection.candidates`；冻结24候选推荐模型不会自动变成训练过BOS的新模型。

```json
{
  "command":"select", "dtype":"float64", "values":[1.25,1.5,null,1.75],
  "selection":{
    "objective":"compressed_bytes", "budget":{"max_trials":3},
    "candidates":[
      {"id":"BOS","method":"algorithm:BOS"},
      {"id":"Zstd3","method":"combo_v1:zstd3"},
      {"id":"LZ4","method":"combo_v1:lz4"}
    ]
  }
}
```

## 实现和验证

版本绑定及上游SHA在 `bos_bfe_v2/README.md`。实现复用已审查算法体，不编译第二份通用baseline，也不覆盖旧依赖；隔离BOS2 namespace避免与旧BOS算子符号冲突。注册表仍为86算法+102算子+30组合，共218项。

`bos_harness_integration_test` 覆盖新BOS2、旧整数冷解码、IEEE边界与NULL、空/全NULL、25013行多块、篡改拒绝、执行审计、选择闭环及禁用项。`--write-fixtures DIRECTORY` 原生导出10组CLI输入，用于跨评估环境完整UCC1字节和独立冷解码对照。

所有独立部署在独立目录构建；新版本验收后才更新 `unified_tree_loop_current` 指针。旧218部署、独立BOS v2部署、运行中实验和冻结135/3519参考结果保留，不重跑150数据集大实验。NNCP及其他排除项继续禁用。
