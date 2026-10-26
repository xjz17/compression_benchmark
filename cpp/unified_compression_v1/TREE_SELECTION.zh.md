# 统一组合、显式算子树与预算闭环

当前公共 Registry 注册218项：86算法、102算子、30固定组合。前两项组合基线为 `combo_v1:zstd3` / `combo_v1:lz4`，其余保留 `combo_v1:...`。裸 `Zstd` / `LZ4` 别名仍指原算法路由；两种路由的封装不同，字节结果不能混用。

## 显式树

```cpp
Registry registry;
auto fixed = registry.compress(column, "combo_v1:dictionary_bitpack_zstd3", {1, 3});
auto explicit_tree = registry.compress_tree(column, tree_json, {1, 3});
```

`compress` JSON 必须且只能提供 `method` 或 `tree`。例如：

```json
{
  "command":"compress", "dtype":"int64", "values":[10,11,10,null,12],
  "tree":{
    "schema":"ucc-operator-tree-v1", "source":{"root_kind":"numeric"},
    "nodes":[
      {"id":"dict","op":"dictionary","inputs":["values"],"parameters":{}},
      {"id":"dictionary_codec","op":"lz4","inputs":["dict:0"],"parameters":{}},
      {"id":"ids_pack","op":"bitpack","inputs":["dict:1"],"parameters":{"alignment_bytes":1,"bit_order":"msb","minimum_width":0,"wide_cutoff":64}},
      {"id":"ids_codec","op":"zstd3","inputs":["ids_pack:0"],"parameters":{}}
    ]
  },
  "warmups":1, "repeats":3, "output_path":"/tmp/explicit-tree.ucc1"
}
```

节点引用 `values` 或此前节点的 `id:port`；流最多消费一次，未消费输出由隐式 STOP 保留。语义白名单是 `zstd3/lz4/columnize/xor_carrier/xor_semantic/for/bitpack/dictionary/a048_planes/float_deconstruct`。profile/action/policy ID 内部固定，不接受注入。每个节点的parameters必须完整等于 `tree_contract.operators` 中对应的固定对象；例如columnize需要mode=transpose，不能省略为空对象。类型、端口、输入域、未知字段均检查。

`tree_contract` 命令和 inventory 导出完整机器契约。最多48节点、64KiB JSON、16层JSON嵌套；执行层另限8层深度、32活跃流、256MiB单流、512MiB工作内存、4MiB metadata、1GiB内层帧。HCB2 binding 在 NGF1 内携带规范树、SHA、dtype、形状和NULL摘要；解码前重编配方并比较实际节点，再运行逆变换。旧算法/算子UCC1路由与fixed30/HCB1兼容，新组合/树使用UCC1 kind=3/4。旧版decoder会拒绝新路由。

## 预算闭环

`selection.hpp` 提供 `select(registry, column, options)`。C++ `Proposer` 回调接收此前全部 `TrialFeedback`，返回下一项注册方法或显式树；也可提供静态候选列表。默认在fixed30适用子集中先测Zstd-3/LZ4，再按反馈选择后端和后续变换。

```json
{
  "command":"select", "dtype":"int64", "values":[10,11,10,null,12],
  "selection":{
    "objective":"compressed_bytes",
    "budget":{"max_trials":8,"soft_wall_time_ns":100000000},
    "constraints":{"max_compressed_bytes":100000,"max_decompression_time_ns":10000000},
    "measurement":{"warmups":1,"repeats":3}
  },
  "output_path":"/tmp/selected.ucc1"
}
```

目标：`compressed_bytes/compression_time/decompression_time/balanced`。约束：完整字节、压缩ns、解压ns上限。候选次数是硬预算，失败、重复、不支持的显式提案及基线均计费；默认策略先排除已知dtype/空输入限制。墙钟是软准入预算，到期不启动新试验，不中断在途codec。每次试验处理完整输入；本版没有采样推断整列成本。

响应 `trials` 逐轮列出提案、错误、严格审计、可行性和实测成本，另报停止原因、选型开销、探测及总墙钟。胜者完整帧直接复用，不再次压缩；输出临时文件原子改名提交。没有可行项返回 `no_feasible_candidate`，不写输出。公共接口校验逐位恢复；192验收额外用独立进程、缺失catalog的环境冷解码输出。

## 接入已有冻结模型

`recommender.hpp` 的 `PreparedRecommender` 读取既有冻结 `PreparedModel`，只传递旧24候选与fixed30语义交集的排名，新12组合没有因此成为已训练候选。调用CLI `recommend_select` 并增加 `model_path/split_path`；常驻进程缓存模型和SHA，验证development训练集合、frozen split和confirm禁令。

使用 `operator_prepared135.../models/MODEL_confirm.json`：名字表示服务于confirm验证，训练只用development。包含全部360历史样本的旧 `MODEL.json` 缺formal约束，会被拒绝。模型特征、预测、映射耗时计入软预算；首次读取/构建单独报告。基线仍实测并收费，之后优先模型seed，再按反馈探索。每轮结果反馈给提案策略，不在线修改模型参数。

所有排名必须逐位无损、审计明确strict且无旁路。公共压缩允许原目录旁路并报告。旧算法配置若含 tagged-raw/fallback、又没有逐块审计，则保守标 `adapter_bypass_unverified`、strict=false、fallback_used=null；可压缩但不进入严格排名。多轮返回全部审计，完整帧字节数须一致，任一轮非严格都阻止排名。

本版不拟合、不调参、不读取oracle标签。结果限已试候选与实测约束，不是135/3519完整推荐质量结论，也不是85算法×97算子的联合最优。冻结oracle的算法、公共Zstd3后端算子、已观测组合三类目标及历史计时边界继续独立保留。

## 192隔离验证

`scripts/run_unified_tree_loop_192_v1.py` 使用独立source/build/result目录，复用20261022已部署静态依赖，不重建或覆盖旧runtime。单worker、CPU62、SCHED_IDLE、nice19、idle I/O、2GiB地址空间、无GPU；前后核对旧二进制、依赖、catalog及仍存活进程的调度设置。NNCP和其他排除家族前向/逆向禁用；不停止、重启或调度现有实验。
