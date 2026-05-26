  ---
  Report: 评估“全流程只使用 NPU 格式”方案
  
  任务背景

  当前 RKNPU 路径已经把 weight 侧 的重排从运行时挪到了离线 GGUF 生成阶段，工具是 tools/rknpu-prepack/rknpu-prepack.cpp。
  因此运行时不再重排权重，而是直接使用离线 prepack 后的 payload DMA 地址。

  但 activation 侧 仍然在每次 matmul 前动态重排/量化，所以在 prefill 阶段，随着 M（token 数）变大，A 的重排成本线性放大，成为明显瓶颈。

  这次调研的核心问题是：

  ▎ 能否让整个计算流程尽量都工作在 NPU 友好的 packed format 上，从而减少甚至避免 matmul 前后的 activation 重排？

  ---
  结论先说
  
  结论 1：当前离线 prepack 只解决了 weight/RHS，没有解决 activation/LHS

  这是已确认事实，不是推测。

  - 离线工具只为权重生成：
    - <tensor>.__rknpu_meta
    - <tensor>.__rknpu_payload
    - 见 tools/rknpu-prepack/rknpu-prepack.cpp:372
  - 模型加载时也只把这两类 aux tensor 解析并注册为 offline prepack：
    - src/llama-model-loader.cpp:704
    - src/llama-model.cpp:4462
  - 真正的 activation 变换仍在运行时发生于：
    - ggml/src/ggml-rknpu-re.cpp:2183 (rknpu2_matmul_pre_scale)
    - ggml/src/ggml-rknpu-re.cpp:2269 (rknpu2_matmul_pre1)

  结论 2：你提出的“全流程 NPU format”在理论上可做，但不是小改动

  阻力不在“再 prepack 一些权重”，而在于：

  - 当前 matmul 之外的大量算子 都默认活在普通 ggml layout / contiguous F32 world 里；
  - RKNPU 路径只接管了 mul_mat，其余算子仍按 CPU/普通 ggml tensor 语义工作。

  所以如果真要“全流程 packed”，本质上要做的是：

  1. 重新定义一套 packed activation 表示；
  2. 让大量非 matmul op 也能读写这种 packed 表示，或把更多子图整体搬进 NPU 域；
  3. 否则仍会在 matmul 前后不断发生 packed <-> normal 的来回转换。

  结论 3：完全 end-to-end packed 不适合作为第一步；更现实的是做“matmul island 扩张”

  更现实的路线是：

  - 不是要求“整个模型所有 tensor 永远都用 NPU format”
  - 而是找出连续 matmul 密集段，尽量延长 packed activation 的生命周期，减少中间 unpack/repack 次数

  但即便如此：
  - Attention 路径很难，因为 rope / softmax / permute / cont 很快就把你拉回普通 layout；
  - FFN 路径相对更有机会，但也仍有 add / mul / silu / gelu 等中间算子需要 packed-aware 实现。

  ---
  1. 当前 RKNPU 调用流程
  
  1.1 模型侧：离线 prepack 数据如何进入运行时

  离线生成

  - prepack 工具：tools/rknpu-prepack/rknpu-prepack.cpp:148
  - 目标张量筛选规则：ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:193
  - 输出两个 aux tensor：
    - .__rknpu_meta
    - .__rknpu_payload
    - 见 tools/rknpu-prepack/rknpu-prepack.cpp:372

  加载解析

  - loader 识别 __rknpu_meta / __rknpu_payload：src/llama-model-loader.cpp:713
  - 将它们配对并解析 header：src/llama-model-loader.cpp:748
  - 如果 canonical CPU weight 已删除，则 synthesize 一个逻辑上的 canonical tensor：
    - src/llama-model-loader.cpp:783

  分配与注册

  - src/llama-model.cpp:4274
    - meta tensor 进 CPU buffer
    - payload tensor 进 HOST_DMA buffer
  - 数据 load 完成后，注册 offline prepack：
    - src/llama-model.cpp:4462
    - ggml/src/ggml-rknpu-re.cpp:1666 (ggml_rknpu2_register_offline_prepack)

  ---
  1.2 执行侧：RKNPU 不是独立 backend 调度，而是 CPU mul_mat 路径直接调用
  
  这点非常关键。

  backend 注册被刻意禁用了

  - ggml/src/ggml-backend-reg.cpp:198
  // DISABLED: Using NPU directly in CPU compute functions instead of as a separate backend
  // register_backend(ggml_backend_rknpu2_reg());

  真正的执行入口在 CPU mul_mat

  - ggml/src/ggml-cpu/ggml-cpu.c:1270
  - ggml/src/ggml-cpu/ggml-cpu.c:1292

  当前 ggml_compute_forward_mul_mat() 里如果 ggml_backend_rknpure_supports_op_out(dst) 成立，就直接走：

  1. rknpu2_matmul_pre0
  2. rknpu2_matmul_pre_scale
  3. rknpu2_matmul_pre1
  4. rknpu2_matmul_submit
  5. rknpu2_matmul_post

  对应：
  - ggml/src/ggml-cpu/ggml-cpu.c:1292-1305
  
  也就是说：

  ▎ 当前架构不是“图调度到 RKNPU backend”，而是“CPU mul_mat 内部把一部分 matmul 偷换成 NPU 实现”。

  ---
  1.3 RKNPU matmul 的 5 个阶段
  
  pre0

  - ggml/src/ggml-rknpu-re.cpp:2083

  做的事：
  - 校验输入约束；
  - 创建/查找 matmul kernel；
  - 清空输出；
  - 查找或构建离线 weight prepack cache。

  关键约束：
  - src0 contiguous
  - src1 contiguous
  - dst contiguous
  - src1->type == GGML_TYPE_F32
  - dst->type == GGML_TYPE_F32
  - src0->type 只支持 F16 / Q8_0

  都在：
  - ggml/src/ggml-rknpu-re.cpp:2090-2137
  
  这已经说明：

  ▎ 当前 NPU 路径入口假设 activation 是普通 contiguous F32，不是 packed activation tensor。

  pre_scale

  - ggml/src/ggml-rknpu-re.cpp:2183

  仅对 INT8 路径重要：
  - 对 activation tile 求 max_abs，得到 input scale：2220-2234
  - 读取离线 weight block scale：2237-2247

  pre1

  - ggml/src/ggml-rknpu-re.cpp:2269

  这是 activation 重排/量化的真正热点。

  FP16 路径

  - ggml/src/ggml-rknpu-re.cpp:2289-2307

  把 F32 activation A[ii * k + jj + t] 打包进 NPU 输入 buffer。

  INT8 路径

  - ggml/src/ggml-rknpu-re.cpp:2308-2323

  把 F32 activation 量化成 int8，并按 NPU 需要的 tile 格式写入输入 buffer。

  同时 weight 侧并不再复制 payload，而是直接把每个 block 的 DMA 地址 patch 给任务：
  - ggml/src/ggml-rknpu-re.cpp:2327-2350
  
  submit

  - ggml/src/ggml-rknpu-re.cpp:2357

  post

  - ggml/src/ggml-rknpu-re.cpp:2385

  从 NPU 输出格式读回，并恢复/累加到普通 F32 dst。

  ---
  2. 当前 NPU 排布方式
  
  2.1 weight 离线排布

  公共 header:
  - ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:34
  
  block 规则

  - K：对齐到 32，最大 4096
  - N：对齐到 32，最大 4096，并且还受 3 核切分约束
  - 见 ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:72-85

  FP16 权重排布

  - 公式：ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:56-62
  - 注释说明的 native 微块布局是：
    - (N/16, K/32, 16, 32)
  - 佐证：ggml/src/ggml-rknpu-re.cpp:717-741

  INT8 权重排布

  - 公式：ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:64-70
  - native 微块布局：
    - (N/32, K/32, 32, 32)
  - 佐证：ggml/src/ggml-rknpu-re.cpp:770-798

  元数据

  离线 header 记录：
  - orig_type
  - logical k, n
  - packed block K, N
  - block_count
  - weight_bytes_per_block
  - scale_type
  - scales_bytes_total
  - packed_bytes_total

  见：
  - ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:34-48
  
  ---
  2.2 activation 运行时排布
  
  注意：activation 没有持久化格式，只有运行时临时 packed format。

  FP16 路径

  - ggml/src/ggml-rknpu-re.cpp:2289-2307

  写法等价于把 activation A 打成：
  - (K/8, M, 8)
  
  INT8 路径

  - ggml/src/ggml-rknpu-re.cpp:2308-2323

  写法等价于：
  - (K/16, M, 16)
  
  并且 activation scale 是按 tile 动态算出来的：
  - ggml/src/ggml-rknpu-re.cpp:2220-2234
  
  输出排布

  post() 读取时按 4 通道一组展开，等价于：
  - (N/4, M, 4)
  
  参见：
  - ggml/src/ggml-rknpu-re.cpp:2427-2455

  ---
  3. 为什么 prefill 慢
  
  这部分和你的观察完全一致。

  - weight prepack 已经离线化，所以 weight 不再 runtime 重排；
  - 但 activation 仍然在每次 matmul 前：
    - 逐 tile 重排
    - 逐 tile 量化（INT8）

  而这两段复杂度都随 M * K 增长。

  所以：
  - decode 时 M = 1，问题不大；
  - prefill 时 M 大很多，重排成本被放大。

  ---
  4. “全流程 NPU format” 的可行性分析


  -K输出mdst：普通 contiguous F32
  - 只有 weight 是离线 packed
  见：

  - ggml/src/ggml-rknpu-re.cpp:2090-2116
  4.1 为什么不是简单地“多 prepack 一些权重”

  真正的 blocker 不是 weight，而是 activation 和中间算子。

  RKNPU matmul 当前的契约是：
  - 输入 activation：普通 contiguous F32
  - 输出 dst：普通 contiguous F32
  - 只有 weight 是离线 packed

  见：
  - ggml/src/ggml-rknpu-re.cpp:2090-2116

  因此，如果你想全流程 packed，不能只扩展 tools/rknpu-prepack。
  必须解决：

  1. packed activation 如何在 ggml graph 中被表达？
  2. packed activation 如何被非 matmul op 消费？
  3. packed output 是否还要立即回写成 F32？

  ---
  4.2 现有图里，matmul 之间夹着很多普通 ggml op
  
  norm / add / mul

  - src/llama-graph.cpp:657-680

  FFN

  - src/llama-graph.cpp:701-756
  - up matmul -> add/mul -> silu/gelu -> ...

  attention

  - src/llama-graph.cpp:1248-1291

  典型链路是：
  - kq = ggml_mul_mat(...)
  - kq = ggml_add(...)
  - kq = ggml_soft_max_ext(...)
  - kqv = ggml_mul_mat(...)
  - ggml_permute(...)
  - ggml_cont_2d(...)

  也就是说：

  ▎ matmul 输出并不会立刻进入下一个 matmul，中间经常要过 softmax / rope / norm / add / silu / permute / cont 等操作。

  ---
  4.3 这些 op 当前都还是 CPU/普通 ggml layout 语义
  
  在 CPU backend 的 op dispatch 里，RKNPU 特判只出现在 MUL_MAT。
  而 RMS_NORM / SOFT_MAX / ROPE 等仍然是普通 CPU 路径：

  - ggml/src/ggml-cpu/ggml-cpu.c:1981
  - ggml/src/ggml-cpu/ggml-cpu.c:1997
  - ggml/src/ggml-cpu/ggml-cpu.c:2061
  - ggml/src/ggml-cpu/ggml-cpu.c:2069

  这说明当前体系里：

  ▎ 只有 matmul 被 RKNPU 化了，其他 op 没有 packed-activation 语义。

  ---
  4.4 因此完整方案的真实含义是什么
  
  如果真的想“全流程只使用 NPU 格式”，至少要选一种：

  方案 A：给大量 CPU op 增加 packed-activation 版本

  需要新增/改造：
  - RMSNorm
  - Rope
  - Softmax
  - Add / Mul / Scale
  - SiLU / GELU
  - Permute / Cont / Reshape 相关逻辑
  - 以及它们对应的 scheduler / tensor layout 认知

  这已经接近：

  ▎ 给 ggml 再引入一套新的 activation layout 生态

  这是大改。

  方案 B：把更大的子图整体搬进 NPU 域

  例如不只是 mul_mat，而是尝试把：
  - QK^T
  - softmax
  - V * softmax
  - FFN 的中间激活链

  也做成 NPU 子图。

  但这已经不是“复用现有 RKNPU matmul path”了，而是接近重新做 backend/subgraph lowering。

  也是大改。

  方案 C：局部延长 packed activation 生命周期

  这是最有现实意义的方向。

  例如：
  - 尝试在某些连续 matmul 之间不立刻 unpack；
  - 或者只为 FFN 某一段加 packed-aware elementwise op；
  - 或者只优化 prefill 时 activation packing/quantization 的开销。

  这是中等到大改，但远比 A/B 现实。

  ---
  5. 如果坚持做，全流程 packed 还需要哪些“额外权重/排布”
  
  这里要分清：

  5.1 已经覆盖的

  所有大多数 .weight 型 matmul 权重，其实已经可以离线 prepack 了：
  - 规则见 ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:193-232
  
  所以：
  - Q / K / V / O projection
  - FFN up / gate / down
  - 等主 matmul 权重
  本质上已经是这套体系的适配对象。

  5.2 没覆盖、但也未必该“prepack”的

  norm 权重 / bias / scale

  例如：
  - RMSNorm 权重
  - 各种 bias / scale

  这些往往不是“大矩阵乘权重”，而是广播向量或 elementwise 参数。
  它们的问题通常不是“也去做 NPU matmul prepack”，而是：

  ▎ 如果 activation 改成 packed layout，norm/add/mul kernel 该如何按 packed 方式访问这些参数。

  也就是说，这里更需要的是 packed-aware kernel，不是再生成一份 __rknpu_payload。

  5.3 rope 因子 / mask / softmax 中间量

  这些更不是“weight prepack”问题，而是：
  - rope 是否能直接在 packed Q/K 上做
  - softmax 是否能直接在 packed logits 上做

  如果不能，还是得 unpack。

  ---
  6. 新 agent 应该优先得出的判断
  
  推荐判断

  “完全全流程 packed”：

  理论可行，工程上高风险，不适合直接作为第一阶段目标。

  更现实的目标：

  以 prefill 为中心，减少 activation 运行时变换成本。

  优先级建议：

  1. 先确认 activation packing 是否就是 prefill 瓶颈主因
    - 已有证据很强，重点看 pre_scale / pre1
    - ggml/src/ggml-rknpu-re.cpp:2183
    - ggml/src/ggml-rknpu-re.cpp:2269
  2. 优先探索“局部 packed 生命周期延长”而不是“全图 packed”
    - 例如 FFN 某段
    - 或只在 decode/prefill 不同 path 上做不同策略
  3. 如果要继续扩大范围，先挑 FFN，不要先挑 attention
    - attention 中间夹着 softmax / rope / permute / cont
    - 比 FFN 更难保住 packed activation

  ---
  7. 建议新 agent 的具体调研/实现方向
  
  方向 1：先做一份“activation packing 成本剖析”

  重点函数：
  - ggml/src/ggml-rknpu-re.cpp:2183 (rknpu2_matmul_pre_scale)
  - ggml/src/ggml-rknpu-re.cpp:2269 (rknpu2_matmul_pre1)

  要回答：
  - prefill 时耗时到底更多在 scale 统计，还是在 pack/quantize？
  - FP16 path 和 INT8 path 哪个更重？
  - 是否能 fuse pre_scale + pre1？

  方向 2：评估“packed activation 只在 FFN 内短生命周期存在”

  从：
  - src/llama-graph.cpp:701-756
  开始看

  问题是：
  - up/gate matmul 输出后，add / mul / silu 能否改为 packed-aware？
  - 如果能，是否能一路撑到 down 前再解包？

  方向 3：明确需要新增的 tensor/layout abstraction

  因为当前入口硬编码要求：
  - src1 contiguous F32
  - dst contiguous F32

  见：
  - ggml/src/ggml-rknpu-re.cpp:2090-2116
  
  所以如果要保留 packed activation，至少要决定：
  - 是新增一种 ggml tensor type / layout tag？
  - 还是只在 RKNPU 私有 side buffer 里维护 packed activation，并在 graph 边界显式转化？

  ---
  8. 关键文件列表
  
  运行流程

  - src/llama-model-loader.cpp:704
  - src/llama-model.cpp:4274
  - src/llama-model.cpp:4462
  - ggml/src/ggml-backend-reg.cpp:198
  - ggml/src/ggml-cpu/ggml-cpu.c:1292

  RKNPU layout / runtime transform

  - tools/rknpu-prepack/rknpu-prepack.cpp:148
  - ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:34
  - ggml/src/ggml-rknpu-re.cpp:2083
  - ggml/src/ggml-rknpu-re.cpp:2183
  - ggml/src/ggml-rknpu-re.cpp:2269
  - ggml/src/ggml-rknpu-re.cpp:2385

  图结构 / 非 matmul 阻碍

  - src/llama-graph.cpp:657
  - src/llama-graph.cpp:701
  - src/llama-graph.cpp:1248
  - ggml/src/ggml-cpu/ggml-cpu.c:1981
  - ggml/src/ggml-cpu/ggml-cpu.c:2061
  - ggml/src/ggml-cpu/ggml-cpu.c:2069
  - ggml/src/ggml-cpu/ggml-cpu.c:1292

  RKNPU layout / runtime transform

  - tools/rknpu-prepack/rknpu-prepack.cpp:148
  - ggml/src/ggml-rknpu_re/rknpu-prepack-common.h:34
  - ggml/src/ggml-rknpu-re.cpp:2083
  - ggml/src/ggml-rknpu-re.cpp:2183
  - ggml/src/ggml-rknpu-re.cpp:2269
  - ggml/src/ggml-rknpu-re.cpp:2385

  图结构 / 非 matmul 阻碍

  - src/llama-graph.cpp:657
  - src/llama-graph.cpp:701
  - src/llama-graph.cpp:1248
  - ggml/src/ggml-cpu/ggml-cpu.c:1981
  - ggml/src/ggml-cpu/ggml-cpu.c:2061
  - ggml/src/ggml-cpu/ggml-cpu.c:2069

  ---
  最终一句话结论

  当前设计已经把 weight-side prepack 做到了离线化；真正拖慢 prefill 的是 activation-side runtime packing。
  如果目标是“彻底不再重排 activation”，那不是多 prepack 一些权重就能解决，而是需要让一大批非 matmul op 也理解 packed
  activation。这个方向能做，但属于较大的执行模型改造；更现实的第一步是先做“局部 packed 生命周期延长”或直接优化 pre_scale/pre1。

  如果你要，我下一条可以把这份 report 再压缩成一个更像“给新 agent 的 prompt”的版本。