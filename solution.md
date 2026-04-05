# RKNPU 权重预处理前移到 GGUF 的细化方案（方案 A）

## 1. 目标与约束

### 1.1 本轮目标

基于当前仓库中的 RKNPU 运行时预打包缓存实现，将“静态权重的分块、重排、scale 计算”继续前移到 GGUF 侧完成，运行时只负责：

1. 识别 GGUF 中的 RKNPU 辅助 tensor。
2. 将其中的离线预处理结果构建为 `rknpu_weight_prepack_cache` 的 DMA 镜像。
3. 在 `pre_scale` / `pre1` / `submit` 阶段直接复用这些 DMA 块。
4. 在离线结果不可用时，自动回退到当前运行时预打包路径。

### 1.2 明确采用的路线

本方案明确采用：

- **方案 A：主 GGUF 内嵌 RKNPU auxiliary tensors**

即：

- 原始 CPU/ggml 权重 tensor 继续保留。
- 在同一个 GGUF 中新增 RKNPU 专用辅助 tensor。
- 这些辅助 tensor **不参与普通 llama 主图 tensor 映射**，而是由 loader 单独识别、登记、消费。

### 1.3 明确不做

本轮不做以下内容：

1. 不删除原 CPU 布局权重。
2. 不修改 CPU 路径主语义。
3. 不一步做到“单份权重 + 零冗余 DMA 常驻”。
4. 不立即扩展到所有量化类型与所有算子。

---

## 2. 当前代码现状与关键限制

## 2.1 已存在的运行时预打包能力

当前 `ggml/src/ggml-rknpu-re.cpp` 中已经具备完整的运行时 weight prepack 基础设施：

- `ggml_rknpu2_get_weight_prepack(...)`
  - 按 `K/N` 分块遍历权重。
  - 为每个 block 生成 NPU 所需布局。
  - `INT8` 路径会反量化 + 计算 block scale + 重新量化。
  - 每个 block 最终落在 `rknn_mem` DMA 内存里。
- `rknpu2_matmul_pre_scale(...)`
  - 已能从 cache 中直接读取 block scale。
- `rknpu2_matmul_pre1(...)`
  - 已能把 block 的 `dma_mem->dma` 绑定到 task。

结论：**现有 submit 消费链路已经足够，重点不是改 submit，而是替换 cache 的来源。**

## 2.2 loader 的两个硬约束

### 约束 A：GGUF tensor 数量会被严格统计

`src/llama-model-loader.cpp:814-816` 中，`done_getting_tensors()` 会检查：

- 创建的 tensor 数量 `n_created`
- 是否等于 GGUF 中读取到的 tensor 数量 `n_tensors`

因此：

- 如果 GGUF 中额外增加了 RKNPU tensor，loader 必须显式把它们纳入“已处理对象”。
- 不能简单“多塞进去但不管”。

### 约束 B：未知 tensor 名不会自动安全跳过

`src/llama-model.cpp:1608-1610` 中，普通模型 tensor 会经过 `llm_tensor_info_for(...)` 映射。

如果名字不在主模型 tensor 命名体系里，会直接抛：

- `missing tensor info mapping for ...`

因此：

- RKNPU auxiliary tensor **不能走普通 llama tensor 创建路径**。
- 必须在 loader/模型构建阶段加一条专门分流逻辑。

## 2.3 量化校验的第三个约束

当前 loader 在启用 `check_tensors` 时会调用：

- `ggml_validate_row_data(cur->type, ...)`
  - 见 `src/llama-model-loader.cpp:888`, `1018`, `1043`, `1070`

这意味着：

- 如果把 RKNPU 专用打包数据伪装成 `GGML_TYPE_Q8_0`，它会按 **ggml 原生 Q8_0 语义**检查。
- 但你现在的 NPU 量化语义不是标准 Q8_0，而是“每个大块一个 scale + 对应块内 packed weight”。
- 因此这部分必须绕过原生 Q8_0 row validation。

结论：

> **RKNPU auxiliary tensor 不能声明成普通的 `GGML_TYPE_Q8_0` 语义 tensor 并走默认校验。**

---

## 3. 总体设计

### 3.1 总体原则

第一版采用“**GGUF 内嵌 auxiliary blob tensor + 运行时 DMA 镜像 cache**”方案：

1. GGUF 中为目标权重新增 RKNPU 专用辅助 tensor。
2. 该辅助 tensor 保存已按 NPU block 规则重排后的数据。
3. INT8 情况下，辅助 tensor 同时保存 block scale。
4. 运行时加载模型时，loader 识别这些辅助 tensor，并将其登记到 RKNPU registry。
5. 首次真正用到某个原始权重时，再惰性地把 auxiliary tensor 转成 DMA mirror cache。
6. 后续 token 直接复用已有 cache。

### 3.2 为什么本轮必须把 auxiliary tensor 视作“特殊对象”

因为它同时偏离了 llama.cpp 的两套默认假设：

1. **命名假设**：它不是主模型标准 tensor。
2. **量化假设**：它不是 ggml 标准 Q8_0 row format。

因此它必须具备：

- 自己的命名空间
- 自己的 loader 分流逻辑
- 自己的校验逻辑
- 自己的消费路径

---

## 4. GGUF 中到底放什么

## 4.1 采用“新增辅助 tensor”，不是原 tensor 的附加值

最终选择：

- **大块 NPU packed 数据放在新开的 auxiliary tensor 中**。
- **小元信息放 GGUF KV**。

不采用“把大 packed 数据塞进原 tensor 的附加值/KV”的原因：

1. packed 数据体积大，不适合 metadata 区。
2. 后续 mmap/offset/切片访问更不自然。
3. split gguf 场景下 tensor 路径更稳。

### 4.2 为什么不是两个 tensor（packed / scale）

你提的这个点很关键。

实际运行时：

- NPU 提交时 weight 和 scale 是分开的。

但存文件时：

- 更适合把它们合并成 **一个 RKNPU blob tensor**。

原因：

1. 一个原始权重只对应一个 auxiliary 产物，管理更简单。
2. 避免多一个辅助 tensor 继续增加 tensor 数量与 loader 复杂度。
3. scale 和 packed 数据本来就是同一份离线预处理产物的两个视图。
4. 运行时完全可以在构建 cache 时把它们拆开。

因此本方案采用：

- `原始 tensor` + `一个 RKNPU blob auxiliary tensor`

而不是：

- `原始 tensor` + `packed tensor` + `scale tensor`

---

## 5. RKNPU auxiliary tensor 的命名与识别

### 5.1 命名规则

对于原始 tensor：

- `blk.0.attn_q.weight`

新增一个辅助 tensor：

- `blk.0.attn_q.weight.__rknpu_blob`

说明：

- 后缀 `.__rknpu_blob` 明确表示：这不是普通 llama tensor。
- loader 可以通过这个后缀快速分流。

### 5.2 metadata key

建议至少定义以下全局 key：

- `rknpu.prepack.version` : `uint32`
- `rknpu.prepack.backend` : `string`，如 `rknpu2`
- `rknpu.prepack.format` : `string`，如 `blob-v1`

对于每个原始 tensor，再定义前缀型 key：

- `rknpu.tensor.<tensor_name>.enabled` : `bool`
- `rknpu.tensor.<tensor_name>.layout` : `string`，值如 `fp16` / `int8-block-scale`
- `rknpu.tensor.<tensor_name>.K` : `uint32`
- `rknpu.tensor.<tensor_name>.N` : `uint32`
- `rknpu.tensor.<tensor_name>.block_count` : `uint32`
- `rknpu.tensor.<tensor_name>.blob_tensor` : `string`
- `rknpu.tensor.<tensor_name>.weight_bytes_per_block` : `uint32`
- `rknpu.tensor.<tensor_name>.scale_type` : `string`，如 `f32` / `none`
- `rknpu.tensor.<tensor_name>.scale_count` : `uint32`

如果以后格式升级，再加：

- `rknpu.tensor.<tensor_name>.record_stride`
- `rknpu.tensor.<tensor_name>.flags`

---

## 6. auxiliary tensor 的数据类型与校验策略

## 6.1 不要把它声明成 `GGML_TYPE_Q8_0`

这点必须明确。

因为你的 NPU 格式虽然“也是 int8 + scale”，但语义上已经不是 ggml 原生 Q8_0：

- ggml Q8_0：固定小 block 粒度、固定 row layout
- 你的 NPU 格式：按更大的 `(N, K)` block 统一计算 scale，再按 NPU 原生布局重排

所以：

- **不能依赖 ggml 的 Q8_0 类型去表达这个辅助 tensor**
- **也不能让 loader 对它跑 `ggml_validate_row_data(GGML_TYPE_Q8_0, ...)`**

## 6.2 建议的数据类型

建议把 `.__rknpu_blob` 视作 **opaque blob tensor**，用字节语义保存。

最简单稳妥的办法：

- tensor type 使用 `GGML_TYPE_I8` 或 `GGML_TYPE_I16` / `GGML_TYPE_F16` 都不理想，因为会带来“按元素解释”的误导。
- 第一版更建议：**按 1D byte blob 的思路使用 `GGML_TYPE_I8` 表示原始字节流**。

也就是说：

- `ne[0] = blob_nbytes`
- `ne[1..] = 1`
- 类型仅表示“这个 tensor 是一段字节”，不表示量化语义

好处：

1. 避开 ggml Q8_0 校验。
2. 避开“这是不是一个合法量化 row”的问题。
3. 对 fp16/int8 混合封装格式都通用。
4. 运行时按自定义格式解析即可。

## 6.3 自定义校验

对于 `.__rknpu_blob`，不走 `ggml_validate_row_data(...)`，改为 RKNPU 自己的 blob 校验：

- version 是否支持
- tensor_name 是否能反查到原始 tensor
- `(k, n, K, N)` 是否一致
- `block_count` 是否正确
- `blob_nbytes` 是否与格式计算值一致
- 对于 int8，scale 区和 packed 权重区长度是否一致

---

## 7. blob 内部如何排布数据

这是本方案最关键的文件格式问题。

## 7.1 目标

你运行时实际消费的是：

- `weight_dma`
- `scale`

而 GGUF 里只保存一个 blob tensor。

因此 blob 需要满足：

1. 顺序可预测，运行时可按 block 直接定位。
2. scale 可快速读取。
3. packed weight 可快速 memcpy 到 DMA。
4. 不要重复存 `(nn, kk)` 等可推导信息。

## 7.2 不推荐的排布：每个 block 一个小 header

例如：

- `[scale][packed weight][scale][packed weight]...`

这种格式能用，但我不推荐第一版这么做，原因：

1. 每个 block 都带小 header，不够干净。
2. 后续如果 scale 类型升级，record layout 更容易碎。
3. 想批量扫 scale 或批量拷贝 packed 区时不够方便。

## 7.3 推荐排布：单 header + scales 区 + packed 区

建议 `blob-v1` 格式如下：

```text
[ blob_header | scales_region | packed_region ]
```

其中：

### `blob_header`

固定长度头部，包含：

- magic（例如 `RKP1`）
- version
- layout
- tensor_type（fp16 / int8-block-scale）
- k
- n
- K
- N
- block_count
- weight_bytes_per_block
- scale_type
- scale_bytes_total
- packed_bytes_total
- scales_offset
- packed_offset

### `scales_region`

- 对 fp16：长度为 0
- 对 int8：按 block 顺序保存所有 scale

第一版建议 scale 类型固定为：

- `float32`

即：

```text
scale[0], scale[1], ..., scale[block_count-1]
```

### `packed_region`

- 按 block 顺序连续保存所有 block 的 packed 权重

即：

```text
packed_block[0] | packed_block[1] | ... | packed_block[block_count-1]
```

其中每个 block 长度固定为：

- `weight_bytes_per_block`

## 7.4 block 顺序

block 顺序必须与当前运行时 `ggml_rknpu2_get_weight_prepack(...)` 一致。

建议固定为：

```text
for nn in [0, N, 2N, ...]
  for kk in [0, K, 2K, ...]
```

也就是当前代码已有的：

- 先外层 `nn`
- 再内层 `kk`

这样运行时可通过 block index 推回：

- `nn = (block_id / n_kk) * N`
- `kk = (block_id % n_kk) * K`

所以不需要在 blob 里重复存 `(nn, kk)`。

## 7.5 为什么这种排布最好

对你的场景，这个排布的优点是：

1. **一个 tensor 就能保存全部离线结果**。
2. **运行时可以先扫 scales，再按 block 定位 packed 权重**。
3. **没有 per-block 额外 header 开销**。
4. **scale 和 weight 在文件中逻辑分离，但物理仍属于同一产物**。
5. **后续如果要 mmap 直接读 packed 区，也更容易做 offset 计算**。

---

## 8. loader 侧如何绕过“tensor 数量”和“普通 tensor 路径”问题

## 8.1 loader 必须把 auxiliary tensor 作为“已处理对象”

因为 `done_getting_tensors()` 会检查总数一致，所以 loader 需要显式处理 `.__rknpu_blob`。

建议处理方式：

1. 在 `weights_map` 建立后，先扫描所有 tensor 名。
2. 若名字带 `.__rknpu_blob` 后缀：
   - 放入 `rknpu_aux_weights_map`
   - 不进入普通 llama tensor 构造逻辑
   - 但要记入“已处理数量”

也就是说，逻辑上分成两类：

- 普通模型 tensor
- RKNPU auxiliary tensor

两类都必须计入 loader 的“已消费”统计。

## 8.2 具体建议

在模型构建阶段，增加一条分流：

- 普通 tensor：继续走 `llm_tensor_info_for(...)`
- `.__rknpu_blob`：走 `register_rknpu_aux_tensor(...)`

`register_rknpu_aux_tensor(...)` 的职责：

1. 验证命名是否合法。
2. 反查其对应的原始 tensor 名。
3. 校验对应 metadata。
4. 把它登记到 RKNPU auxiliary registry。
5. 记为已处理，从而不触发 `n_created != n_tensors`。

注意：

- 它不是“unused tensor”
- 它是“auxiliary tensor”
- 因此不要混用现有 unused 逻辑

---

## 9. 运行时 registry 与挂载方式

## 9.1 registry 结构

建议新增：

```cpp
struct rknpu_tensor_offline_prepack_ref {
    uint32_t version;
    std::string format;       // blob-v1
    std::string layout;       // fp16 / int8-block-scale
    int64_t k;
    int64_t n;
    int K;
    int N;
    uint32_t block_count;
    uint32_t weight_bytes_per_block;

    const ggml_tensor * src_tensor;   // 原始 CPU tensor
    const ggml_tensor * blob_tensor;  // __rknpu_blob
};
```

## 9.2 挂载方式

建议采用：

- 全局 registry 拥有对象生命周期
- `src_tensor->extra` 挂轻量引用或句柄

原因：

1. `tensor->extra` 适合后端私有元信息。
2. registry 便于统一管理 auxiliary tensor 与原始 tensor 的映射。
3. 可以避免直接在 `extra` 里塞太重的对象。

---

## 10. 运行时如何从 blob 构建 DMA cache

## 10.1 改造原则

保留现有 `rknpu_weight_prepack_cache` 和 `pre_scale/pre1` 消费逻辑，只改 cache 的构建来源。

## 10.2 新的构建流程

当 `ggml_rknpu2_get_weight_prepack(src0, ...)` 被调用时：

1. 先看 `src0` 是否挂有 `rknpu_tensor_offline_prepack_ref`
2. 若有：
   - 解析 blob header
   - 校验 block_count / offsets / bytes
   - 对每个 block：
     - 读取 `scale[block_id]`（int8）
     - 定位 `packed_block_ptr`
     - 分配 `rknn_mem`
     - memcpy packed 权重到 DMA
     - 建立 `rknpu_weight_prepack_block { nn, kk, scale, dma_mem }`
3. 若没有或失败：
   - 回退到当前在线构建路径

## 10.3 为什么这里不需要 scale 单独成 tensor

因为运行时构建 cache 时，本来就要做一次“文件表示 -> 运行时对象”的转换。

所以：

- 文件里可以是单 blob
- 运行时拆成 `scale + dma_mem`

这正好匹配“文件侧一体化、运行时侧分离消费”的需求。

---

## 11. 需要绕过的两类检查

## 11.1 绕过普通 llama tensor 命名映射检查

必须让 `.__rknpu_blob` 不走：

- `llm_tensor_info_for(...)`

否则会在 `src/llama-model.cpp:1608-1610` 报错。

## 11.2 绕过 ggml 原生量化 row 校验

必须让 `.__rknpu_blob` 不走：

- `ggml_validate_row_data(cur->type, ...)`

否则你这种“每大块一个 scale”的 NPU 量化语义会被误判为非法 Q8_0 数据。

### 实现建议

对 auxiliary tensor 引入专门判断：

- 如果名字后缀是 `.__rknpu_blob`
- 则：
  - 不做 `ggml_validate_row_data`
  - 改做 `validate_rknpu_blob(...)`

---

## 12. 代码结构建议

## 12.1 `ggml/src/ggml-rknpu-re.cpp`

### [ ] 任务 A1：抽离纯 prepack 算法

从当前 `ggml_rknpu2_get_weight_prepack(...)` 中抽出：

- block 遍历
- fp16 pack 规则
- int8 pack + scale 规则

形成一个不依赖 `rknn_mem` 的 helper。

### [ ] 任务 A2：新增 blob 编解码 helper

建议增加：

- `rknpu_pack_blob_v1(...)`
- `rknpu_unpack_blob_header(...)`
- `rknpu_get_blob_scale_ptr(...)`
- `rknpu_get_blob_weight_ptr(...)`

### [ ] 任务 A3：新增“从 blob 构建 cache”的入口

建议加函数：

- `ggml_rknpu2_build_weight_prepack_from_blob(...)`
- `ggml_rknpu2_build_weight_prepack_from_tensor(...)`

前者：输入 `.__rknpu_blob` 描述，输出 DMA cache。
后者：保留现有在线 fallback 路径。

### [ ] 任务 A4：统一 `ggml_rknpu2_get_weight_prepack(...)` 调度逻辑

改成：

1. 先检查 `src0` 是否挂有 offline blob ref。
2. 若有，优先从 blob 构建 cache。
3. 若无，再走原始 tensor 在线预打包。
4. 若构建失败，返回 fallback 路径结果或 `nullptr`。

### [ ] 任务 A5：保留 `pre_scale/pre1` 消费逻辑

只调整内部查 cache 的来源，不改接口。

---

## 12.2 `src/llama-model-loader.cpp`

### [ ] 任务 B1：识别 auxiliary tensor

在 `weights_map` 建立后，扫描：

- `name.ends_with(".__rknpu_blob")`

并将其登记到 `rknpu_aux_weights_map`。

### [ ] 任务 B2：对 auxiliary tensor 单独计数与消费

让它们不进入普通 tensor 创建路径，但要记入“已处理对象”，避免触发：

- `wrong number of tensors; expected X, got Y`

### [ ] 任务 B3：自定义校验

对于 auxiliary tensor：

- 跳过 `ggml_validate_row_data`
- 使用 `validate_rknpu_blob(...)`

### [ ] 任务 B4：建立 `src_tensor -> blob_tensor` registry

根据 metadata 中的：

- `blob_tensor`
- `layout`
- `K/N`
- `block_count`

建立映射。

---

## 12.3 新增工具：`tools/rknpu-prepack/`

### [ ] 任务 C1：读取原始 GGUF

读取模型、定位目标 tensor。

### [ ] 任务 C2：生成 blob-v1

输出：

- blob header
- scales region
- packed region

### [ ] 任务 C3：写 auxiliary tensor + KV

对于每个原始 tensor：

- 新增 `__rknpu_blob`
- 写入相关 metadata

### [ ] 任务 C4：提供白名单机制

第一版建议工具支持：

- 全量处理
- 按 tensor name regex 处理
- 按层范围处理

---

## 13. 回退策略

### 13.1 必须支持的回退场景

以下任一情况触发回退到在线旧路径：

1. 没有 RKNPU metadata。
2. 找不到 `.__rknpu_blob`。
3. blob version 不匹配。
4. blob header 非法。
5. `k/n/K/N` 与运行时期望不匹配。
6. block_count 不匹配。
7. scale 区 / packed 区长度不匹配。
8. DMA 分配失败。
9. auxiliary tensor 尚未 ready。

### 13.2 回退原则

- 回退必须优先保证功能正确。
- 不能因为离线 blob 异常直接 assert 终止整个推理。
- assert 只保留给真正不应发生的内部不变量错误。

---

## 14. 风险点与应对

## 14.1 最大风险：离线与运行时规则漂移

应对：

1. pack 规则只保留一份实现。
2. 离线工具直接复用运行时同源算法。
3. blob header + metadata 中写版本号与块参数。

## 14.2 第二风险：内存冗余

第一版会同时存在：

- 原始 tensor
- auxiliary blob tensor
- DMA mirror

应对：

1. 第一版接受冗余。
2. 增加统计项。
3. 后续再考虑删原始 CPU 权重或惰性 DMA 分配。

## 14.3 第三风险：DMA 资源受限

应对：

1. 仍采用“首次真正使用时再构建 DMA cache”。
2. 不在模型加载时一次性全量灌入 DMA。
3. 后续必要时再引入 LRU/逐层驱逐。

---

## 15. 建议增加的统计项

建议增加：

- `offline_blob_tensor_count`
- `offline_blob_tensor_hit_count`
- `offline_blob_tensor_miss_count`
- `offline_blob_build_dma_count`
- `offline_blob_fallback_count`
- `offline_blob_version_mismatch_count`
- `offline_blob_validate_fail_count`
- `offline_blob_dma_fail_count`
- `offline_blob_bytes_total`
- `offline_blob_dma_bytes_total`

---

## 16. 我建议你本轮优先实现的 TODO

### [ ] TODO-1 抽出共用 prepack helper
从 `ggml_rknpu2_get_weight_prepack(...)` 中抽出纯算法层。

### [ ] TODO-2 定义 blob-v1 格式
固定：

- header 字段
- scales 区
- packed 区
- block 顺序

### [ ] TODO-3 定义 auxiliary tensor 命名和 metadata
固定：

- `.__rknpu_blob`
- `version/layout/K/N/block_count/blob_tensor`

### [ ] TODO-4 loader 分流 auxiliary tensor
让它们不走普通 llama tensor 映射，同时计入已处理数量。

### [ ] TODO-5 为 auxiliary tensor 加自定义校验
绕过 ggml 原生 Q8_0 row validation。

### [ ] TODO-6 `ggml_rknpu2_get_weight_prepack(...)` 优先走 blob
有离线 blob 时，从 blob 构建 DMA cache。

### [ ] TODO-7 跑 correctness / latency / memory 验证
形成对比数据，再决定下一阶段是否删 CPU 冗余权重。

---

## 17. 结论

本方案的关键结论有三个：

1. **RKNPU 重排结果应该放在新的 auxiliary tensor 里，而不是塞进原 tensor 的附加值。**
2. **这个 auxiliary tensor 不能走普通 llama tensor 路径，也不能按 ggml 原生 Q8_0 语义校验。**
3. **文件里用单 blob 保存 `scale + packed weight`，运行时再拆成 `scale + DMA weight`，是最自然也最稳的组织方式。**

这条路线最符合你当前代码状态：

- 复用现有 `rknpu_weight_prepack_cache` 消费链路
- 只替换生产来源
- 不破坏 CPU 路径
- 能有效规避 loader 计数、名字映射、量化校验这三类问题
