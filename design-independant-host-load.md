## 当前确认的下一阶段方案：NPU payload 连续数据段 + 独立 host reload

这一节用于记录我们当前已经达成共识的下一阶段实现方向。后续如果方案发生变化，应优先更新这里。

### 总体目标

下一阶段的目标是：

- 后续可以把**整个 GGUF（包括 metadata）都加密**；
- host 不再走 `llama.cpp` 的 GGUF parser / model loader 路径；
- host 只做最小知情的 **entry 级搬运**；
- guest 继续负责推理，以及未来的 metadata 解密。

换句话说，我们希望把 host 从“懂模型的 loader”降成“只懂 offset / entry 的搬运器”。

### 文件格式方向

新的目标产物仍然保留整体上类似：

- header
- metadata
- data

这样的布局，但会对 **NPU payload** 做专门处理。

当前共识是：

1. 给 host 暴露一个**极小的明文 header**；
2. 把所有 NPU payload 收拢到**文件数据区的最后面**；
3. 这个尾部区域在文件里必须是**连续的**；
4. 这个连续尾段里的第 `x` 个字节，应当对应 guest payload 池 offset `x` 的字节。

这样 host 拿到 reload 区间后，只需要：

- 从 `payload_start + reload_offset` 开始读文件；
- 把数据写到 guest payload 池 offset `reload_offset` 对应的位置；
- 按 `entry_size` 逐块推进。

host 不需要知道：

- tensor name
- tensor shape
- tensor type
- GGUF metadata 细节

### 明文 header 的当前最小集合

当前确认保留这 4 个字段：

- `payload_start`
- `payload_bytes`
- `entry_size`
- `compute_buffer_size`

其中：

- `payload_start` 表示连续 NPU payload 尾段在文件中的起始偏移；
- `payload_bytes` 表示该连续尾段总大小；
- `entry_size` 和 `compute_buffer_size` 用于在 host reload 时与 `ioctl` 返回值做一致性校验。

如果这两个值与运行时不一致，则说明当前 guest/kernel 布局已经变化，文件应直接拒绝加载。

### prepack 的新职责

当前共识是不再把这件事拆成“prepack + repack”两个独立阶段，而是：

> **直接扩展现有 `tools/rknpu-prepack/rknpu-prepack.cpp`，一步产出最终文件。**

也就是说，这一步 prepack 需要同时做完下面几件事：

1. 把候选 matmul 权重重排成现有 RKNPU 需要的 payload 格式；
2. 生成给 host 读取的极小明文 header；
3. 把所有 `.__rknpu_payload` 对应的数据，按 guest 实际 payload 布局顺序，收拢到文件数据区尾部；
4. 显式写出中间的 padding / hole；
5. 让 `payload_start` 指向这个连续尾段的文件偏移。

保留 `.__rknpu_meta` / `.__rknpu_payload` 现有语义的意义主要在于：

- 继续复用当前已有的 prepack blob 格式；
- 继续复用已有的 payload 构造逻辑；
- 避免同时重写 GGUF loader 和 guest runtime。

### payload 排序规则

NPU payload 连续数据段不能按“GGUF 中出现的顺序”直接拼接，而必须复用 guest 当前实际使用的 payload tensor 排序。

当前共识是复用 `src/llama-model.cpp` 里的 `llama_order_rknpu_prepack_metas(...)` 语义：

- 非 `output.weight` 在前；
- `blk.N.*` 按层号升序；
- 层内按固定组件顺序：
  - `attn_q`
  - `attn_k`
  - `attn_v`
  - `attn_output`
  - `ffn_gate`
  - `ffn_up`
  - `ffn_down`
- 最后再做 lexical fallback。

后续实现里，最好把这段排序规则抽成共享 helper，避免 packer 和 runtime 分别维护两套逻辑。

### payload 放置与跨 domain padding 规则

NPU payload 连续数据段不是简单 append，而是要**严格复现 guest 侧 payload 池中的布局规则**。

当前已经核对过，guest 侧 `ggml/src/ggml-rknpu_re/npu_interface.c` 中的跨 domain 逻辑，本质上是：

- 每次分配看的是当前 `payload_window_bytes`；
- 如果一个 payload tensor 在当前 window/domain 内放不下；
- 那么它就**不能跨过去**；
- 而是直接把起点 bump 到下一个 window/domain 开头；
- 中间被跳过的那一段空间，视为 padding。

按我们当前的 guest ABI，可以把它理解为：

- 每个 domain 固定为 4 GiB；
- 其中一部分是 compute buffer；
- 剩余部分构成该 domain 可用于 payload 的窗口大小；
- 若某个 payload tensor 会越过这个窗口，就整段移到下一个 domain。

因此，NPU payload 连续数据段里必须显式保留这类 hole/padding，而不能偷偷压缩掉。否则 guest 运行时看到的 offset 就会和文件里的 offset 对不上。

### host reload 程序的方向

下一阶段还会新增一个**独立的 host reload 小程序**，职责尽量简单：

1. 打开最终文件；
2. 读取明文 header；
3. 通过 `npu_interface` 获取当前 layout / reload 区间；
4. 校验：
   - `entry_size` 一致；
   - `compute_buffer_size` 一致；
   - `reload_offset` / `reload_bytes` 在 `payload_bytes` 范围内；
5. 按 `entry_size` 逐块：
   - `fseek` 到 `payload_start + current_offset`
   - `fread` 一个 entry
   - 写入 guest payload 池对应位置
   - 调用现有 `ensure/finish` 接口推进 mapping / flush
6. 直到本轮 reload 区间全部完成。

这个工具的目标是完全脱离：

- GGUF parser
- tensor metadata
- `llama_model_loader`
- 当前 host 版本 `llama-cli` 的 partial-load 逻辑

也就是说，host 只知道：

- 文件里从哪里读；
- guest payload 池里往哪里写；
- 每次写多少（`entry_size`）。

### guest 侧本阶段暂不处理的内容

本阶段先不处理 guest 侧 metadata 解密。

也就是说：

- 先把“一步式 prepack + NPU payload 连续数据段 + 独立 host reload”做通；
- 让 host 先完全脱离现有 GGUF loader；
- 后续再在 guest 侧补“先解密 metadata，再继续走现有建模/推理流程”的能力。

这样可以把风险拆开，避免一次同时重写 host loader、guest loader 和文件格式。

### 当前阶段的关键验证项

后续实现时，至少要重点验证下面几类内容：

1. **布局一致性**
   - prepack 计算出的每个 payload tensor `pool_offset`，要和 guest/runtime 当前实际分配偏移一致；
   - 尤其要检查跨 domain 时 hole 的位置是否一致。

2. **NPU payload 连续数据段正确性**
   - 每个 payload tensor 的数据都应当落在预期 offset；
   - 中间 padding 区域必须保零；
   - `payload_bytes` 应等于最终尾段总长度，而不是原始 payload 字节简单求和。

3. **host reload 校验路径**
   - `entry_size` / `compute_buffer_size` 不一致时必须立即失败；
   - reload 时必须按 `entry_size` 顺序推进；
   - 不能回头写已经 finish 过的 entry。

4. **与现有路径的等价性**
   - 在同一 reclaim/return 条件下，现有 loader 路径与新 host reload 路径应当给 guest 提供一致的 payload 内容；
   - 最终推理结果或关键 block dump 应保持一致。

### 这一阶段完成后的预期形态

如果这一步做成，那么 host 侧最终只需要做三件事：

1. 读一个极小的明文 header；
2. 根据 `ioctl` 拿到当前 reload 区间；
3. 按 `entry_size` 从 NPU payload 连续数据段顺序读取并写入 guest payload 池。

这样即使未来把 GGUF metadata 也全部加密，host 仍然不需要理解模型结构，只要继续当一个最小知情的搬运器即可。
