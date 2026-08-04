# Observation Partition Performance Implementation Plan

> **For agentic workers:** Execute inline; do not use Git or run Blender/data tests per user instruction.

**Goal:** 减少 observation 切分中的重复 evaluated mesh 工作和 glTF primitive 数，并提供可定位的阶段日志。

**Architecture:** 将 evaluated mesh 转为一次性 NumPy `SourceGeometry` 缓存。每个 observation 从缓存筛选三角形并构造一个合并 mesh；只映射实际使用的材质，同时保持原 face provenance 契约。

**Tech Stack:** Blender 4.5 Python API、NumPy、glTF exporter。

---

### Task 1: Geometry cache

**Files:**
- Modify: `scripts/blender/sample_observation_partitions.py`

- [ ] 新增 `SourceGeometry` 数据契约。
- [ ] 将 evaluated mesh 提取改为一次性 float32/int32 缓存。
- [ ] 让 floor discovery 和 anchor sampling 读取缓存。

### Task 2: Memory-bounded sector selection

**Files:**
- Modify: `scripts/blender/sample_observation_partitions.py`

- [ ] 删除 `[F, 7, 3]` samples 临时数组。
- [ ] 顺序检查质心、顶点和边中点并累计 Context mask。

### Task 3: Merge observation mesh and compact materials

**Files:**
- Modify: `scripts/blender/sample_observation_partitions.py`

- [ ] 合并所有选中 source 为一个 mesh。
- [ ] 重建全局 vertex、triangle、UV 和 face provenance。
- [ ] 只复制使用中的材质并移除无尺寸图片节点。
- [ ] 保持 object-name/local-triangle 映射可被 segmentation 重新关联。

### Task 4: Progress and timing

**Files:**
- Modify: `scripts/blender/sample_observation_partitions.py`
- Modify: `docs/technical-implementation-details.md`

- [ ] 增加带 flush 的阶段日志和每 observation 统计。
- [ ] 更新技术文档中的缓存、合并 mesh 与限制说明。

### Task 5: Static verification

- [ ] AST 解析修改后的 Python。
- [ ] 检查旧的重复提取调用已从 observation loop 移除。
- [ ] 检查 `source_faces.npz` 所需字段保持不变。
