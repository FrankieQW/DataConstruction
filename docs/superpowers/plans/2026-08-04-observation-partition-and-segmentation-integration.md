# 观察锚点局部 Scene 采样与语义分割集成实施计划

**目标：** 使用“可站立观察锚点 + 半径范围 + 参考方向 + 视角范围”替换现有 XY 二叉自适应切分，使每个 Partition 对应一个接近人类观察局部空间的场景，并让现有 Mosaic3D + SAM3 + Open3DIS 风格分割流水线直接处理这些 Partition。

**范围：** 本计划实现新的观察锚点切分、Partition 导出、相机参考信息、语义分割输入切换、可见性统计以及组合阶段接口。旧切分代码保留为 legacy 命令。完整 Object 归一化、承载面求解、物理放置和最终渲染另立计划实现。

**执行方式：** Inline Execution。只编写代码、配置和文档；不下载权重、不运行 Blender、模型推理或自动测试，测试与运行由用户完成。

---

## 一、最终流水线

```text
data/scene/<scene>.fbx
  -> 寻找候选地面
  -> 采样可站立观察锚点
  -> 采样参考观察方向
  -> 导出扇柱形局部 Partition Scene
  -> 基于锚点生成分割相机组
  -> Mosaic3D + SAM3 + 2D/3D 融合
  -> 输出语义、实例、可见性和组合接口
  -> 后续 Object 组合
  -> 后续组合结果渲染
```

切分阶段不按遮挡删除几何。墙后几何可以保留在 Partition 中。相机可见性由分割渲染深度统计，最终 Object 放置还需要针对最终相机执行 BVH 射线检查。

## 二、命令设计

新增单 Scene 命令：

```bash
scenecompose sample-observations \
  --scene data/scene/example.fbx \
  --output data/work/example/observations \
  --config configs/observation_partition.json \
  --blender "$SCENECOMPOSE_BLENDER"
```

新增批量命令：

```bash
scenecompose sample-observations-all \
  --scene-root data/scene \
  --output-root data/work \
  --config configs/observation_partition.json \
  --workers 4
```

新增 Partition 分割命令：

```bash
scenecompose segment-observations \
  --observation-root data/work \
  --config configs/segmentation.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 8
```

旧命令 `partition`、`partition-all` 和 `segment-scenes` 暂时保留，标记为 legacy/debug，不作为默认组合流水线入口。

## 三、观察切分配置

新增 `configs/observation_partition.json`：

```json
{
  "schema_version": 1,
  "observations_per_scene": 8,
  "seed": 20260804,
  "meters_per_blender_unit": null,
  "up_axis": "Z",
  "floor_max_slope_deg": 15.0,
  "floor_height_band_m": 0.25,
  "min_floor_component_area_m2": 2.0,
  "min_anchor_spacing_m": 2.0,
  "anchor_attempts_per_output": 64,
  "observer_height_m": 1.6,
  "min_head_clearance_m": 1.8,
  "min_body_clearance_radius_m": 0.35,
  "body_clearance_ray_count": 16,
  "radius_m": 6.0,
  "horizontal_angle_deg": 100.0,
  "vertical_below_anchor_m": 0.25,
  "vertical_above_anchor_m": 4.0,
  "camera_motion_radius_m": 1.0,
  "context_margin_m": 0.5,
  "direction_trials": 8,
  "minimum_core_triangles": 500,
  "export_format": "glb",
  "include_materials": true
}
```

所有随机过程必须使用 `seed + scene_hash` 初始化。相同 Scene、配置和 seed 必须生成相同锚点、方向和 observation ID。

## 四、数据结构与输出契约

新增目录结构：

```text
data/work/<scene-id>/observations/
|-- observations.json
|-- summary.json
`-- observation_<content-hash>/
    |-- observation.json
    |-- partition/
    |   |-- scene_partition.glb
    |   `-- source_faces.npz
    |-- segmentation/
    |   |-- manifest.json
    |   |-- geometry/
    |   |-- views/
    |   |-- mosaic3d/
    |   |-- sam3/
    |   |-- fusion/
    |   `-- visualization/
    `-- composition/
```

`observation.json` 至少包含：

```json
{
  "schema_version": 1,
  "observation_id": "observation_<hash>",
  "source_scene": "relative/path/scene.fbx",
  "anchor_floor": [0.0, 0.0, 0.0],
  "anchor_eye": [0.0, 0.0, 1.6],
  "reference_forward": [1.0, 0.0, 0.0],
  "reference_right": [0.0, 1.0, 0.0],
  "up": [0.0, 0.0, 1.0],
  "radius_m": 6.0,
  "horizontal_angle_deg": 100.0,
  "vertical_range_world": [-0.25, 4.0],
  "camera_motion_radius_m": 1.0,
  "context_margin_m": 0.5,
  "partition_geometry": "partition/scene_partition.glb",
  "source_faces": "partition/source_faces.npz",
  "quality": {}
}
```

`source_faces.npz` 包含：

- `output_triangle_index`
- `source_object_index`
- `source_polygon_index`
- `source_instance_index`
- `is_core`
- `is_context`

Partition 几何保持原始 Scene 世界坐标，不平移到局部原点。后续 Object 放置和最终渲染因此可以直接复用世界坐标。

## 五、观察锚点采样算法

### 5.1 候选地面提取

不依赖语义分割，仅使用几何：

1. 遍历 evaluated mesh 和实例化 mesh。
2. 对三角形计算世界坐标、面积、法线和质心。
3. 保留朝上的三角形：`dot(normal, up) >= cos(floor_max_slope_deg)`。
4. 按三角形共享边建立连通组件。
5. 使用质心高度带 `floor_height_band_m` 将跨层组件拆分。
6. 计算组件总面积、XY 范围和高度方差。
7. 删除面积小于 `min_floor_component_area_m2` 的组件。

桌面也可能是水平面。降低误采样的方法：优先选择面积大、局部高度方差小、上方净空充足、周围存在连续下方支撑的组件。多楼层 Scene 可以保留多个有效 floor component。

### 5.2 锚点生成

1. 按 floor component 面积分配采样配额。
2. 在候选三角形上按面积加权随机采样重心坐标。
3. 使用 Poisson-disk 约束保证锚点间距至少 `min_anchor_spacing_m`。
4. 每个 observation 最多尝试 `anchor_attempts_per_output` 次。
5. 无法生成足够锚点时输出已有结果并在 summary 中记录警告，不复制无效锚点。

### 5.3 可站立性检查

使用 Blender `BVHTree`：

- 从 floor 点略上方沿 `+Z` 发射射线，确保 `min_head_clearance_m` 内没有顶面。
- 在膝部、躯干和眼高三个高度进行水平径向射线检测。
- 每层发射 `body_clearance_ray_count` 条射线。
- `min_body_clearance_radius_m` 范围内出现障碍则拒绝。
- 在锚点周围小圆上的若干位置向下射线，确认地面连续，避免采到台面边缘或悬空薄片。

此检查只验证锚点附近是否适合作为相机参考区域，不进行整个 Partition 的遮挡裁剪。

## 六、参考方向与扇柱形范围

### 6.1 参考方向

对每个有效锚点随机生成 `direction_trials` 个 yaw。每个方向按核心扇区内的三角形数量和表面积评分，选择内容量最大的方向；评分不执行遮挡检测。

若所有方向的核心三角形数都低于 `minimum_core_triangles`，丢弃该锚点并继续采样。

### 6.2 Core 范围

三角形采样点 `x` 属于 Core 的条件：

```text
horizontal_distance(anchor, x) <= radius_m
angle(reference_forward, x - anchor) <= horizontal_angle_deg / 2
anchor_z - vertical_below_anchor_m <= x.z
x.z <= anchor_z + vertical_above_anchor_m
```

### 6.3 Context 范围

为了允许相机围绕锚点移动且避免几何边界突然截断：

```text
context_radius = radius_m + camera_motion_radius_m + context_margin_m
context_half_angle = core_half_angle
  + atan((camera_motion_radius_m + context_margin_m) / radius_m)
```

垂直上下界也增加 `context_margin_m`。

### 6.4 三角形选择

不执行布尔裁剪。若三角形的质心、任一顶点或最长边中点落入 Context，则保留整个三角形。

`is_core` 由三角形质心是否位于 Core 决定。这样可以保留完整拓扑、UV 和材质，但边界会包含少量范围外几何，这是预期行为。

## 七、Partition 导出

使用新的 Blender 脚本 `scripts/blender/sample_observation_partitions.py`：

- 使用 `--background --factory-startup`。
- 导入 FBX 后忽略原始灯光和相机。
- 遍历 dependency graph，保留实例变换。
- 为每个 observation 重建选中三角形的 mesh。
- 保留材质槽、material index、UV 和平滑标记。
- 默认导出 GLB，避免当前 `.blend` partial-write loopback 问题。
- 导出采用 staging 目录，全部 observation 完成后再原子替换正式目录。
- 不删除或覆盖源 Scene。

若 GLB 纹理体积过大，后续可增加 `external_gltf` 模式；第一版优先保证稳定可加载。

## 八、语义分割输入改造

### 8.1 Observation 发现

新增 `src/scenecompose/segmentation/observation_discovery.py`：

- 递归发现 `observation.json`。
- 验证其引用的 GLB 和 `source_faces.npz`。
- 按源 Scene 相对路径和 observation ID 确定性排序。
- 生成 `ObservationInput`，替代当前只包含 FBX 的 `SceneInput`。

### 8.2 输出位置

分割结果写入 observation 自身目录：

```text
<observation-directory>/segmentation
```

不再写入新的 `<scene-id>/segmentation` 目录，从而确保 Partition、分割结果和后续组合结果属于同一个 observation。

### 8.3 Blender 几何准备

修改 `scripts/blender/prepare_segmentation_scene.py`：

- 支持读取 observation manifest。
- 导入 `scene_partition.glb`。
- 读取 anchor、reference frame 和相机移动范围。
- 保持当前几何采样、纹理采样和 source triangle 映射。
- 将 Partition triangle ID 与 `source_faces.npz` 关联，最终 face label 可以回溯到原 Scene。

## 九、基于锚点的分割相机组

当前按占用网格生成相机的策略改为 observation camera rig：

1. `anchor_eye` 是相机组中心参考点。
2. 相机位置在 `camera_motion_radius_m` 内采样。
3. 相机不得低于 floor，也不得超出 Partition Context。
4. 使用 BVH 检查相机位置不在几何内部，并满足近距离碰撞约束。
5. 朝向围绕 `reference_forward` 产生小范围 yaw/pitch 扰动。
6. 相机水平 FOV 使用 observation 的 `horizontal_angle_deg` 或单独配置的较小 FOV。
7. 输出 RGB、OpenEXR depth、内参和双向外参。

分割相机组是为了获得 SAM3/Open3DIS 的多视角证据，不是最终组合渲染相机。最终相机必须仍从同一个 anchor camera domain 中采样。

## 十、可见性统计

不在切分阶段删除遮挡几何。分割视图产生后，使用深度图统计：

- `point_visibility_count[N]`
- `face_visibility_count[F]`
- `instance_visible_views`
- 每个实例在各视图中的可见点比例

修改 `fusion/point_labels.npz`，加入：

- `visibility_count`
- `is_observed`

修改 `fusion/face_labels.npz`，加入：

- `visibility_count`
- `is_core`

不可见点仍可保留 Mosaic3D 语义标签，但后续组合默认只使用：

```text
is_core == true
visibility_count >= composition.min_visible_views
```

## 十一、组合阶段接口

本轮不实现完整 Object 组合，但必须稳定输出以下接口：

- observation anchor 和参考坐标系
- camera motion domain
- Partition GLB
- 原 Scene face provenance
- point/face semantic labels
- instance IDs 和包围盒
- point/face visibility count
- Core/Context 标记

后续组合阶段的顺序固定为：

```text
读取 observation 与 segmentation
  -> 选择语义承载实例
  -> 提取几何承载面
  -> 采样最终相机位姿
  -> 过滤相机视锥外候选
  -> BVH 检查候选点对最终相机可见
  -> 放置 Object
  -> 碰撞、稳定性和边界检查
  -> 渲染组合结果
```

BVH 可见性只检查候选放置点或 Object 包围盒采样点，不用于第一步 Partition 几何裁剪。

## 十二、代码文件规划

### 新增

```text
configs/observation_partition.json
src/scenecompose/observation/__init__.py
src/scenecompose/observation/config.py
src/scenecompose/observation/contracts.py
src/scenecompose/observation/discovery.py
src/scenecompose/observation/manifest.py
src/scenecompose/observation/pipeline.py
scripts/blender/sample_observation_partitions.py
scripts/run_observation_partition.sh
scripts/run_observation_partition_all.sh
src/scenecompose/segmentation/observation_discovery.py
docs/observation-partition-artifacts.md
```

### 修改

```text
src/scenecompose/cli.py
src/scenecompose/segmentation/pipeline.py
src/scenecompose/segmentation/manifest.py
src/scenecompose/segmentation/artifacts.py
src/scenecompose/segmentation/lifting.py
scripts/blender/prepare_segmentation_scene.py
configs/segmentation.json
README.md
READMECHINESE.md
```

### 保留但降级为 legacy

```text
src/scenecompose/regions.py
scripts/blender/partition_scene.py
configs/partition.json
scripts/run_partition.sh
scripts/run_partition_all.sh
```

不删除旧文件，避免破坏已有命令和中间结果。

## 十三、实施任务顺序

### Task 1：观察切分配置和契约

- [ ] 实现严格 JSON 配置解析和范围校验。
- [ ] 定义 `ObservationAnchor`、`ObservationRegion` 和输出 schema。
- [ ] 实现稳定 observation ID 和配置摘要。
- [ ] 增加默认配置文件。

### Task 2：Blender 地面分析和锚点采样

- [ ] 提取 evaluated triangle、法线、面积和 provenance。
- [ ] 建立水平三角形连通组件和高度带。
- [ ] 实现面积过滤和 floor component 排序。
- [ ] 实现面积加权采样、Poisson 间距和固定随机种子。
- [ ] 实现头部、身体和地面连续性 BVH 检查。

### Task 3：方向选择和扇柱形几何选择

- [ ] 生成多个随机 yaw 候选。
- [ ] 按 Core 内表面积和三角形数量评分。
- [ ] 计算 Core 与 Context 参数。
- [ ] 使用顶点、质心和边中点保守选择完整三角形。
- [ ] 生成 `is_core/is_context` 和 source face 映射。

### Task 4：Partition 导出和批处理

- [ ] 重建带 UV/材质的 observation mesh。
- [ ] 导出 GLB 和 observation manifest。
- [ ] 使用 staging 目录原子发布。
- [ ] 实现单 Scene 和批量 CLI。
- [ ] 实现失败隔离、summary 和 `--force`。

### Task 5：Observation 分割发现与调度

- [ ] 实现 observation manifest 发现和校验。
- [ ] 新增 `segment-observations` CLI。
- [ ] 将一个 observation 作为一个 GPU work item。
- [ ] 输出直接写入 observation 目录。
- [ ] 保持 `segment-scenes` 兼容。

### Task 6：锚点相机组

- [ ] 修改 Blender 准备脚本读取 observation metadata。
- [ ] 在 camera motion domain 内生成分割相机。
- [ ] 实现相机位置碰撞检查。
- [ ] 输出 RGB、depth 和 camera JSON。
- [ ] 在 view metadata 中记录相对 anchor 的偏移。

### Task 7：可见性与分割产物扩展

- [ ] 在 depth lifting 时累积 point visibility。
- [ ] 将 visibility 投票映射到 mesh face。
- [ ] 为实例统计观察视图和可见比例。
- [ ] 将 `is_core` 传播到最终 face labels。
- [ ] 更新 PLY 可视化和实例 JSON。

### Task 8：README 和组合接口文档

- [ ] 将新观察切分标记为默认第一步。
- [ ] 将旧自适应切分标记为 legacy。
- [ ] 增加 Mamba/Pixi 使用命令。
- [ ] 记录参数、目录、坐标系和失败行为。
- [ ] 记录后续组合阶段必须执行的 BVH 可见性约束。

## 十四、用户验证清单

以下命令和检查由用户在服务器执行，代码实施阶段不运行：

### 14.1 切分检查

```bash
scenecompose sample-observations \
  --scene data/scene/<small-scene>.fbx \
  --output data/work/<small-scene>/observations \
  --config configs/observation_partition.json
```

人工检查：

- 锚点位于地面而不是桌面、墙体或场景外。
- GLB 以锚点为参考包含前方扇形空间。
- 相机移动 margin 内没有明显几何缺口。
- 材质和 UV 正常。
- Core/Context face 数量合理。

### 14.2 分割检查

```bash
scenecompose segment-observations \
  --observation-root data/work \
  --config configs/segmentation.json \
  --gpus 0 \
  --workers 1
```

人工检查：

- 相机位于 anchor motion domain 内。
- RGB 和深度匹配。
- table/floor/wall 等标签落在正确位置。
- 不同家具具有不同实例 ID。
- `visibility_count` 与实际渲染可见性一致。
- 墙后几何可以有 Mosaic3D 标签，但不能被标记为已观察。

### 14.3 组合前接口检查

- 每个可组合 face 能回溯到 Partition 和原 Scene。
- Core、Context 和 visibility 字段齐全。
- anchor、forward、camera domain 均为有限世界坐标。
- 最终组合程序无需重新解析源 FBX 即可选择承载面和相机。

## 十五、已知风险与处理

### 地面误判

纯几何可能把大桌面识别为地面。第一版通过组件面积、净空、周围地面连续性和高度聚类降低概率；若仍不可靠，再引入低成本 floor 语义模型或人工规则配置。

### 多楼层和楼梯

高度带允许多个 floor component。楼梯可能被坡度阈值过滤，这是预期行为；锚点优先采样平坦平台。

### GLB 纹理重复

多个 observation 会复制使用到的纹理。第一版优先稳定性；后续根据实际磁盘占用决定是否改为外部 glTF 或共享纹理目录。

### 扇区边界不精确

由于保留完整三角形，Partition 会稍微超出数学扇区。Core/Context 标记用于阻止组合选择边界上下文面。

### 最终 Object 不可见

分割阶段的 visibility 只能表示分割相机组。最终相机确定后仍必须对候选放置点执行 BVH 可见性检查，这是后续组合阶段的硬约束。

## 十六、完成标准

本计划实现完成的标准是：

- 新第一步可以从完整 FBX 确定性地产生一个或多个 observation Partition。
- Partition 由可站立锚点、参考方向、半径和视角定义。
- 切分阶段不执行遮挡几何删除。
- 现有分割模型可以按 observation 独立运行。
- 输出包含语义、实例、Core/Context、可见性和原 Scene provenance。
- 旧切分命令仍可使用，但不再是默认流水线。
- 不包含完整 Object 组合实现，只提供下一阶段所需的稳定输入契约。

