# Scene/Object Automatic Composition Design

## 目标

为每个已完成语义分割的 observation 随机选择一个语义兼容的 Objaverse Object，在 partition scene 中求解稳定、无明显碰撞、位于 Core 承载区域且从最终相机可见的姿态，输出组合 GLB 和完整放置记录，不执行图片渲染。

组合使用本地 Transformers 模型（例如 Qwen3）处理 metadata 中无法由高置信规则判断的 Object。LLM只输出类别和放置关系，不输出坐标、旋转或尺度。几何姿态完全由 Blender 和确定性约束求解器产生。

## 总体流水线

```text
Objaverse GLB + metadata
  -> Object 几何分析
  -> metadata/LVIS 规则过滤
  -> 可选本地 Qwen3 分类
  -> object_catalog.json

Observation partition + segmentation
  -> 语义承载实例筛选
  -> 水平连续承载面提取
  -> 随机选择一个兼容 Object
  -> Object upright/尺度归一化
  -> 候选位置和 yaw 采样
  -> 支撑、稳定、边界、BVH 碰撞、相机可见性检查
  -> combined_scene.glb + placement/camera/manifest JSON
```

## 命令

### Object Catalog

```bash
scenecompose build-object-catalog \
  --object-root data/obj \
  --metadata data/obj/metadata/annotations.json \
  --output data/work/object_catalog \
  --config configs/composition.json
```

### Observation 组合

```bash
scenecompose compose-observations \
  --observation-root data/work \
  --object-root data/obj \
  --catalog data/work/object_catalog/object_catalog.json \
  --config configs/composition.json \
  --workers 8
```

旧切分和全 Scene 分割命令保持兼容，但组合命令只发现包含 `observation.json`、partition GLB 和完整 segmentation 产物的 observation。

## 配置

新增 `configs/composition.json`，严格拒绝未知字段，包含以下部分：

### `catalog`

- `seed`：稳定随机种子。
- `blender_batch_size`：单个 Blender 进程分析的 GLB 数量。
- `workers`：并行 Blender 分析进程数。
- `minimum_faces`、`maximum_faces`：初步退化/超大模型过滤。
- `maximum_components`：主体组件数量上限。
- `minimum_primary_component_ratio`：最大连接组件占比下限。
- `allowed_licenses`：允许进入组合的许可证。
- `reject_metadata_categories`：architecture、people、characters、vehicles 等默认拒绝类别。
- `accept_metadata_categories`：furniture-home、food-drink、electronics-gadgets 等候选类别。
- `high_confidence_terms`：无需 LLM 即可确认的类别词表。

### `llm`

- `enabled`：是否加载本地模型。
- `backend`：第一版仅支持 `transformers`。
- `model_path`：服务器本地模型目录，不自动下载。
- `device`：例如 `cuda:0`。
- `dtype`：`bfloat16`、`float16` 或 `float32`。
- `batch_size`：metadata prompt 批量大小。
- `max_new_tokens`：结构化输出上限。
- `trust_remote_code`：默认 false。
- `local_files_only`：固定为 true。
- `temperature`：默认 0，保证可复现。
- `max_retries`：无效 JSON 的有限重试次数。

未启用或无法加载 LLM 时，高置信规则对象仍可进入 `accepted`；其余对象标记为 `needs_review`。单个 LLM 输出无效不会终止整个 catalog。

### `object_normalization`

- 各规范 Object 类别的目标高度/最长边范围。
- `component_distance_ratio`：远离主体碎片阈值。
- `upright_axis_candidates`：六个主轴方向。
- `bottom_band_ratio`：底面采样高度带。
- `collision_decimation_faces`：碰撞 mesh 面数上限。
- `minimum_footprint_ratio`：稳定姿态最小底部接触比例。
- `uniform_scale_only`：固定为 true。

### `support`

- `allowed_support_classes`：table、desk、counter、shelf、cabinet、floor。
- `minimum_visible_views`。
- `maximum_slope_deg`。
- `plane_height_tolerance_m`。
- `adjacency_tolerance_m`。
- `minimum_patch_area_m2`。
- `boundary_margin_m`。
- `occupancy_cell_m`。

### `placement`

- `seed`。
- `objects_per_observation`：第一版固定为 1。
- `max_object_trials`。
- `positions_per_surface`。
- `yaw_trials`。
- `minimum_support_coverage`。
- `minimum_clearance_m`。
- `maximum_penetration_m`。
- `minimum_camera_visible_ratio`。
- `camera_samples`。
- `resume` 和失败重试控制。

## Object Catalog

### 发现与 metadata 关联

递归发现 `object_root` 下 `.glb`。优先以文件 stem 作为 UID，并通过 metadata 的 `_scenecompose.glb_relative_path` 验证相对路径。以下情况记录失败而不进入分类：

- GLB 不存在对应 annotation；
- UID 对应多个 GLB；
- metadata 路径与实际路径冲突；
- license 不在允许集合；
- metadata 明确表示场景、建筑、人、动物、车辆、武器或抽象资产。

### Blender 几何分析

使用分批 headless Blender 作业，避免为每个 GLB 启动一次进程，也避免在单进程中加载全部 1,000 个模型。每批完成后退出 Blender并释放资源。

每个 GLB 输出 profile：

- evaluated 三角面和顶点数；
- mesh/object/connected-component 数量；
- 主组件三角面占比；
- 世界包围盒尺寸和原始对角线；
- 六个 upright 候选的高度、footprint、底部接触率和稳定评分；
- 材质和纹理槽数量；
- 是否包含动画、蒙皮或非 mesh 主体；
- 几何拒绝原因。

Profile 独立写入 `object_profiles/<uid>.json`，支持中断恢复。Catalog 汇总只读取完成且 schema/config digest 匹配的 profile。

### 规则分类

文本输入由以下字段规范化组合：

- LVIS 类别；
- `name`；
- `description`；
- `tags[].name`；
- `categories[].name`。

规则输出规范类别、放置类型、兼容承载类和置信度。LVIS 精确映射优先于普通关键词；互相冲突时不自动接受。

### 本地 LLM 分类

LLM只处理规则无法高置信决定的 Object。Prompt 要求输出单个 JSON object：

```json
{
  "decision": "accepted | rejected | needs_review",
  "canonical_class": "cup",
  "placement_type": "surface | floor | wall | unsupported",
  "compatible_support_classes": ["table", "desk", "counter"],
  "target_size_m": {"minimum": 0.08, "maximum": 0.18, "dimension": "height"},
  "confidence": 0.91,
  "reason": "short reason"
}
```

输出必须通过枚举、数值范围和 JSON schema 校验。只接受 `confidence >= llm_acceptance_threshold` 的 accepted 结果；否则标记 `needs_review`。原始 prompt hash、模型路径标识、原始输出和解析错误写入 catalog 日志，避免重复推理。

### Catalog 输出

```text
data/work/object_catalog/
|-- object_catalog.json
|-- summary.json
|-- llm_cache.jsonl
|-- object_profiles/
|   `-- <uid>.json
`-- logs/
```

每个对象最终状态为 `accepted`、`rejected`、`needs_review` 或 `failed`。只有 `accepted` 且 `placement_type` 为 `surface` 或 `floor` 的对象进入第一版组合。

## Object 归一化

组合 Blender 进程重新导入选定 GLB，并复用 catalog profile 中的 upright 候选：

1. 只保留主 mesh 连接组件；与主体相邻且在距离阈值内的附件可保留。
2. 应用选定 upright 旋转，使规范上方向为世界 `+Z`。
3. 按规范类别目标尺寸执行统一缩放。
4. 将原点移动到接触底面的 XY 中心、最低 Z 高度。
5. 计算底部 footprint 采样点、重心和简化碰撞 mesh。
6. 不修改长宽高比例，不让 Object 为适配承载面产生非均匀变形。

若没有稳定 upright、尺度超出允许倍率、主体占比过低或归一化后几何退化，则换下一个 Object。

## 承载面提取

### 输入

- `observation.json`
- `partition/scene_partition.glb`
- `segmentation/geometry/geometry.npz`
- `segmentation/geometry/observation_mapping.npz`
- `segmentation/fusion/face_labels.npz`
- `segmentation/fusion/instances.json`

### 候选面

面必须满足：

- `is_core == true`；
- `visibility_count >= minimum_visible_views`；
- semantic ID 对应允许的承载类别；
- 法线与 `+Z` 夹角不超过 `maximum_slope_deg`；
- 非退化且面积为正。

### 连续 patch

按 semantic/instance、近似高度、共享边或空间邻近关系聚类。对每个 patch 建立局部 XY occupancy grid，而不只使用凸包；这样桌面缺口、L 形边界和孔洞不会被错误填满。

每个 patch 记录：

- support semantic/instance ID；
- face indices 和原 Scene provenance；
- 高度、面积、法线统计；
- occupancy grid 原点、分辨率和压缩 bitset；
- Core/visible 覆盖率；
- 可用内缩区域和稳定 ID。

输出 `composition/support_surfaces.json` 和压缩 occupancy NPZ，并用 segmentation/config digest 控制恢复。

## 随机 Object 选择

每个 observation 使用 `placement.seed + observation_id hash` 初始化 RNG。先根据当前承载面类别筛选 catalog 中兼容对象，再按以下因素加权随机：

- Object 类别均衡权重；
- catalog 置信度；
- Object footprint 与承载 patch 面积匹配度；
- 避免同一批次过度重复 UID。

第一版每个 observation 最多输出一个组合。若选定 Object 求解失败，按稳定随机顺序尝试下一个，最多 `max_object_trials`。

## 候选姿态与约束

### 生成

- 按可用面积在 support patch occupancy 中采样位置。
- 对每个位置测试配置数量的 yaw。
- floor Object 只匹配 floor patch；surface Object 不匹配 floor，除非 catalog 明确允许。
- Object 最低接触平面放在 support patch 高度上方的微小 epsilon 处。

### 硬约束

- Object footprint 的支持覆盖率达到阈值。
- footprint 到 occupancy 边界满足 margin。
- 重心垂直投影位于有效支持区。
- Object 与 partition scene 的 BVH 重叠不超过 penetration 容差。
- 除底部接触带外不得与 Scene 相交。
- Object AABB 和主体采样点位于 partition Context 范围。
- 至少一个最终相机候选达到最小 Object 可见比例。

### 评分

有效候选按以下指标加权：

- 支撑覆盖率；
- 边缘净空；
- 碰撞净空；
- 相机可见比例；
- 与 anchor reference forward 的构图位置；
- 承载面置信度。

LLM分数只用于 Object/承载类别兼容性，不进入连续几何坐标评分。

## 最终相机

从 observation 的 `anchor_eye`、reference frame 和 `camera_motion_radius_m` 中采样，不读取分割渲染图作为最终相机。相机候选必须：

- 不与 Scene 或新 Object 碰撞；
- 保持在 observation camera domain；
- Object 投影面积达到阈值且不过度截断；
- 通过 BVH 射线采样获得足够可见比例；
- 同时保留可理解的 Scene 上下文。

只输出 `camera.json` 中的内参、camera-to-world、world-to-camera、FOV 和可见性指标，不执行渲染。

## 组合导出

每个 observation 写入：

```text
composition/
|-- combined_scene.glb
|-- placement.json
|-- camera.json
|-- support_surfaces.json
|-- support_occupancy.npz
|-- candidate_summary.json
|-- manifest.json
`-- logs/
```

`combined_scene.glb` 包含原 partition scene 和一个变换后的 Object，保留两者材质。Scene 几何保持原世界坐标，不烘焙到 anchor 局部坐标。

`placement.json` 至少包含：

- observation ID、Object UID、源 GLB 和 catalog 类别；
- 支撑 semantic/instance/patch ID；
- Object 原始和规范化包围盒；
- upright、统一 scale、translation、yaw 和最终 4x4 transform；
- 支撑覆盖、边缘净空、重心、碰撞、Context 和可见性指标；
- partition、segmentation、catalog 和配置 digest。

`manifest.json` 使用 `complete`、`failed` 状态。失败也生成 manifest 和 `candidate_summary.json`，但不生成虚假的 combined GLB。

## 批处理和恢复

- 一个 observation 对应一个外部 Blender 工作项。
- `--workers` 控制并发 Blender 进程，按 CPU/RAM 设置，不直接等于 GPU 数。
- Object Catalog 的 LLM 阶段单独运行并只加载一次模型，不在每个 Blender 组合进程加载 Qwen3。
- observation 输出采用 staging 目录，所有必需文件完成后再替换正式 `composition`。
- resume 校验 observation、segmentation、catalog、Object GLB 和配置 digest。
- `--force` 只替换对应 observation 的 composition，不修改 partition 或 segmentation。

## README 中的本地 LLM 部署指南

`README.md` 和 `READMECHINESE.md` 必须包含等价说明：

1. 在服务器准备独立的本地 Qwen3 模型目录，SceneCompose 不自动下载模型。
2. 安装与所选 Qwen3 checkpoint 匹配的 `transformers`、`accelerate`、`safetensors` 和 PyTorch/CUDA。
3. 通过配置指定 `model_path`、device、dtype、batch size、`local_files_only=true` 和 `trust_remote_code`。
4. 说明单 GPU catalog 推理与 8 卡 Scene 分割/Blender 组合的资源分工。
5. 提供无 LLM 的规则模式、启用 LLM 模式、缓存恢复和重新分类命令。
6. 说明不使用远程 API、不启动 HTTP 服务、不上传 metadata。
7. 给出显存不足时降低 batch size、换小参数模型或使用量化 checkpoint 的建议，但第一版代码不负责执行量化。

实际依赖版本和 Qwen3 加载范式在编写 README 时以 Qwen 与 Transformers 官方文档为准。

## 错误处理

- 单个 Object profile/LLM/归一化失败不会终止其他 Object。
- 单个 observation 无兼容承载面或无可放置 Object 时记录失败并继续批处理。
- 输入 schema/digest 不匹配、GLB 缺失或 segmentation 必需字段缺失时，该 observation 立即失败。
- 不允许 NaN/Inf 变换、非正 scale、空 mesh 或零面积 support patch。
- 输出已有且未指定 force/resume 时不覆盖。

## 非目标

- 不生成最终图片或视频。
- 不把多个新 Object 放入同一 observation。
- 不处理墙挂、天花板悬挂、容器内部或铰接交互。
- 不训练 placement 模型。
- 不让 LLM直接生成空间坐标。
- 不自动下载 Qwen3、其他模型或权重。

## 用户验证

代码阶段不运行 Blender、LLM、网络下载或自动测试。用户在服务器依次验证：

1. Object Catalog 的 accepted/rejected/needs_review 分布和随机样例。
2. upright、尺度和主组件保留结果。
3. support patch 是否只落在桌面、台面、架面或地面。
4. combined GLB 是否保持 Scene 和 Object 材质。
5. Object 是否无穿插、稳定且位于承载面边界内。
6. `camera.json` 对应视角是否能看到完整 Object 和足够 Scene 上下文。
7. 相同 seed/config/input 是否产生相同 Object 和放置 transform。
