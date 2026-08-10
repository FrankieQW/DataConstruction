# annotation_construction.json 到 TokenLight 数据合成审阅

结论：`annotation_construction.json` 不能直接交给 TokenLight 训练。它是“语义候选表”，中间还缺一整层 M4：

```text
annotation_construction.json
→ 组合任务 render_jobs.jsonl
→ Blender 完成放置/替换并冻结几何与相机
→ 渲染线性光照分量
→ TokenLight train/validation/test.jsonl
→ TokenLightDataset
```

当前仓库确实还没有实现这座桥。README 明确说明 `prepare-geometry`、`build-render-jobs`、`render` 仍是接口草案，[README.md](README.md#L361)；CLI 目前只有 M1–M3，[cli.py](src/lightconstruction/cli.py#L15)。

另外，当前工作区没有实际的 `data/annotation_construction.json`、`object.json` 和 `scene.json`，所以我现在能确定接口和算法，但不能统计你服务器上实际可生成多少组合。

### 三种数据不要混在一起

| 层级 | 作用 | 主要字段 |
|---|---|---|
| 语义标注 | 判断可以放到哪里/替换什么 | `object_index`、`targets_by_object_category` |
| 组合任务 | 固定一次具体组合 | `scene_id`、`object_uid`、`operation`、`target_entity_id`、`seed` |
| TokenLight manifest | 训练时构造 source/target | `ambient`、`dark`、`point_lights`、`diffuse`、`camera`、`canonical` |

TokenLight 校验器要求 manifest 至少包含 `id/asset/ambient/dark/point_lights/diffuse/camera/canonical`，[validate_components.py](Lumina-T2X/tools/tokenlight_data/validate_components.py#L63)。Dataset 随后在线性 RGB 中组合 source/target，而不是读取组合标注，[dataset.py](Lumina-T2X/lumina_next_t2i/tokenlight/dataset.py#L74)。

### 第一步：从 annotation 生成组合任务

对每个 `object_index[object_uid] = category`：

1. 读取该类别的 `place_on_entity_ids` 和 `replace_entity_ids`。
2. 在 `scene.json` 中把每个 entity ID 反查到所属 `scene_id`。
3. 根据配额确定性采样，避免 `Scene × Object × Entity` 全笛卡尔积。
4. 为 place 和 replace 分别生成任务。
5. 每个任务携带固定 seed。

推荐每行如下：

```json
{
  "job_id": "bistro_interior__abc123__place__entity456",
  "scene_id": "bistro_interior",
  "scene_blend": "data/cache/scenes/bistro_interior.blend",
  "object_uid": "abc123",
  "object_asset": "00/abc123.glb",
  "object_category": "cup",
  "operation": "place",
  "target_entity_id": "bistro_interior:entity:456",
  "seed": 202608100001,
  "annotation_digest": "...",
  "object_digest": "...",
  "scene_digest": "..."
}
```

这与原项目计划中的 `(scene_id, object_uid, operation, target_entity_id, seed)` 完全一致，[plan.md](plan.md#L405)。

生成任务前应该拒绝以下情况：

- `unresolved_pairs` 非空；
- object UID 在 `object.json` 中不存在；
- entity ID 在 `scene.json` 中不存在；
- place 目标不是 `support_surface=true`；
- replace 目标不是 `replaceable=true`；
- annotation 中记录的 object/scene digest 与当前输入不一致。

### 第二步：Blender 完成真实组合

对于每条 job：

- 打开对应的 normalized base `.blend`；
- 使用 `lc_scene_object_id` 定位 entity 的全部 mesh；
- append/import 对象 GLB；
- 执行几何归一化；
- 完成 place 或 replace；
- 碰撞、支撑、悬空、尺度和可见性检查；
- 固定 object transform 和 camera；
- 失败时写 JSONL，不生成训练样本。

Place：

- 从 target entity 提取朝上的支撑面；
- 根据物体 footprint 收缩可采样区域；
- 采样位置和 yaw；
- 将最低接触点贴到表面；
- AABB broad phase + BVH narrow phase；
- 检查支撑比例、穿透量、边缘距离和画面占比。

Replace：

- 隐藏 entity 对应的全部 node；
- 用目标 OBB、底面和主轴计算新物体的尺度、位置和朝向；
- 碰撞检测时排除被替换的旧 entity；
- 检查尺度失真和周围碰撞。

通过后先写一份“冻结组合 manifest”。同一个组合后续所有光照分量必须使用完全相同的几何、材质、相机和 transform。

### 第三步：为每个冻结组合渲染 TokenLight 分量

每个固定的“scene + inserted object + operation + target + camera”成为 TokenLight 的一个 scene 样本：

```text
components/<composition_id>/
  ambient.exr
  dark.exr
  point_lights/
    light_000.exr
    ...
  diffuse/
    spread_00.exr
    spread_01.exr
    ...
  metadata.json
```

必须是线性 RGB OpenEXR：

- `dark.exr`：关闭环境光和所有受控灯光；
- `ambient.exr`：只开启基准环境光/HDRI；
- `point_lights/*.exr`：每次只开启一个中性点光源；
- `diffuse/*.exr`：固定位置和能量，只改变面积灯尺寸；
- 所有分量保持几何、相机、材质和分辨率完全一致。

最终 `metadata.json` 可以写成：

```json
{
  "id": "bistro_interior__abc123__place__entity456__v00",
  "asset_uid": "abc123",
  "asset": "00/abc123.glb",
  "base_scene_id": "bistro_interior",
  "operation": "place",
  "target_entity_id": "bistro_interior:entity:456",
  "ambient": "components/.../ambient.exr",
  "dark": "components/.../dark.exr",
  "point_lights": [
    {
      "path": "components/.../point_lights/light_000.exr",
      "position": [0.2, 0.4, 1.5],
      "base_energy": 500.0,
      "diffuse": 0.1
    }
  ],
  "diffuse": [
    {
      "path": "components/.../diffuse/spread_00.exr",
      "level": 0.0,
      "size": 0.05
    },
    {
      "path": "components/.../diffuse/spread_01.exr",
      "level": 0.2,
      "size": 0.2
    }
  ],
  "in_scene_lights": [],
  "camera": {
    "location": [],
    "rotation_euler": [],
    "focal_length": 50.0
  },
  "canonical": {
    "position_axes": "x=right,y=camera-forward,z=up"
  },
  "composition": {
    "object_transform": [],
    "scale": 1.0,
    "collision": {},
    "seed": 202608100001
  }
}
```

`asset_uid` 建议继续填写插入的 Objaverse UID。这样现有 `build_manifests.py` 会让同一个 object 的所有场景和视角进入同一 split，[build_manifests.py](Lumina-T2X/tools/tokenlight_data/build_manifests.py#L27)。

不过目前只有三个 Bistro scene：按 object UID 切分只能测试“新对象泛化”，不能证明“新场景泛化”。正式评测以后还应增加更多 base scenes，并额外建立 scene-held-out test。

### 现有 `render_assets.py` 不能直接复用来渲染组合场景

不要把组合 `.blend` 目录直接配置成 `paths.object_root` 后运行现有脚本。它在每个 asset 开始时会：

- 清空场景；
- 导入单个 asset；
- 把整个 asset 统一缩放到 `canonical_size`；
- 把整体移动到原点和地面；
- 额外添加程序化地面/背景。

相关逻辑在 [render_assets.py](Lumina-T2X/tools/tokenlight_data/render_assets.py#L565)、[normalize_asset](Lumina-T2X/tools/tokenlight_data/render_assets.py#L328) 和 [add_stage](Lumina-T2X/tools/tokenlight_data/render_assets.py#L351)。这会破坏 Bistro 原始空间、entity 定位和已经计算好的放置关系。

正确做法是新增一个 composition renderer，复用现有脚本中这些部分：

- EXR/Cycles 配置；
- ambient/dark/point-light/diffuse 渲染；
- metadata 格式；
- worker 调度；
- manifest 构建和组件校验；

但替换其 `clear_scene → import_asset → normalize_asset → add_stage` 流程，改为：

```text
打开 normalized base scene
→ 应用冻结组合状态
→ 使用冻结相机
→ 禁用原场景不可控灯光
→ 渲染 TokenLight 分量
→ 回滚组合状态
```

### 关于 `in_scene_light`

`annotation_construction.json` 不包含灯具 mesh、灯光绑定或 mask，所以它本身不能生成 `in_scene_light` 数据。

你当前配置实际上启用了：

- `tasks: [..., in_scene_light]`
- `fixture_mask_enabled: true`
- `in_scene_lights_per_scene: 1`

见 [config.yaml](Lumina-T2X/lumina_next_t2i/config.yaml#L29)。现有 renderer 会人为添加球形 fixture，这不是 Bistro 场景中的真实灯具。如果这一批数据的目标主要是 scene/object 合成，第一版建议先生成：

```yaml
data:
  tasks: [ambient_scale, global_diffuse, add_light]

model:
  fixture_mask_enabled: false

render:
  in_scene_lights_per_scene: 0
```

等后续建立“灯具 entity → light source → fixture mask”的可靠绑定后，再单独启用 `in_scene_light`。

所以你接下来的实际开发目标不是“转换 JSON 格式”，而是新增两个阶段：

1. `annotation_construction.json → render_jobs.jsonl`
2. `render_jobs.jsonl → 固定组合 → TokenLight component EXR + manifest`

完成后，现有的 `validate_components.py`、`inspect_dataset.py` 和 `TokenLightDataset` 基本可以原样使用。
