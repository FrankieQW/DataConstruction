# LightConstruction 数据合成项目实施计划

> 状态：M0–M3 代码已于 2026-08-10 落地；目标服务器上的数据生成与人工验收待执行，终点是生成并验收 `data/annotation_construction.json`。M4 及之后仅保留为后续计划，等待训练代码进入仓库后再联合适配。本文不运行测试、不提交 Git。

当前实现入口为 `python -m lightconstruction.cli`，包含 `prepare-objects`、`prepare-scenes` 和 `annotate-construction` 三个子命令。配置、schema、断点缓存、原子写入、Blender 回定位属性、scene 人工抽检报告和 LLM review queue 均已实现；正式 JSON 不在缺少 Blender 4.5 或存在未解决 LLM 请求时降级生成。

## 1. 目标与本阶段边界

项目最终目标是把 Objaverse 单物体资产放入或替换到 FBX 场景中，生成几何关系合理、可复现、可批量渲染的组合场景，供后续“视角与光照可控”的 diffusion 模型训练使用。

分阶段边界：

1. **当前范围 M0–M2——数据处理**：生成 `data/object/object.json`、`data/scene/scene.json` 和一次性归一化场景缓存。
2. **当前范围 M3——预组合标注**：按类别而非逐资产调用本地 LLM，生成 `data/annotation_construction.json`；该文件通过验收即结束当前阶段。
3. **后续范围 M4——训练前即时组合与渲染**：等待训练代码加入后，根据其 Dataset/DataLoader、条件字段、图像输出和批处理方式适配；当前只保留设计，不实现代码、不生成 render jobs。
4. **贯穿范围 M5——README 与端到端说明**：当前先提供覆盖 M0–M3、整体架构和 M4 后续计划的中文 README；待 M4 与训练接口稳定后，再补齐真实渲染、训练和最终验收命令。

M1、M2 应先完成并人工抽查，再进入 M3。否则 scene 类别或逻辑实体划分一旦变化，annotation 会整体失效。当前阶段不实现组合、碰撞、相机、灯光或渲染 worker。

## 2. 已核验的现有数据

### 2.1 Object 数据

- `data/object/Objaverse.md` 是相对路径清单，不是对象元数据表；当前含 **1,000** 条 `.glb` 路径。
- `data/object/Objaverse/lvis-annotations.json` 的结构是 `LVIS 类别 -> Objaverse UID[]`，含 **1,156** 类、**46,207** 个唯一 UID；当前没有一个 UID 同时属于多个类别。
- `data/object/Objaverse/object-paths.json` 的结构是 `UID -> glbs/.../*.glb`，含 **798,759** 条路径；LVIS 的 46,207 个 UID 均能在其中找到。
- 当前严格计算 `Objaverse.md ∩ lvis-annotations.json` 只得到 **80** 个对象；另外 46,127 个 LVIS 对象不在当前 Markdown 清单内。

资产范围已经确认：只使用 `Objaverse.md ∩ lvis-annotations.json` 的 80 个对象，不提供自动扩展到 46,207 个 LVIS 对象的默认路径。当前 80 个对象对应 74 个唯一 LVIS 类别。

`data/object/metadata/annotations.json` 已覆盖 Markdown 中的 1,000 个对象；入选的 80 个对象均包含名称、标签、作者、GLB 统计、缩略图和许可证。许可证分布为：74 个 `by`、2 个 `by-nc`、2 个 `by-nc-sa`、1 个 `by-sa`、1 个 `cc0`。M1 直接按 UID 合并这些字段。`metadata/summary.json` 中存在旧工程绝对路径，只用于查看下载统计，不复制到正式输出；正式数据只保存 UID、来源 URI 和相对路径。

### 2.2 Scene 数据

当前有三份 FBX：

- `BistroInterior.fbx`
- `BistroInterior_Wine.fbx`
- `BistroExterior.fbx`

对 `BistroInterior.fbx` 的只读样本检查发现：清空 Blender 初始场景后导入得到 1,191 个对象，其中 1,186 个为 mesh；对象名唯一，但导入后没有父子层级，mesh 自定义属性基本只有 `currentUVSet`。名称中确实包含 `Glass_Beer`、`Glass_Wine`、`Plate`、`LiquorBottle` 等明确语义，也有十进制或十六进制样式的尾缀，但它们不能直接当成已验证的 FBX 内部 ID。

本机 Blender 5.2.0 导入该 FBX 时还触发了灯光字段兼容错误；临时跳过灯光后才能完成只读枚举。因此实现不能依赖“任意最新版 Blender”，必须固定 Blender 版本并设置导入冒烟门槛。

## 3. 关键设计建议

### 3.1 不直接依赖 FBX 数字 ID

以 FBX 内部数字 ID 作为主键风险较高：Blender 导入器通常不会把它稳定暴露给 `bpy`，重新导入、对象名截断或导出都可能改变可见结果。

采用以下两级定位：

- `source_node_key`：`sha1(scene_relative_path + "\0" + raw_import_name)`，用于第一次导入时确定来源节点。
- `scene_object_id`：格式为 `scene_id:node:<hash前16位>`，写入 Blender 对象的自定义属性 `lc_scene_object_id`。

第一次预处理后保存 `data/cache/scenes/<scene_id>.blend`。后续组合只打开此 `.blend`，通过自定义属性查找对象，不再重复导入 FBX。若未来使用能读取 FBX element ID 的解析器，可把它作为 `source_fbx_element_id` 辅助字段保存，但不替换主键。

### 3.2 区分 mesh 节点和逻辑实体

FBX 中一个“物体”可能由多个 mesh 构成，而当前样本导入后层级已展平。`scene.json` 需要同时包含：

- `nodes`：与 Blender mesh 一一对应，用于精确回定位、隐藏、碰撞和变换。
- `entities`：逻辑物体实例，如一只杯子、一张桌子；包含一个或多个 `node_id`，用于 LLM 标注和组合。

自动分组依据名称 stem、空间邻近、共享材质/变换和包围盒关系；所有低置信分组写入 review report，并允许用 `configs/scene_overrides.yaml` 手工合并、拆分或改类。不要让 LLM 直接修改几何分组。

### 3.3 LLM 判断类别兼容性，不做全量实例笛卡尔积

不要把 46,207 个 object 与上千个 scene mesh 逐对送入 LLM。正确流程是：

1. object UID 聚合为唯一 object 类别；scene entity 聚合为唯一 scene 类别。
2. 先用规则和几何属性裁剪候选：放置目标只保留可作为支撑面的类别；替换目标只保留语义相近的类别。
3. LLM 对“object 类别 + scene 类别”输出两个独立布尔值：`can_place_on`、`can_replace`，以及置信度和简短理由。
4. 把类别规则确定性展开到 scene entity ID；object UID 只引用所属类别规则，避免重复写入大量相同 ID。

LLM 只判断语义合理性。尺度、朝向、支撑面积和碰撞仍由几何代码作硬约束；LLM 的 `true` 不保证最终组合成功。

### 3.4 坐标、尺度和朝向必须在组合前统一

内部统一为米、右手坐标、`+Z` 向上。源 GLB 不被改写；每个对象单独保存 `normalization_transform` 和归一化缓存。

朝向采用“自动候选 + 80 个对象全量人工确认”：

1. Blender 导入后应用对象层 transform，但保留原始矩阵。
2. 以 OBB 的三个主轴生成 24 个轴对齐旋转候选。
3. 对每个候选计算凸包底部接触面积、质心投影是否落在支撑多边形内、静置高度和对称性，筛出稳定姿态。
4. 用类别先验消除稳定但倒置的情况，例如瓶/杯的开口朝上、椅背高于座面、车辆车轮朝下。
5. 为每个对象渲染正面、侧面、俯视和透视图，人工从候选中确认；结果写入 `object_overrides.yaml`，以后无需重复判断。

尺度不信任 GLB 原始数值，采用分层策略：

1. `category_dimensions.yaml` 保存类别真实尺寸范围和主要匹配维度，例如杯子按高度、球按直径、扳手按长度、椅子按座高。
2. 直接放置时，先缩放到类别中位尺寸，再检查目标支撑面是否容纳；放不下则拒绝，不无限缩小。
3. 替换时，以 scene 目标 OBB 的语义维度计算 uniform scale，例如椅子匹配高度和座宽、瓶子匹配高度；默认禁止非均匀拉伸。
4. 若原始尺寸相对类别先验偏差超过阈值、稳定姿态不唯一或 scale 极端，则进入人工队列，不自动组合。
5. 最终 scale、朝向、类别先验版本和人工覆盖来源全部写入 manifest，保证可复现。

每个 object 缓存规范化 mesh、AABB/OBB、凸包、最低接触点、已确认 up/front axis、三角形数、材质数和可加载状态。由于本项目只有 80 个对象，建议把全部对象的朝向和尺度等级都人工确认一次，而不是只抽样。

### 3.5 材质与灯光的具体含义

“几何导入成功”只说明顶点、三角面和 transform 存在，不代表渲染外观正确。FBX/GLB 还包含材质槽、贴图路径、透明度、法线、金属度、粗糙度和自发光；灯光则包含类型、位置、方向、颜色、功率以及环境贴图。

Bistro 的贴图需按其 README 显式连接到 Blender Principled BSDF：

- `*_BaseColor.dds`：RGB 按 sRGB 接到 Base Color；Alpha 用于透明度。
- `*_Specular.dds`：按 Non-Color 读取，R 是 AO、G 是 Roughness、B 是 Metalness；使用 Separate Color 拆通道。AO 可与 Base Color 相乘或单独保留为训练 pass。
- `*_Normal.dds`：按 Non-Color 读取。源法线是 DirectX 约定，需要反转绿色通道后再接 Normal Map。
- `*_Emissive.*`：接 Emission Color/Strength，代表灯箱、灯泡等会主动发光的表面。
- Objaverse GLB 通常已有 glTF PBR 节点，但仍需检查透明、双面、transmission、缺贴图和异常纹理路径。

Bistro 同时提供“自发光表面”和“解析灯光”。Cycles 是路径追踪器，如果两套光源同时全强度开启，可能出现重复照明。因此定义互斥的渲染 profile：

- `native_emissive`：启用自发光表面，关闭对应解析灯，保留场景原始/HDR 环境。
- `native_analytic`：启用解析灯，降低或关闭对应自发光贡献，用于快速且低噪声的基线。
- `controlled`：关闭原场景中会干扰实验的灯光，按配置创建 Area/Point/Sun/HDRI，完整记录位置、方向、功率、色温、HDRI、曝光和随机 seed。

`.pyscene` 是 Falcor 的附加场景参数，Blender 不会自动理解。第一版只解析项目确实需要的灯光/材质字段，其余内容在 report 中列为未应用差异。材质预处理必须输出缺失贴图、无法解码 DDS、空材质槽和异常透明度报告，并用固定相机渲染少量基准图人工比对。

### 3.6 许可证信息纳入数据协议

M1 从 `data/object/metadata/annotations.json` 按 UID 读取 `license`、`uri/embedUrl`、作者、名称、tags、GLB face/vertex/texture 统计和缩略图。`object.json` 保留这些信息，训练前由配置明确许可白名单；不同许可条件不能被合并成一个模糊的“可用”标记。Bistro 的 `LICENSE.txt` 和归属信息也写入数据集 manifest。

### 3.7 Scene 逻辑分组的人工抽检方案

生成 `outputs/reports/scene_group_review.html`。每个被审查 entity 显示三张图：带上下文透视图、隔离透视图、俯视图；当前 entity 用高亮色显示，不同 member node 使用同色，页面同时显示 `scene_id/entity_id/category/node_count/category_confidence/grouping_confidence/OBB尺寸`。

第一轮按风险分层抽检：

1. `grouping_confidence < 0.9`、`category_confidence < 0.9`：100% 检查。
2. `node_count > 1` 的多 mesh entity：100% 检查。
3. `replaceable=true` 或 `support_surface=true`：100% 检查，因为它们会直接参与组合。
4. 高频且相邻重复的 glass、bottle、plate、cutlery、chair、table：每个类别每个 scene 至少抽 20 个，不足 20 则全查。
5. 其余 entity：每个 scene 随机抽 10%，且至少 100 个；随机 seed 固定并写进报告。

人工 verdict 只允许 `pass`、`wrong_category`、`merge`、`split`、`not_replaceable`、`not_support_surface`。修正写到 `configs/scene_overrides.yaml`，不手改生成的 JSON。应用修正后进行第二轮：100% 复查所有被修正 entity、所有可替换/支撑面 entity，再对其余项随机复查 10%。验收门槛为目标 entity 类别正确率不低于 98%，错误合并率和错误拆分率分别不高于 1%，所有可替换和支撑面目标均已人工确认。

### 3.8 Base Scene 常驻与回滚策略

`.blend` 文件不会作为文件整体放在显存中：Blender 数据块主要驻留系统内存，Cycles 在渲染同步阶段把所需 mesh、BVH、shader 和纹理传到 GPU。正确的复用单元是“每个 scene 一个长生命周期 Blender worker”。worker 启动时加载一次 `normalized_blend`，设置 `scene.render.use_persistent_data = True`，按 scene-major 顺序处理该 scene 的全部 object job，最后退出并切换下一个 scene。

base scene 必须视为只读。每个 job 只允许修改受控的瞬时状态：

- 新 object 统一放进 `LC_TRANSIENT_OBJECTS` collection，并写入 `lc_transient_job_id`。
- 替换只临时修改目标 node 的 `hide_render`/可见性，不删除 base node。
- 相机复用固定 `LC_CAMERA`，只更新 transform 和参数。
- 灯光复用固定 `LC_LIGHT_RIG`，只更新 transform、能量、颜色和启用状态。
- World、曝光、render pass 等改动在 job 开始前写入 state journal。

每个 job 完成后按 journal 反向回滚：移除瞬时 collection 中的 object，恢复目标可见性、相机、灯光、World 和渲染设置，更新 depsgraph，并核对 base fingerprint。fingerprint 至少包含 base object ID 集合、transform、可见性、collection membership、材质槽、World 和灯光状态。禁止在 worker 中调用保存 `.blend` 的操作。

Objaverse 对象预先转换为规范化 object `.blend`。worker 每个 job 只 append 当前小对象，不重新导入 GLB，也不重新读取大型 scene；渲染该组合的全部视角/光照后，显式移除本 job 创建的 object、mesh、material 和 image datablock。不要每次调用全局 `orphans_purge`，以免误删 base 资源。若保留 object LRU cache，必须设置显存/RAM上限并验证不会让纹理持续累积。

Persistent Data 可以减少重复渲染的同步和 BVH 构建成本，但会增加内存占用；添加/删除 object 或切换目标可见性仍会触发 Cycles 场景更新，不能假设静态场景永远无需同步。实现后比较三种模式：逐 job 重载、常驻 worker 但关闭 Persistent Data、常驻 worker 并开启 Persistent Data。记录 scene load、scene sync/BVH、render、rollback、峰值 RAM/VRAM 和每 10 个 job 的内存趋势。

当前 Bistro 有 633 个纹理文件，磁盘压缩体积约 1.48 GB，三份 FBX 合计约 210 MB；纹理解码/上传后的显存不能按磁盘压缩体积直接估算。服务器验收用至少 100 个连续 job：base fingerprint 必须始终一致，显存和 RAM 在预热后应进入稳定平台而不是随 job 单调增长。若 20 个 job 的移动窗口增长超过配置阈值（初始建议 512 MB），提前重启 worker；阈值以后根据 A100 40 GB/80 GB 的实测余量调整。

安全兜底：rollback fingerprint 不一致、显存/RAM连续增长、Cycles 异常或达到 `restart_after_jobs` 时，worker 退出；调度器从未修改的 base `.blend` 启动新 worker并重试未完成 job。这样正常路径不频繁加载 scene，同时不会让一次清理失败污染后续训练图像。

## 4. 建议的仓库结构

```text
.
├── configs/
│   ├── default.yaml
│   ├── category_dimensions.yaml
│   ├── category_aliases.yaml
│   ├── object_overrides.yaml
│   └── scene_overrides.yaml
├── data/
│   ├── object/
│   │   ├── Objaverse.md
│   │   ├── Objaverse/
│   │   ├── object.json
│   │   └── object_rejects.jsonl
│   ├── scene/
│   │   └── scene.json
│   ├── cache/
│   │   ├── objects/
│   │   └── scenes/
│   └── annotation_construction.json
├── outputs/
│   ├── renders/
│   ├── manifests/
│   └── reports/
├── scripts/
│   └── blender_entry.py
├── src/lightconstruction/
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── object_index.py
│   ├── scene_extract.py
│   ├── scene_semantics.py
│   ├── llm_annotate.py
│   ├── geometry.py
│   ├── compose.py
│   └── manifest.py
├── environment-core.yml
├── environment-llm.yml
└── README.md
```

大型缓存和渲染结果不应提交版本库；JSON 协议、配置、脚本和小型统计报告应版本化。正常流程不保存组合后的 `.blend`，仅在 `debug.save_composed_blend=true` 时为排错保存少量样本。

## 5. 数据协议

所有正式 JSON 顶层都包含：`schema_version`、`generated_at`、`generator_version`、`config_digest`、`source_digests` 和 `stats`。写文件采用临时文件加原子重命名，重复运行结果排序稳定。

### 5.1 `data/object/object.json`

```json
{
  "schema_version": "1.0",
  "inventory_mode": "markdown",
  "object_root": "${OBJECT_ROOT}",
  "stats": {
    "inventory_glbs": 1000,
    "annotated_unique_uids": 46207,
    "selected_objects": 80,
    "rejected_objects": 920
  },
  "objects": [
    {
      "uid": "...",
      "primary_category": "wine_glass",
      "categories": ["wine_glass"],
      "inventory_path": "000-000/<uid>.glb",
      "canonical_path": "glbs/000-000/<uid>.glb",
      "license": null,
      "metadata_status": "missing"
    }
  ]
}
```

处理规则：

1. 逐行读取 Markdown，只接受规范的相对 `.glb` 路径，拒绝绝对路径和 `..`。
2. 从文件名提取并校验 32 位十六进制 UID。
3. 反转 LVIS 映射得到 `UID -> categories[]`。
4. 用 UID 而不是原始路径格式求交集，解决 Markdown 无 `glbs/` 前缀的问题。
5. 用 `object-paths.json` 补充 canonical path，并记录缺失、重复、非法 UID。
6. 按 UID 排序输出；统计必须满足 `selected + rejected = inventory_glbs`。

### 5.2 `data/scene/scene.json`

```json
{
  "schema_version": "1.0",
  "scenes": [
    {
      "scene_id": "bistro_interior",
      "source_fbx": "Bistro_v5_2/BistroInterior.fbx",
      "normalized_blend": "data/cache/scenes/bistro_interior.blend",
      "units": "meter",
      "up_axis": "+Z",
      "nodes": [
        {
          "node_id": "bistro_interior:node:...",
          "raw_import_name": "Bistro_Research_Interior_Glass_Beer_...",
          "source_fbx_element_id": null,
          "object_type": "MESH",
          "transform_world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
          "aabb_world": {"min": [0, 0, 0], "max": [0, 0, 0]},
          "materials": ["..."]
        }
      ],
      "entities": [
        {
          "entity_id": "bistro_interior:entity:...",
          "node_ids": ["bistro_interior:node:..."],
          "raw_label": "Glass_Beer",
          "category": "beer_glass",
          "category_confidence": 0.98,
          "grouping_confidence": 1.0,
          "replaceable": true,
          "support_surface": false,
          "obb_world": {}
        }
      ]
    }
  ]
}
```

类别提取先采用可解释规则：去场景前缀、实例编号/哈希尾缀、mesh/LOD 等技术词，拆分 snake/camel case，再查 `category_aliases.yaml`。仅把未匹配的唯一 stem 批量交给 LLM 或人工补齐；每个结果保留原始名称和置信度。

### 5.3 `data/annotation_construction.json`

```json
{
  "schema_version": "1.0",
  "model": {"name": "Qwen/Qwen3-14B", "prompt_version": "v1"},
  "class_rules": [
    {
      "object_category": "cup",
      "scene_category": "table",
      "can_place_on": true,
      "can_replace": false,
      "confidence": 0.99,
      "reason": "杯子可稳定放在桌面上"
    }
  ],
  "targets_by_object_category": {
    "cup": {
      "place_on_entity_ids": ["bistro_interior:entity:..."],
      "replace_entity_ids": ["bistro_interior:entity:..."]
    }
  },
  "object_index": {
    "<objaverse_uid>": "cup"
  }
}
```

生成时使用 JSON Schema 约束输出，`temperature=0`，缓存 key 为 `model + prompt_version + object_category + scene_category`。解析失败、超时或低置信结果写入 review queue，不自动归为 `true`。

## 6. 实施步骤与验收标准

这里的 `M` 是 **Milestone（里程碑）** 的缩写，不是模型名称，也不是必须使用的技术术语。`M0` 表示正式业务处理前的公共基础，`M1`、`M2` 表示按依赖顺序推进的可验收阶段：M0 先定义配置和数据协议，M1 生成 object 数据，M2 生成 scene 数据，M3 做 LLM 标注，M4 做渲染时即时组合，M5 完成文档。编号的作用是方便讨论进度、依赖和验收；例如“完成 M2”就表示 scene JSON 与归一化 blend 均已通过验收。

### M0：配置、Schema 与统一 CLI

实现 Pydantic schema、YAML 配置、日志、随机种子、原子输出和以下统一入口：

```bash
python -m lightconstruction.cli <subcommand> --config configs/default.yaml
```

验收：路径不依赖当前工作目录；所有随机过程从全局 seed 派生；输出携带输入摘要和版本；失败任务可重跑而不破坏已成功结果。

### M1：Object 清单构建（阶段一）

```bash
python -m lightconstruction.cli prepare-objects \
  --config configs/default.yaml \
  --inventory-mode markdown \
  --workers 16
```

验收：严格得到 80 个对象、74 个唯一 LVIS 类别；每条 UID、类别、相对路径、许可证、来源和作者可追溯；重复/缺失/非法项均在 stats 与 rejects 中；不把 metadata summary 中的旧绝对路径写进 JSON。

### M2：Scene 抽取与归一化（阶段一）

```bash
python -m lightconstruction.cli prepare-scenes \
  --config configs/default.yaml \
  --blender-bin /opt/blender-4.5/blender \
  --workers 1
```

服务器版本确定使用 Blender 4.5。控制程序为每个 FBX 启动独立 Blender 子进程，Blender 内部脚本负责：清空初始 Cube/Light/Camera、导入、单位/轴归一化、写稳定 ID、计算包围盒、提取名称与材质、生成逻辑 entity、保存 `.blend`。先用 `--workers 1` 验证三份场景，之后按内存容量提高到 2；不要在线程中并行调用 `bpy`。

验收：

- 三个 scene 均有唯一 `scene_id`，每个 mesh 都有且只有一个稳定 `node_id`。
- `scene.json` 中每个 node 都能在对应 `.blend` 通过 `lc_scene_object_id` 找到。
- 每个 entity 的 `node_ids` 均存在，无悬空引用。
- 原始名称、变换、AABB/OBB、类别和置信度齐全。
- 输出未混入 Blender 初始对象；失败导入生成明确错误报告，不产生半成品 `.blend`。

### M3：类别兼容标注（阶段二）

默认模型选择 **Qwen3-14B 非思考模式**。原因是当前只有 80 个对象、74 个 object 类别，推理吞吐不是瓶颈，A100 更适合优先换取 14B 的指令遵循与常识关系判断稳定性。Qwen3-8B 保留为显存不足或快速调试的回退模型。Qwen3.5-9B 是多模态模型；只有未来把 metadata 中的缩略图一并送入模型、判断“标注类别与外观是否一致”时才启用，当前纯文本类别对分类不使用它。Qwen3-30B-A3B 需要加载更大的总权重，本任务收益有限，不作为默认。

```bash
conda activate lc-llm
vllm serve Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --generation-config vllm
```

```bash
conda activate lc-core
python -m lightconstruction.cli annotate-construction \
  --config configs/default.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-14B \
  --concurrency 16 \
  --resume
```

提示词明确使用 `/no_think` 或相应 chat-template 参数；返回值受 JSON Schema 约束。正式全量标注前先构造 200 个类别对的人工 gold set，覆盖明显可放置、明显可替换、两者都不行和歧义案例；用同一 prompt、`temperature=0` 比较 8B 与 14B 的两个布尔字段准确率、`true` 的 precision 和重复运行一致性。误把不合理关系判成 `true` 会污染组合候选，因此以 precision 优先；若 8B 与 14B 差异不超过 1 个百分点才切换到 8B 节省资源。人工抽查优先检查低置信、建筑结构、灯具、透明容器和同名歧义类。

验收：成功生成唯一正式产物 `data/annotation_construction.json`；每个 object 类别有目标集合或明确的空集合；所有目标 ID 存在于 `scene.json`；文件记录 object/scene digest、模型、prompt 版本和统计；相同输入重复运行结果稳定；低置信与失败项均进入 review queue，没有未经记录的默认兜底。通过这些检查即视为当前 M0–M3 阶段完成。

### M4：几何预处理与渲染时即时组合（后续计划，当前不实施）

启动条件：用户将训练代码放入仓库，并能够读取其 Dataset/DataLoader、训练条件、样本 manifest、图像/辅助 pass 格式和并行方式。在此之前，本节命令与接口均为设计草案，不创建对应实现，避免先做出的渲染数据结构与训练管线不匹配。

只在渲染前处理 Objaverse 几何并缓存，避免每个任务重复分析 GLB：

```bash
python -m lightconstruction.cli prepare-geometry \
  --config configs/default.yaml \
  --workers 8 \
  --resume
```

`annotation_construction.json` 只生成一次并保持只读。它负责提供语义允许关系，但不决定本次选择哪个 target、相机或灯光。因此先从它确定性生成一个轻量工作清单；该清单是可随时重建的调度产物，不是第二份语义标注：

```bash
python -m lightconstruction.cli build-render-jobs \
  --config configs/default.yaml \
  --annotations data/annotation_construction.json \
  --output outputs/manifests/render_jobs.jsonl \
  --seed 20260810 \
  --resume
```

每条 job 是 `(scene_id, object_uid, operation, target_entity_id, seed)`。同一个 scene-object 若同时支持放置和替换，默认生成两个彼此独立的 job，两者都从原始 base scene 开始；若要求每个 scene-object 严格只产生一个 job，则在配置中设置 `operation_policy=prefer_replace_then_place`。

渲染时执行即时组合：

```bash
python -m lightconstruction.cli render \
  --config configs/default.yaml \
  --jobs outputs/manifests/render_jobs.jsonl \
  --scene-major \
  --persistent-scene-worker \
  --persistent-data \
  --restart-after-jobs 100 \
  --attempts-per-pair 32 \
  --resume
```

调度器为每个 scene 启动一个长生命周期 Blender worker。worker 只打开一次对应 `normalized_blend`，依次处理该 scene 的 object job：定位 target entity、append 一个已规范化 object、在内存中完成放置或替换、渲染该组合的全部相机与光照、按 state journal 回滚，再处理下一个 job。base scene 始终不保存、不覆盖；只有回滚校验失败或达到重启阈值时才重新加载。

放置算法：

1. 从目标 entity 提取朝上的三角面，按法线、连通域和面积形成支撑面。
2. 在支撑面内采样位置，避开边缘；将 object 的规范化 up axis 对齐 `+Z`，采样 yaw。
3. 按类别尺寸先验和支撑面尺寸做 uniform scale，把最低接触点落到表面。
4. AABB 做 broad phase，Blender `BVHTree` 做 narrow phase；允许接触容差，不允许明显穿透。
5. 以支撑比例、穿透量、边缘距离、尺度偏差和可见性评分，保存最优解；无可行解时记录失败原因，不强行输出。

替换算法：

1. 隐藏目标 entity 的全部 node，读取其 OBB、底面、主轴和世界变换。
2. 以目标 OBB 和类别尺寸先验计算 uniform scale，继承底面位置与主方向；只在配置显式允许时做小范围非均匀缩放。
3. 碰撞检测排除被替换的旧 entity，但检查周围场景。
4. 保存 object UID、target entity ID、旧/新变换、scale、碰撞统计和随机 seed。

正常输出：

```text
outputs/renders/<job_id>/<view_id>_<light_id>.<ext>
outputs/manifests/<job_id>.json
outputs/reports/render_summary.json
```

组合后的 `.blend` 默认不保存；只有 debug 配置开启时才保存少量失败或抽检样本。逐 job manifest 记录实际 target、操作、变换、scale、碰撞结果、相机、灯光、annotation/object/scene digest 和 seed。

验收：每个 job 都从未修改的 base scene 开始；输出 manifest 可以单独复现组合和渲染；目标可回定位；放置物体有足够支撑且无超过阈值的穿透；替换物体尺度不过度失真；所有失败均有机器可读原因；base `.blend` 的 digest 在渲染前后保持不变。

### M5：中文 README（基础版已完成，后续随 M4 更新）

当前 `README.md` 已包含项目目标、目录结构、数据准备、两套 conda 环境、Blender 安装与版本检查、M0–M3 完整命令、断点续跑、输出 JSON 说明、并行参数、A100 资源分配、常见错误、许可证注意事项和 M4 后续计划。M4 与训练代码完成适配后，再把设计草案替换成已经验证的渲染和训练命令。

## 7. 环境配置建议

Blender 自带 Python，建议作为独立系统程序安装，不要尝试把 `bpy` 强塞进主 conda 环境。vLLM 与几何处理也拆成两个环境，减少 CUDA/PyTorch 依赖冲突。

### `lc-core`

建议 Python 3.11，包含 Pydantic、PyYAML、orjson、NumPy、SciPy、trimesh、Shapely、rtree、Pillow、tqdm、OpenAI Python client。Blender 相关逻辑只使用 Blender 自带模块，通过子进程交互。

```bash
conda env create -f environment-core.yml
conda activate lc-core
pip install -e .
```

### `lc-llm`

仅安装与服务器 CUDA/驱动兼容的 PyTorch、vLLM 和模型依赖。具体 CUDA wheel 不在规划阶段硬编码，应在 A100 服务器上根据 `nvidia-smi` 和 vLLM 安装矩阵锁定，随后把成功版本写回 `environment-llm.yml`。

```bash
conda env create -f environment-llm.yml
conda activate lc-llm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
vllm --version
```

### Blender

服务器确定使用 Blender 4.5 LTS，并固定完整版本与安装路径；仍需对三份 FBX 各执行一次导入冒烟检查，避免个别文件或插件差异。当前本机 5.2.0 的灯光属性错误只作为已知兼容性记录，不影响服务器的 4.5 方案。

```bash
/opt/blender-4.5/blender --version
/opt/blender-4.5/blender --background --factory-startup \
  --python scripts/blender_entry.py -- smoke-import \
  --input data/scene/Bistro_v5_2/BistroInterior.fbx
```

## 8. 并行与 A100 资源策略

- Markdown/JSON 解析主要是流式 I/O；大 JSON 只加载一次，UID 反向索引后再并行，不让每个 worker 重复读取 71 MB 的 path map。
- GLB 几何预处理采用进程池；worker 数以 CPU 内存为约束，默认 `min(8, cpu_count)`，服务器再调高。
- FBX/Blender 使用进程级并行，绝不共享 `bpy`；同一输出文件只由一个进程写。
- LLM 使用异步批请求、缓存和 `--resume`；类别规则去重后再请求。
- 单张 A100 上，vLLM 标注阶段与 Cycles 渲染阶段顺序执行。完成 annotation 后关闭 vLLM，再把 GPU 交给 Blender，避免显存争抢。
- 即时组合任务按 scene 分 shard，每个 scene 使用一个长生命周期 Blender worker；base `.blend` 只加载一次并开启 Cycles Persistent Data。job 之间通过 state journal 和 base fingerprint 严格回滚，同一 job 的多个视角和灯光复用一次组合。单张 A100 默认只运行一个 GPU render worker；CPU 可并行准备下一 object，但不能同时提交多个 Cycles GPU 渲染。
- 每个阶段生成 progress manifest，记录 `pending/running/succeeded/failed`；进程异常后只重跑失败项。

## 9. 建议的质量门槛

在开始大规模生成前设置一个小型人工验收集：每个高频 object 类别至少抽 5 个资产，每种操作至少 50 个组合。检查：

- 类别是否正确，逻辑 entity 是否把多 mesh 物体完整分组。
- 放置关系是否语义合理、是否悬空、是否越过桌面边缘。
- 替换是否保留正确位置/朝向、尺度是否自然。
- 是否存在明显穿模、极端面数、缺材质、透明材质错误或单位异常。
- manifest 是否能复现同一结果。

只有通过小规模门槛后才扩大并行度。后续训练数据还应输出 RGB、depth、normal、albedo、object/entity mask、camera、light 和组合 manifest；仅保存 RGB 不利于排错和可控条件建模。

## 10. 已确认项与剩余风险

已确认：

- 资产范围固定为 Markdown 与 LVIS 的严格交集，共 80 个对象。
- 服务器固定使用 Blender 4.5 LTS。
- `scene.obj` 是笔误，正确输出为 `scene.json`。
- Objaverse metadata 已下载到 `data/object/metadata`，80 个入选对象均有许可证和来源信息。

剩余风险：

1. **scene 逻辑分组**：样本没有父子层级，纯名称规则可能把多部件物体拆开或把重复实例合并；按 3.7 节生成 review HTML、分层抽检并用 overrides 修正。
2. **类别名不是几何真值**：例如 `glass` 既可能表示杯子也可能表示玻璃材质；类别提取必须结合完整名称、尺寸和上下文，保留置信度。
3. **Objaverse 尺度/朝向不统一**：按 3.4 节生成稳定姿态候选、应用类别尺寸先验，并对全部 80 个对象做一次人工确认。
4. **材质与灯光**：按 3.5 节重建 Bistro PBR 通道并选择互斥 lighting profile；几何组合成功不能替代材质与照明验收。
5. **许可证策略**：metadata 已齐全，但不同 CC 条款仍需通过配置决定是否进入具体训练集，并保留逐对象归属信息。

## 11. 推荐实施顺序

当前实施：

1. 完成 M0、M1，生成固定的 80 条 `object.json`，并合并 metadata 中的许可证与来源字段。
2. 使用 Blender 4.5 完成三份 FBX 的 M2；按 3.7 节检查 `scene.json` 的类别和 entity 分组。
3. 冻结 `object.json`、`scene.json` 的 schema 与 digest 后运行 M3，生成和验收 `data/annotation_construction.json`，随后停止当前实施。

恢复 M4 的条件与顺序：

1. 用户把训练代码加入当前仓库。
2. 先分析训练 Dataset/DataLoader、条件编码、所需 RGB/depth/normal/mask pass、图像分辨率、相机/灯光字段、batch 与分布式训练方式。
3. 再冻结 render job 和逐样本 manifest 协议，把 3.8 节的持久 scene worker 接到训练数据生产或离线渲染入口。
4. 用少量对象做端到端样本读取验证后，再扩展到全部 80 个对象；最后更新 M5 README 中的渲染、训练和验收流程。

## 12. 参考资料

- [Blender 4.5 命令行参数](https://docs.blender.org/manual/en/4.5/advanced/command_line/arguments.html)：后台执行、参数顺序和脚本入口。
- [Blender Cycles GPU Rendering](https://docs.blender.org/manual/en/4.5/render/cycles/gpu_rendering.html)：NVIDIA CUDA/OptiX 设备与驱动要求。
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)：用 JSON Schema 约束 LLM 输出。
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)：`vllm serve` 与客户端接口。
- [Qwen3 官方发布说明](https://qwenlm.github.io/blog/qwen3/)：8B/14B 模型、非思考模式和 vLLM 部署建议。
- [Qwen3-14B Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen3-14B)：本项目默认文本模型。
- [Qwen3-8B Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen3-8B)：低显存与快速调试回退模型。
- [Qwen3.5-9B Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)：未来使用缩略图做多模态复核时的候选。
- [Objaverse 1.0 API](https://objaverse.allenai.org/docs/objaverse-1.0/)：UID、对象路径、annotation 和逐对象 license 元数据。
