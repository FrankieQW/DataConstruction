# LightConstruction

LightConstruction 用于整理 Objaverse 对象资产和 FBX 场景资产，并生成“对象可以放置到哪些场景实体上、可以替换哪些场景实体”的预组合语义标注。后续将把这些标注接入 Blender 常驻场景 worker，在不同相机与光照条件下即时组合和渲染训练数据，用于训练光照可控的 diffusion 图像生成模型。

项目的完整规划覆盖 M0–M5；当前代码实现到 M3：

- M0：配置、数据协议、严格 schema 和统一 CLI。
- M1：构建 `data/object/object.json`。
- M2：构建归一化 scene `.blend`、`data/scene/scene.json` 和人工抽检报告。
- M3：使用本地 vLLM 生成 `data/annotation_construction.json`。
- M4：对象几何归一化、放置/替换、碰撞检测、常驻 Blender worker 和渲染。尚未实现，等待训练代码加入后适配。
- M5：文档与端到端验收。本 README 先覆盖 M0–M3；M4 完成后再补充真实训练和渲染命令。

完整设计、数据协议、风险和 M4 算法见 [`plan.md`](plan.md)。

## 1. 当前数据范围

### Object 数据

- 清单：`data/object/Objaverse.md`
- LVIS 标注：`data/object/Objaverse/lvis-annotations.json`
- UID 到 GLB 路径：`data/object/Objaverse/object-paths.json`
- metadata：`data/object/metadata/annotations.json`
- 实际处理范围：Markdown 清单与 LVIS 标注的严格交集，共 80 个对象、74 个归一化类别。

M1 只构建索引，不会从服务器下载或读取全部 GLB。GLB 在服务器上的根目录将在 M4 通过 `OBJECT_ROOT` 接入。

### Scene 数据

当前处理 `data/scene` 下递归发现的三份 Bistro FBX：

- `BistroInterior.fbx`
- `BistroInterior_Wine.fbx`
- `BistroExterior.fbx`

FBX 导入后，每个 mesh 都会获得稳定的 `lc_scene_object_id` 自定义属性。`scene.json` 中的 `node_id` 与该属性一致，因此后续可以直接在归一化 `.blend` 中找到对应物体，不依赖不稳定的 Blender 显示名称。

## 2. 当前目录结构

```text
.
├── configs/
│   ├── default.yaml
│   ├── category_aliases.yaml
│   ├── scene_overrides.yaml
│   ├── category_dimensions.yaml      # M4 预留
│   └── object_overrides.yaml         # M4 预留
├── data/
│   ├── object/
│   ├── scene/
│   └── annotation_construction.json  # M3 正式输出
├── scripts/
│   └── blender_entry.py
├── src/lightconstruction/
├── environment-core.yml
├── environment-llm.yml
├── plan.md
└── pyproject.toml
```

生成的 `.blend`、缓存、日志、抽检图片和渲染输出已写入 `.gitignore`。`object.json`、`scene.json` 和 `annotation_construction.json` 没有被忽略，可在人工验收后选择纳入版本管理。

## 3. 环境配置

### 3.1 基础环境 `lc-core`

建议在仓库根目录执行：

```bash
conda env create -f environment-core.yml
conda activate lc-core
```

环境文件会以 editable 模式安装当前项目。若代码更新后入口不可用，可重新执行：

```bash
pip install -e .
```

检查 CLI：

```bash
python -m lightconstruction.cli --help
```

### 3.2 LLM 环境 `lc-llm`

vLLM 单独使用一个环境，避免其 PyTorch/CUDA 依赖与数据处理环境冲突：

```bash
conda env create -f environment-llm.yml
conda activate lc-llm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
vllm --version
```

`environment-llm.yml` 给出项目建议范围，但 CUDA、驱动和 PyTorch 的最终组合应根据 A100 服务器实际环境锁定。

### 3.3 Blender 4.5

Blender 作为独立程序安装，相关脚本由 Blender 自带 Python 执行，不要在 Conda 环境中安装 `bpy`。

```bash
/opt/blender-4.5/blender --version
```

M2 默认严格要求 Blender 4.5.x。其他版本不会被静默接受，因为不同 FBX importer 版本可能产生不同名称、层级、材质和灯光行为。

## 4. 完整运行流程：M0–M3

以下命令都从仓库根目录执行。

### 4.1 M1：生成 Object 索引

```bash
conda activate lc-core

python -m lightconstruction.cli prepare-objects \
  --config configs/default.yaml \
  --inventory-mode markdown \
  --workers 16
```

正式输出：

- `data/object/object.json`
- `data/object/object_rejects.jsonl`

程序会执行以下检查：

- Markdown 中的路径必须符合 `<shard>/<32位UID>.glb`。
- UID 必须存在于 LVIS 标注和 object path map。
- 合并类别、许可证、作者、来源 URI、名称、tags、GLB 统计和缩略图。
- 默认严格要求得到 80 个对象和 74 个归一化类别；数量变化会直接报错。
- 正式 JSON 在写盘前通过 Pydantic schema，并采用原子替换，避免留下半个文件。

### 4.2 M2：提取 Scene 并生成归一化 Blend

第一次运行建议只启动一个 Blender 进程：

```bash
conda activate lc-core

python -m lightconstruction.cli prepare-scenes \
  --config configs/default.yaml \
  --blender-bin /opt/blender-4.5/blender \
  --workers 1
```

确认服务器内存余量后可把 `--workers` 提高到 2。每个 FBX 使用独立 Blender 子进程，绝不在线程间共享 `bpy`。

正式输出：

- `data/cache/scenes/<scene_id>.blend`
- `data/scene/scene.json`
- `outputs/reports/scene_group_review.jsonl`
- `outputs/reports/scene_group_review.html`

每个 scene mesh 对应一个 `node`；逻辑物体对应一个 `entity`，entity 通过 `node_ids` 引用 mesh。当前实现采用保守的单 mesh 初始分组，再通过人工抽检和 overrides 合并、拆分或改类，避免自动错误合并相邻的重复杯子、椅子等实例。

程序会验证：

- `scene_id`、`node_id` 和 `entity_id` 唯一。
- 每个 entity 引用的 node 都存在。
- 一个 node 不会同时属于多个 entity。
- 每个 mesh node 都有对应 entity。
- transform、AABB/OBB、材质、面数和顶点数符合 schema。

#### 人工抽检与修正

抽检优先覆盖：

- 类别或分组置信度低于阈值的 entity。
- 多 mesh entity。
- 所有 `replaceable=true` 或 `support_surface=true` 的 entity。
- glass、bottle、plate、chair、table 等高频类别。
- 其余 entity 的固定随机种子分层样本。

不要直接编辑生成的 `scene.json`。把修正写入：

```text
configs/scene_overrides.yaml
```

支持的修正包括：

- `rename_categories`
- `entity_flags`
- `merge_entities`
- `split_entities`

修改后重新运行 `prepare-scenes`。默认开启缓存复用；只要 FBX 和配置摘要没有改变，就不会重复导入 FBX，但会重新应用 entity overrides 并生成最终 `scene.json`。

如需生成 context、isolated、top 三种抽检预览，将 `configs/default.yaml` 中的：

```yaml
scene:
  render_review_images: false
```

改为 `true` 后重新运行。预览数量可能很大，建议先在一份 scene 或较小抽检集上确认资源开销。

### 4.3 M3：启动 Qwen3-14B

在第一个终端中启动 OpenAI-compatible vLLM 服务：

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

默认关闭 Qwen thinking，使用 `temperature=0` 和严格 JSON Schema。LLM 只判断类别之间的语义关系，不负责尺度、姿态或碰撞。

### 4.4 M3：生成预组合标注

在第二个终端中执行：

```bash
conda activate lc-core

python -m lightconstruction.cli annotate-construction \
  --config configs/default.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-14B \
  --concurrency 16 \
  --resume
```

正式输出：

- `data/annotation_construction.json`
- `data/annotation_cache.jsonl`
- `data/annotation_review.jsonl`

M3 先按类别聚合，不对 object UID 和 scene entity 做实例级笛卡尔积。放置只保留支撑面类别，替换只保留词形、同义词或名称相似的候选，然后由 LLM 输出：

- `can_place_on`
- `can_replace`
- `confidence`
- `reason`

类别规则随后确定性展开为 scene entity ID。最终 JSON 中每个 object 类别都有 `place_on_entity_ids` 和 `replace_entity_ids`，即使两者都是空集合也会明确保存。

## 5. 断点续跑和失败处理

### Scene

M2 根据 FBX digest 和配置 digest 复用：

```text
data/cache/scenes/fragments/
data/cache/scenes/*.blend
```

使用 `--no-resume` 可强制重新导入。某个 FBX 失败时，错误会写入：

```text
data/cache/scenes/scene_failures.json
data/cache/scenes/logs/
```

只有全部 scene 成功后才会更新正式 `scene.json`。

### LLM

每个成功的类别对会立即追加到 `data/annotation_cache.jsonl`。进程中断后重新运行同一命令即可自动复用成功结果。

如果仍有请求失败，程序先写 review queue，然后拒绝生成不完整的正式标注。只有明确接受不完整结果时才使用：

```bash
python -m lightconstruction.cli annotate-construction \
  --config configs/default.yaml \
  --allow-partial
```

训练数据正式生产不建议使用 `--allow-partial`。

## 6. 主要配置

统一配置位于 `configs/default.yaml`：

- `paths`：输入、缓存、正式输出和 review 文件路径。
- `object`：预期对象数、类别数和许可证策略。
- `scene`：Blender 路径/版本、并发数、ID 长度、类别别名、人工 overrides 和抽检比例。
- `annotation`：vLLM 地址、模型、并发数、重试、置信度阈值和 thinking 开关。

类别名称规则在 `configs/category_aliases.yaml` 中维护。修改任何会影响输出语义的配置都会改变 `config_digest`，下游 JSON 同时记录输入文件 digest，便于复现和检查数据是否过期。

## 7. 输出 JSON 的作用

### `object.json`

保存 80 个对象的 UID、类别、GLB 相对路径、许可证、作者与来源 metadata。它是后续几何归一化和资产加载的索引，不包含已经加载到内存中的几何。

### `scene.json`

保存 scene、mesh node 和逻辑 entity。`node_id` 用于在 `.blend` 中精确查找对象；`entity_id` 用于 LLM 标注以及后续放置/替换。

### `annotation_construction.json`

保存类别级语义判断和展开后的目标 entity ID。该文件只回答“语义上允许放在哪里或替换什么”，不保存本次渲染所选的尺度、位置、相机和灯光。

## 8. A100 和并行建议

- M1 主要是 JSON 处理，不需要 GPU。
- M2 使用独立 Blender 进程；先用 `--workers 1`，根据系统内存提高到 2。
- M3 使用单张 A100 部署 vLLM，并通过异步请求提高吞吐；默认 `--concurrency 16`。
- 不要同时进行大规模 vLLM 标注和 Cycles GPU 渲染。完成 M3 后关闭 vLLM，再把 GPU 交给 Blender。
- `annotation_cache.jsonl` 是增量缓存，不需要为了重新运行而删除。

## 9. 许可证与数据管理

`object.json` 保留逐对象许可证和作者来源。当前 metadata 中包含 CC0、CC-BY、CC-BY-SA、CC-BY-NC 和 CC-BY-NC-SA 等不同条件；进入训练集前必须根据实际用途设置明确的许可证白名单，不能把所有对象笼统视为同一种许可。

Bistro 的 `LICENSE.txt`、Objaverse metadata 和最终训练样本 manifest 都应随数据版本保留。原始 FBX、GLB、纹理、模型权重、缓存与渲染结果默认不提交 Git。

## 10. 常见问题

### Blender 版本不匹配

程序会拒绝非 4.5.x 版本。请通过 `--blender-bin` 指向服务器上的 Blender 4.5，或修改配置进行仅限诊断的版本兼容实验。

### scene node 能否在 Blend 中找到

可以。遍历 Blender 对象并读取：

```python
obj.get("lc_scene_object_id")
```

该值与 `scene.json` 的 `node_id` 对应。`source_fbx_element_id` 当前只是辅助字段，不作为主键。

### 为什么不让 LLM 决定位置和尺度

类别关系适合由 LLM 判断；稳定支撑、尺度、朝向和穿模属于几何约束。M4 会用支撑面、OBB、类别尺寸先验和 BVH 碰撞检测完成这些工作。

### 为什么没有保存组合后的 Blend

M4 计划使用只读 base scene 和长生命周期 Blender worker。每个 job 在内存中临时加入 object、渲染后回滚，默认不保存组合 `.blend`，减少磁盘占用并避免污染基础场景。

## 11. 后续计划：M4–M5

M4 要等训练代码加入仓库后再实现，以便先冻结 Dataset/DataLoader 所需的图像分辨率、条件字段、RGB/depth/normal/mask pass 和 manifest 格式。计划包括：

1. 为 80 个 Objaverse 对象缓存规范化几何、稳定姿态、尺度先验、AABB/OBB、凸包和最低接触点。
2. 从 `annotation_construction.json` 确定性生成 scene-major render jobs。
3. 每个 scene 启动一个长生命周期 Blender worker，只加载一次 base `.blend`。
4. 放置任务提取朝上支撑面、采样位置/yaw、匹配尺度并执行 AABB + BVH 碰撞检测。
5. 替换任务隐藏原 entity 的全部 node，以目标 OBB、底面和主轴匹配新对象。
6. 在一个组合状态下渲染多角度、多光照及训练需要的辅助 pass。
7. 每个 job 完成后按 state journal 回滚，并校验 base fingerprint；回滚失败时重载 scene。
8. 输出可复现的逐样本 manifest，再与训练代码做端到端读取验收。

M4 中规划的 `prepare-geometry`、`build-render-jobs` 和 `render` 命令目前只是 [`plan.md`](plan.md) 中的接口草案，当前 CLI 尚未提供这些命令。训练代码接入并完成 M4 后，将再次更新 README，补齐最终环境锁定、真实渲染命令和训练运行流程。
