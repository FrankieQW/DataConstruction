# LightConstruction：从 scene/object 标注到 TokenLight 训练

LightConstruction 把 Objaverse 对象、带语义命名的 Bistro 场景和 `annotation_construction.json` 转成 TokenLight 所需的线性光照分量数据，并把数据验证、固定 manifest 训练 smoke、checkpoint 恢复和正式训练串成一条可审计流程。

当前代码覆盖：

- M1：生成 `data/object/object.json`。
- M2：把 FBX 转成保留稳定 entity ID 的 `.blend`，生成 `data/scene/scene.json`。
- M3：使用本地 vLLM 生成 `data/annotation_construction.json`。
- M4：准备对象几何契约、生成确定性组合 job、在 Bistro 中执行 place/replace，并渲染 TokenLight 分量。
- M5：严格验证、object/scene 隔离切分、Dataset smoke、训练 smoke、checkpoint 恢复和发布记录。

完整设计背景见 [`plan.md`](plan.md)，实现与验收清单见 [`TOKENLIGHT_ADAPTATION_IMPLEMENTATION_PLAN.md`](TOKENLIGHT_ADAPTATION_IMPLEMENTATION_PLAN.md)。TokenLight 模型细节仍由 [`Lumina-T2X/lumina_next_t2i/TOKENLIGHT_GUIDE.md`](Lumina-T2X/lumina_next_t2i/TOKENLIGHT_GUIDE.md) 维护；本文是从原始数据到训练验收的主入口。

## 1. 端到端数据流

```text
Objaverse inventory + metadata
              ↓
        prepare-objects
              ↓
       data/object/object.json

Bistro FBX ── prepare-scenes ──→ normalized .blend + data/scene/scene.json
                                      ↓
object.json + scene.json ── annotate-construction
                                      ↓
                      annotation_construction.json
                                      ↓
      prepare-geometry ──→ prepared_geometry.json + quarantine
                                      ↓
      build-render-jobs ──→ deterministic render_jobs.jsonl
                                      ↓
             Blender composition render
                                      ↓
 ambient / dark / point / diffuse / fixture components + metadata
                                      ↓
        build_manifests → validate_components → inspect_dataset
                                      ↓
                 fixed-manifest training smoke
                                      ↓
                    checkpoint resume smoke
                                      ↓
                    formal train / evaluate
```

`annotation_construction.json` 只表达语义允许关系，例如某类对象能放在什么 entity 上、能替换什么 entity。具体对象、场景、target、transform、相机、灯具和随机 seed 都在可重建的 `render_jobs.jsonl` 中冻结。

## 2. 关键语义边界

### place 与 replace

- 每个可用的 `object_uid × base_scene_id` 最多生成一个固定组合 job。
- 当同一组合同时允许 place 和 replace 时，按 `m4.relation_priority` 选择；默认优先 replace。
- place 将对象底面中心对齐到 support entity 顶面中心，并检查对象 XY footprint 不超过支撑面限制。
- replace 隐藏目标 entity 对应的 mesh，将对象按目标包围盒进行 uniform fit。
- 每个 Blender job 都重新打开只读 base `.blend`，不会把上一个样本的状态带入下一个样本。

### `in_scene_light`

固定组合和相机以后执行 real-first/fallback：

1. 从 `fixture_candidate_entity_ids` 找真实灯具候选。
2. 为每个候选渲染带遮挡的可见 mask。
3. 按可见像素降序、距画面中心升序、entity ID 字典序选择。
4. 有合格真实灯具时：
   - `fixture_source: scene_native`
   - 使用真实灯具 mesh 作为 fixture
   - 不创建程序化球体
5. 没有合格候选时：
   - 创建球形 fixture
   - `fixture_source: procedural_fallback`

两种来源都使用受控点光产生独立 `_on.exr`，但 mask 覆盖的几何来源不同。最终统计必须分别报告二者数量。

### TokenLight 线性分量

组件渲染时禁用 Bistro 原生解析灯和自发光能量。ambient 之后将 World strength 设为零，确保现有 Dataset 公式不会重复计入环境光：

```text
ambient_scale  = dark + (ambient - dark) * scale
global_diffuse = ambient + diffuse_component - dark
add_light      = ambient + max(point_component - dark, 0) * color * intensity
in_scene_light = ambient + max(fixture_on - dark, 0) * color * intensity * transition
```

## 3. 目录与正式产物

```text
configs/
  default.yaml
  category_dimensions.yaml
  object_overrides.yaml
  scene_overrides.yaml
data/
  object/object.json
  scene/scene.json
  annotation_construction.json
outputs/
  manifests/
    prepared_geometry.json
    render_jobs.jsonl
    render_jobs.summary.json
  reports/
    geometry_quarantine.jsonl
    render_job_rejects.jsonl
  tokenlight_dataset/
    components/<job_id>/
      ambient.exr
      dark.exr
      point_lights/*.exr
      diffuse/*.exr
      in_scene_lights/*_on.exr
      in_scene_lights/*_mask.png
      metadata.json
    manifests/{train,validation,test}.jsonl
    validation_summary.json
    validation_errors.jsonl
    dataset_release.json
    render_errors.jsonl
```

`metadata.json` 包含 composition transform、相机与 canonical 坐标、fixture 来源、base scene fingerprint、输入 digests、许可证决策和生成器版本。只有通过 validator 的 metadata 才能进入正式 manifest。

## 4. 环境

建议分开准备数据环境、Blender 和 TokenLight 训练环境。

### LightConstruction

```bash
conda create -n lightconstruction python=3.11 -y
conda activate lightconstruction
pip install -e .
```

M1–M4 外层命令均从仓库根运行：

```bash
python -m lightconstruction.cli --help
```

### Blender

`configs/default.yaml` 默认指向 `/opt/blender-4.5/blender`。服务器先确认：

```bash
/opt/blender-4.5/blender --version
```

M2 默认要求 Blender 4.5。M4 composition worker 通过 `--factory-startup` 启动，并为每个 job 重新打开 base blend。

### TokenLight

```bash
cd Lumina-T2X
conda create -n tokenlight python=3.11 -y
conda activate tokenlight
pip install -e .
```

训练还需要与服务器匹配的 PyTorch、CUDA、Flash Attention、VAE 和 Lumina-Next-T2I 2B 单 shard checkpoint。上游 checkpoint 加载时只允许 TokenLight 新增的 `lighting_encoder` 和 `fixture_mask_embedder` keys 缺失。

## 5. 正式运行前必须填写的配置

代码会拒绝猜测对象尺度和许可证。以下配置为空时不能进入正式 M4。

### `OBJECT_ROOT`

`object.json` 保存相对 canonical asset path，实际根目录通过环境变量提供：

```bash
export OBJECT_ROOT=/mnt/afs_fangwenqi/data/Objaverse
```

### 对象许可证白名单

在 `configs/default.yaml` 中填写已经批准的许可证名称：

```yaml
object:
  license_allowlist: [CC0, CC-BY-4.0]
```

这里的值必须与 `object.json` 的规范化 `license` 完全匹配。空白名单在 `m4.enforce_license_allowlist: true` 时会直接失败。

### Bistro 许可证与用途政策

不要使用示例值。由责任方确认后填写：

```yaml
m4:
  license_policy_version: your-approved-policy-v1
  base_scene_license:
    name: <approved license>
    source_uri: <canonical source>
    attribution: <required attribution>
    decision: allowed
```

未设置 `decision: allowed` 时，render job 为 `unverified`，正式 renderer 默认拒绝执行。

### 对象尺寸与朝向

`configs/category_dimensions.yaml` 为类别默认值：

```yaml
categories:
  mug:
    target_dimensions: [0.10, 0.10, 0.12]
    up_axis: +Z
    front_axis: -Y
    contact_axis: -Z
    fit_mode: uniform_fit
```

`configs/object_overrides.yaml` 只保存已审阅的对象例外：

```yaml
objects:
  <objaverse_uid>:
    target_dimensions: [0.20, 0.08, 0.12]
    up_axis: +Y
    front_axis: +Z
    contact_axis: -Y
    fit_mode: uniform_fit
```

未知类别、缺失资产、错误轴、非白名单许可证和 `disabled: true` 对象进入 quarantine，不会被隐式缩放后渲染。

## 6. M1：生成 object 数据

输入：

- `data/object/Objaverse.md`
- `data/object/Objaverse/lvis-annotations.json`
- `data/object/Objaverse/object-paths.json`
- `data/object/metadata/annotations.json`

运行：

```bash
python -m lightconstruction.cli prepare-objects \
  --config configs/default.yaml \
  --inventory-mode markdown
```

输出：

- `data/object/object.json`
- `data/object/object_rejects.jsonl`

M1 只构建索引和来源信息，不遍历服务器上的全部 GLB。

## 7. M2：生成 scene 数据和归一化 blend

把 FBX 放到 `data/scene/`，运行：

```bash
python -m lightconstruction.cli prepare-scenes \
  --config configs/default.yaml \
  --blender-bin /opt/blender-4.5/blender \
  --workers 1 \
  --resume
```

输出：

- `data/scene/scene.json`
- `data/cache/scenes/<scene_id>.blend`
- `outputs/reports/scene_group_review.jsonl`
- `outputs/reports/scene_group_review.html`

每个 Blender object 保留：

- `lc_scene_object_id`
- `lc_scene_entity_id`
- `lc_scene_category`
- `lc_raw_import_name`

人工修正只写入 `configs/scene_overrides.yaml`，然后重新生成 `scene.json`；不要手工编辑正式 JSON。

## 8. M3：生成 `annotation_construction.json`

先启动 OpenAI-compatible vLLM：

```bash
vllm serve Qwen/Qwen3-14B --host 127.0.0.1 --port 8000
```

再运行：

```bash
python -m lightconstruction.cli annotate-construction \
  --config configs/default.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-14B \
  --concurrency 16
```

输出：

- `data/annotation_construction.json`
- `data/annotation_cache.jsonl`
- `data/annotation_review.jsonl`

必须确认 `unresolved_pairs` 符合预期，且文件中的 object/scene digest 与当前输入一致。M4 会拒绝 stale annotation。

## 9. M4：几何准备和组合 job

### 9.1 几何准备

```bash
python -m lightconstruction.cli prepare-geometry \
  --config configs/default.yaml
```

检查：

```text
outputs/manifests/prepared_geometry.json
outputs/reports/geometry_quarantine.jsonl
```

正式批次前应逐项解释 quarantine。不能为了提高成功率而给未知对象添加宽松默认尺度。

### 9.2 生成固定组合

```bash
python -m lightconstruction.cli build-render-jobs \
  --config configs/default.yaml \
  --annotations data/annotation_construction.json \
  --output outputs/manifests/render_jobs.jsonl \
  --seed 20260810
```

输出：

- `render_jobs.jsonl`：Blender 直接消费的 job。
- `render_jobs.summary.json`：输入 digests、统计和完整 job 文档。
- `render_job_rejects.jsonl`：缺 blend、缺 entity 等拒绝原因。

相同输入、配置和 seed 必须产生相同 job ID、target、yaw 和 transform 目标。

## 10. M4：组合渲染

先用小批次设置：

```yaml
m4:
  max_render_jobs: 8
  render_gpu_ids: [0]
  render_workers: 1
  overwrite: false
  render:
    resolution: 256
    samples: 16
```

运行：

```bash
python -m lightconstruction.cli render \
  --config configs/default.yaml \
  --blender-bin /opt/blender-4.5/blender \
  --workers 1
```

正式运行不要使用 `--allow-partial`。失败 job 会进入：

```text
outputs/tokenlight_dataset/render_errors.jsonl
outputs/tokenlight_dataset/render_workers/*/errors.jsonl
```

组件目录先写为 `<job_id>.partial`，只有 metadata 完整后才原子改名。`overwrite: false` 会复用已有完整 `metadata.json`，但拒绝覆盖不完整目录；修复原因后应明确清理对应单个 partial job，再重试。

## 11. 构建 split 和严格验证

编辑 `Lumina-T2X/lumina_next_t2i/config.yaml`，让 TokenLight 指向 M4 输出：

```yaml
paths:
  dataset_root: /absolute/path/to/outputs/tokenlight_dataset
  render_output_root: /absolute/path/to/outputs/tokenlight_dataset
  train_manifest: /absolute/path/to/outputs/tokenlight_dataset/manifests/train.jsonl
  validation_manifest: /absolute/path/to/outputs/tokenlight_dataset/manifests/validation.jsonl
  test_manifest: /absolute/path/to/outputs/tokenlight_dataset/manifests/test.jsonl

data:
  require_composition_contract: true
  split_profile: object-held-out
  tasks: [ambient_scale, global_diffuse, add_light, in_scene_light]
  task_probabilities:
    ambient_scale: 0.25
    global_diffuse: 0.25
    add_light: 0.25
    in_scene_light: 0.25
  inspect_all_tasks: true

model:
  fixture_mask_enabled: true
```

在 `Lumina-T2X/` 下运行：

```bash
python tools/tokenlight_data/build_manifests.py \
  --config lumina_next_t2i/config.yaml

python tools/tokenlight_data/validate_components.py \
  --config lumina_next_t2i/config.yaml

python tools/tokenlight_data/inspect_dataset.py \
  --config lumina_next_t2i/config.yaml
```

`validate_components.py` 会检查：

- camera/canonical/transform 长度和坐标空间
- EXR shape、NaN/Inf、point/diffuse/fixture contribution
- fixture mask 和 `fixture_source`
- scene-native entity 绑定与 fallback 排他性
- base scene fingerprint、lineage、许可证政策
- 配置中的每个任务至少有一个 eligible scene
- 对应 split profile 的跨 split 交集

`inspect_dataset.py` 使用固定 `(index, sample_seed)` 搜索，必须实际读取并展示所有启用任务；不会再用随机前几个样本冒充任务覆盖。

### 切分语义

- `object-held-out`：同一 `asset_uid` 不跨 split；允许复用 base scene，只能声明对象泛化。
- `scene-held-out`：同一 `base_scene_id` 不跨 split；至少需要三个不同 base scene。场景不足时直接失败。

每次构建还会写 `dataset_release.json`，保存 split、fixture 来源、许可证政策、lineage digest 和三个 manifest digest。

## 12. 固定 manifest 训练 smoke

Dataset 可读不等于训练可用。任何数据版本在两阶段 smoke 完成前都不是 `train-ready`。

### 12.1 Smoke 配置

复制正式配置为服务器本地 smoke 配置，并显式修改：

```yaml
paths:
  upstream_checkpoint: /path/to/Lumina-Next-T2I/consolidated_ema.00-of-01.safetensors
  vae: /path/to/sdxl-vae
  dataset_root: /path/to/fixed-small-tokenlight-dataset
  train_manifest: /path/to/fixed-small-tokenlight-dataset/manifests/train.jsonl
  validation_manifest: /path/to/fixed-small-tokenlight-dataset/manifests/validation.jsonl
  output_root: /path/to/smoke-outputs
  resume_checkpoint: null

data:
  num_workers: 0
  tasks: [ambient_scale, global_diffuse, add_light, in_scene_light]

train:
  micro_batch_size: 1
  gradient_accumulation_steps: <fixed schedule large enough to cover all tasks>
  max_steps: <small positive number>
  checkpoint_every_steps: 1
  smoke:
    enabled: true
    required_tasks: [ambient_scale, global_diffuse, add_light, in_scene_light]
    summary_json: smoke_summary.json

logging:
  run_id: tokenlight_fixed_smoke_v1

runtime:
  gpu_ids: [0]
```

Smoke 强制单进程、单 GPU、`num_workers: 0`，并要求固定 sampler 序列覆盖所有启用任务。

### 12.2 第一阶段

在 `Lumina-T2X/` 下仍使用正式训练入口：

```bash
python lumina_next_t2i/train_tokenlight.py \
  --config /path/to/tokenlight_smoke.yaml
```

第一阶段必须完成：

- forward、backward 和至少一次 optimizer update
- optimizer state 的 step 在 `optimizer.step()` 前后实际递增
- finite `loss_total`、逐任务 loss 和 grad norm
- 有效 lighting token 进入模型
- `in_scene_light` 的 fixture mask 非空并进入模型
- 保存模型、optimizer、RNG、dataloader state 和 checkpoint index

成功后 `smoke_summary.json` 必须是：

```json
{"status": "awaiting-resume", "train_ready": false}
```

这不是最终通过状态。

### 12.3 第二阶段恢复

保持相同 manifest、seed、任务、任务概率、batch 和 optimizer 语义，只修改：

```yaml
paths:
  resume_checkpoint: /path/to/smoke-outputs/tokenlight_fixed_smoke_v1/checkpoints/step_XXXXXXXXX

train:
  max_steps: <大于第一阶段 global_step>
```

再次运行同一个入口：

```bash
python lumina_next_t2i/train_tokenlight.py \
  --config /path/to/tokenlight_smoke.yaml
```

恢复门会检查：

- checkpoint 正是第一阶段 summary 记录的 checkpoint
- `global_step`、`samples_seen` 和 optimizer state 已恢复
- manifest SHA256 与 resume signature 一致
- 恢复后的第一个 `(sample_index, sample_seed)` 是未消费的下一项
- 恢复后能继续完成有限 loss 的 optimizer update

全部通过后才会写：

```json
{"status": "pass", "train_ready": true, "resume_verified": true}
```

任一失败只会写 `train_ready: false` 或不产生通过结果，旧数据版本的 summary 不能复用。

## 13. 正式训练与评测

Smoke 通过后，恢复正式 `max_steps`、worker、DDP 和独立 `logging.run_id`。正式训练第一次从上游 Lumina 权重开始：

```yaml
paths:
  resume_checkpoint: null

train:
  smoke:
    enabled: false
```

单卡入口：

```bash
python lumina_next_t2i/train_tokenlight.py \
  --config lumina_next_t2i/config.yaml
```

项目封装的 DDP 入口：

```bash
bash lumina_next_t2i/train.sh
```

正式恢复只能使用相同数据与训练签名的 TokenLight checkpoint。修改 manifest、任务、概率、resolution、batch、world size 或 lighting/flow 配置后，应开始新 run。

评测：

```bash
python lumina_next_t2i/evaluate_tokenlight.py \
  --config lumina_next_t2i/config.yaml
```

当前入口提供按任务汇总的 PSNR、SSIM 和可选 LPIPS。它们不能替代位置轨迹、强度单调性、颜色误差和 mask 局部性评测；这些控制性指标需要固定 sweep 后才能声称模型学会可控光照。

## 14. 测试命令

以下命令用于服务器或具备对应依赖的开发环境。生成数据和训练前应依次执行，但本仓库不会在缺少用户 GPU、checkpoint 和真实 manifest 时伪造通过结果。

LightConstruction 单元测试：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

TokenLight smoke gate 单元测试：

```bash
python -m unittest discover -s Lumina-T2X/tests -p "test_*.py"
```

Python 语法检查：

```bash
python -m compileall src scripts Lumina-T2X/lumina_next_t2i Lumina-T2X/tools
```

真实 Blender 小批次、组件 validator、全任务 Dataset inspection 和两阶段训练 smoke 是发布前必须保留输出的集成测试，不能由单元测试代替。

## 15. 常见阻断条件

### 全部对象进入 quarantine

检查：

- `OBJECT_ROOT` 是否指向真实 Objaverse 根目录
- `category_dimensions.yaml` 是否覆盖实际类别
- `object.license_allowlist` 是否使用 `object.json` 中的规范名称
- object override 是否意外设置 `disabled: true`

### render job 为零

检查 annotation digest 是否过期、对象类别是否存在 targets、scene entity 是否仍存在、归一化 blend 是否在配置路径中。

### 所有 fixture 都进入 fallback

检查 `m4.fixture_categories` 是否覆盖 scene category、真实灯具 entity 是否在固定相机视锥内，以及可见像素是否超过阈值。不要通过无条件降低阈值把遮挡或出视锥灯具标记为真实 fixture。

### 环境光双计数

point、diffuse 和 fixture component 必须在 World strength 为零时渲染。若修改 renderer，重新检查 `component - dark`，不能把 ambient 包进 component。

### Validator 报 split leakage

确认 `data.split_profile` 与要声明的泛化轴一致。只有少量 Bistro scene 时应使用并明确标注 `object-held-out`，不能声称 scene generalization。

### Smoke 无法覆盖全部任务

固定 sampler 序列没有采到所有任务。增加固定 manifest 的有效组件、提高 smoke 样本数或更换 seed，然后冻结新的 manifest/seed；不能删除任务覆盖断言。

### Resume 被拒绝

这是安全行为。检查 manifest digest、任务和概率、batch、world size、lighting/flow 配置以及 checkpoint 是否来自同一个 smoke run。

## 16. 可声明结果的边界

完成本 README 的数据链路和 smoke 后，可以声明：

- annotation 能确定性地产生 TokenLight 组合样本
- 数据满足 reader、分量、坐标、fixture、split、lineage 和许可证技术契约
- 固定小 manifest 能执行训练更新并精确恢复

仍不能仅据此声明：

- 数据分布足以代表所有对象、场景或真实图像域
- 模型已学会位置、强度、颜色和 fixture 局部控制
- 数据、checkpoint 或生成结果已获得特定用途的法律批准
- 小批次通过等价于大规模渲染稳定

这些结论需要代表性覆盖报告、规模化 soak、固定控制评测和责任方批准。
