# TokenLight-Lumina 使用指南

本文只说明 Lumina-T2X 一侧如何消费 LightConstruction 生成的数据、完成验证、训练 smoke、恢复和正式训练。完整的 `object/scene → annotation_construction.json → Blender 组合渲染 → TokenLight` 命令链以仓库根目录的 [`README.md`](../../README.md) 为准。

## 1. 数据契约

正式组合数据根目录应至少包含：

```text
<dataset_root>/
  components/<job_id>/
    ambient.exr
    dark.exr
    point_lights/*.exr
    diffuse/*.exr
    in_scene_lights/fixture_000_on.exr
    in_scene_lights/fixture_000_mask.png
    metadata.json
  manifests/
    train.jsonl
    validation.jsonl
    test.jsonl
  dataset_release.json
  validation_summary.json
  validation_errors.jsonl
```

每条 metadata/manifest 记录必须携带：

- `camera` 和 `canonical` 坐标约定；
- `composition` 的 place/replace target、对象变换和可见性结果；
- `base_scene_id` 与 `base_scene_fingerprint`；
- annotation、object、scene、geometry、config digest；
- 对象与基础场景的许可决策；
- `fixture_source`，其值只能是 `scene_native` 或 `procedural_fallback`。

`in_scene_light` 使用 real-first/fallback 规则：渲染范围内有可见真实灯具时使用真实灯具 mesh，不额外添加球形 fixture；没有合格真实灯具时才创建程序化球形 fixture。两种来源都要生成独立 `_on.exr` 和可见区域 mask，统计时必须分开报告。

Dataset 在线组合公式为：

```text
ambient_scale  = dark + (ambient - dark) * scale
global_diffuse = ambient + diffuse_component - dark
add_light      = ambient + max(point_component - dark, 0) * color * intensity
in_scene_light = ambient + max(fixture_on - dark, 0) * color * intensity * transition
```

这些运算在 linear RGB 中完成；不要把 fixture `_on.exr` 直接当成最终 GT。

## 2. 服务器前置条件

权重放在仓库外。操作者必须显式提供实际路径，代码不猜测：

```text
/path/to/models/
  Lumina-Next-T2I/
    model_args.pth
    consolidated.00-of-01.safetensors
  sdxl-vae/
    config.json
    diffusion_pytorch_model.safetensors
```

上游 checkpoint 必须与 Lumina-Next-T2I 2B 单 shard 结构匹配；首次加载只允许新增加的 `lighting_encoder` 和 `fixture_mask_embedder` 参数缺失。

进入训练环境后检查：

```bash
cd /path/to/lightconstruction/Lumina-T2X
conda activate tokenlight
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import diffusers, fairscale, cv2, yaml; print('dependencies ok')"
nvidia-smi
```

## 3. 配置数据与任务

编辑 `lumina_next_t2i/config.yaml`：

```yaml
paths:
  upstream_checkpoint: /path/to/models/Lumina-Next-T2I
  vae: /path/to/models/sdxl-vae
  dataset_root: /path/to/outputs/tokenlight_dataset
  render_output_root: /path/to/outputs/tokenlight_dataset
  train_manifest: /path/to/outputs/tokenlight_dataset/manifests/train.jsonl
  validation_manifest: /path/to/outputs/tokenlight_dataset/manifests/validation.jsonl
  test_manifest: /path/to/outputs/tokenlight_dataset/manifests/test.jsonl
  output_root: /path/to/tokenlight-runs
  resume_checkpoint: null

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
  inspect_seed_search_limit: 4096

model:
  fixture_mask_enabled: true
```

`object-held-out` 保证 `asset_uid` 不跨 split，但允许复用 base scene，只能据此声明对象泛化。`scene-held-out` 保证 `base_scene_id` 不跨 split，至少需要三个不同基础场景；不足时构建 manifest 必须失败。

## 4. 构建 manifest、验证和 Dataset smoke

从 `Lumina-T2X/` 运行：

```bash
python tools/tokenlight_data/build_manifests.py --config lumina_next_t2i/config.yaml
python tools/tokenlight_data/validate_components.py --config lumina_next_t2i/config.yaml
python tools/tokenlight_data/inspect_dataset.py --config lumina_next_t2i/config.yaml
```

`build_manifests.py` 按所选 profile 分组切分，并写出带 manifest hash、lineage、许可策略和 fixture 来源计数的 `dataset_release.json`。

`validate_components.py` 检查：

- 所有文件存在，EXR shape 一致且没有 NaN/Inf；
- point/diffuse/fixture contribution 非退化；
- camera、canonical、transform 和灯光位置向量有效；
- fixture mask 非空，真实 fixture 带 entity ID，fallback 不伪造 entity ID；
- composition、fingerprint、lineage 和许可决策完整；
- 所有启用任务至少有一个 eligible scene；
- split 没有对应 profile 的泄漏。

`inspect_dataset.py` 使用确定性 `(index, sample_seed)` 搜索并实际读取所有启用任务。三步任一步失败时，不得进入训练 smoke。

## 5. 固定 manifest 训练 smoke

Dataset 可读不等于数据已经 `train-ready`。正式训练前必须在服务器上使用现有 `train_tokenlight.py` 完成两阶段 smoke；不另建训练入口。

### 5.1 操作者必须先提供

- 可用 CUDA GPU 和正式训练对应的 Python/Flash Attention 环境；
- Lumina 上游 checkpoint 和 VAE 的绝对路径；
- 已通过数据验证的固定小 dataset 与 train/validation manifest；
- 独立 smoke `output_root` 和 `logging.run_id`。

固定小 manifest 必须让确定性 sampler 序列实际覆盖全部启用任务，并至少包含有效 fixture mask；有 `scene_native` 时必须覆盖它，有 fallback 时也应覆盖 `procedural_fallback`。

### 5.2 第一阶段

建立服务器本地 smoke 配置：

```yaml
paths:
  upstream_checkpoint: /operator/provided/Lumina-Next-T2I
  vae: /operator/provided/sdxl-vae
  dataset_root: /operator/provided/fixed-small-dataset
  train_manifest: /operator/provided/fixed-small-dataset/manifests/train.jsonl
  validation_manifest: /operator/provided/fixed-small-dataset/manifests/validation.jsonl
  output_root: /operator/provided/smoke-output
  resume_checkpoint: null

data:
  num_workers: 0
  tasks: [ambient_scale, global_diffuse, add_light, in_scene_light]

train:
  micro_batch_size: 1
  gradient_accumulation_steps: <覆盖固定任务序列所需值>
  max_steps: <小的正整数>
  checkpoint_every_steps: 1
  smoke:
    enabled: true
    required_tasks: [ambient_scale, global_diffuse, add_light, in_scene_light]
    summary_json: smoke_summary.json

runtime:
  gpu_ids: [0]

logging:
  run_id: tokenlight_fixed_smoke_v1
```

运行：

```bash
python lumina_next_t2i/train_tokenlight.py --config /path/to/tokenlight_smoke.yaml
```

第一阶段必须至少完成一次 forward、backward 和 optimizer update，并断言：optimizer state 的 step 在更新前后实际递增；总 loss、所有启用任务 loss 和 grad norm 有限；lighting token 实际进入 batch；`in_scene_light` 的 fixture mask 非空且进入模型；checkpoint 保存了 optimizer、RNG、sampler 和 dataloader 状态。

此时 summary 只能处于等待恢复状态，例如：

```json
{"status": "awaiting-resume", "train_ready": false}
```

### 5.3 第二阶段恢复

保持 manifest、seed、任务概率、batch、optimizer、run ID 和输出目录不变，只把：

```yaml
paths:
  resume_checkpoint: /path/to/first-stage/checkpoints/step_XXXXXXXXX
train:
  max_steps: <大于第一阶段 global_step>
```

再次调用同一入口。恢复门会核对 resume signature、manifest digest、`global_step`、`samples_seen`、optimizer state，以及恢复后的首个 `(sample_index, sample_seed)` 是否正是未消费的下一项，并继续完成有限 loss 的 optimizer update。

只有两阶段全部通过，summary 才能写出：

```json
{"status": "pass", "train_ready": true, "resume_verified": true}
```

任何检查失败、未运行，或环境/GPU/checkpoint 尚未由操作者提供时，数据版本必须保持 `not-ready`/`unverified`。

## 6. 正式训练、恢复与评估

Smoke 通过后，使用独立正式 run ID，恢复正式步数、workers 和 DDP 配置，并关闭 smoke：

```yaml
paths:
  resume_checkpoint: null
train:
  smoke:
    enabled: false
logging:
  run_id: tokenlight_formal_v1
```

单卡可以直接运行：

```bash
python lumina_next_t2i/train_tokenlight.py --config lumina_next_t2i/config.yaml
```

项目封装的单卡/DDP 入口是：

```bash
bash lumina_next_t2i/train.sh
```

精确恢复时，`paths.resume_checkpoint` 指向完整 checkpoint 目录，而不是其中某个权重文件。修改 manifest、seed、任务、概率、resolution、batch、world size 或关键 lighting/flow 配置后，resume signature 应拒绝继续；这种情况应新建 run。

选定 checkpoint 后评估：

```bash
python lumina_next_t2i/evaluate_tokenlight.py --config lumina_next_t2i/config.yaml
```

正式 test 必须使用组合渲染前就已隔离的 `test.jsonl`。当前评估提供逐任务 PSNR、SSIM 和可选 LPIPS。

## 7. 仓库级测试命令

这些命令是操作者在相应环境中的验收步骤；文档更新本身不代表它们已经执行：

```bash
# LightConstruction
python -m unittest discover -s tests -v

# Lumina-T2X
cd Lumina-T2X
python -m unittest discover -s tests -v

# 数据产出后的真实数据门
python tools/tokenlight_data/validate_components.py --config lumina_next_t2i/config.yaml
python tools/tokenlight_data/inspect_dataset.py --config lumina_next_t2i/config.yaml
```

训练 smoke 仍必须使用第 5 节的服务器环境、GPU、真实 checkpoint 和固定小 manifest，不能由普通 CPU 单元测试替代。

## 8. 常见失败

- `OpenCV 无法读取 EXR`：OpenCV 构建需要 OpenEXR codec，文件必须是有限 linear RGB。
- `configured_task_has_no_eligible_scene`：启用了任务，但正式 manifest 没有对应分量；修数据，不要静默调低任务概率。
- `fixture_source 非法`：真实灯具必须带 entity ID；程序化 fallback 的 entity ID 必须为空。
- `resume signature mismatch`：数据或训练语义已变化，应开启新 run。
- `loss NaN/Inf`：先重新跑组件验证，再检查曝光、VAE、上游 checkpoint、学习率和 grad norm。
- CUDA OOM：先在 256/512 分辨率验证链路；960 px 保持 micro batch 1，并核对 BF16、Flash Attention 和 activation checkpointing。

实现假设见 [`ASSUMPTIONS.md`](ASSUMPTIONS.md)。任何 release 都应记录实际输入 digest、配置、环境、checkpoint 和执行结果；未通过真实 Blender 数据门与两阶段训练 smoke 的版本不能标记为 `train-ready`。
