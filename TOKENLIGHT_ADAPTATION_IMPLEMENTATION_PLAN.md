# annotation_construction.json 到 TokenLight 数据合成适配实施计划

建议分 7 个阶段修改。原则是先打通一条最小可用链路，再扩展真实灯具、切分和正式发布能力；尽量保持 `TokenLightDataset` 不变，把适配集中在数据生产侧。

## 阶段 1：冻结数据契约

修改：

- `src/lightconstruction/schemas.py`
- `src/lightconstruction/config.py`
- `configs/default.yaml`
- `configs/category_dimensions.yaml`
- `configs/object_overrides.yaml`
- `configs/scene_overrides.yaml`

新增三类中间数据结构：

### 1. `PreparedGeometry`

- `object_uid`
- 规范化尺度和朝向
- 原始、规范化资产路径及 digest
- 包围盒、接触面
- 配置来源：category default 或 object override

### 2. `RenderJob`

- `job_id`
- `annotation_id`
- `base_scene_id`
- `relation_type`
- 目标 `entity_id`
- 插入对象 transform
- 固定 camera
- lighting profile
- fixture 决策配置
- 全部输入 digest

### 3. `TokenLightSampleMetadata`

- `ambient/dark/point_lights/diffuse`
- `camera/canonical`
- `composition`
- `fixture_source`
- `lineage`
- `license`
- `base_scene_fingerprint`

同时明确坐标约定：

- transform 使用世界坐标和米制单位
- `renderer_position` 是 Blender 世界坐标
- `position` 是 TokenLight 使用的相机相对 canonical 坐标
- 矩阵长度、顺序和旋转约定固定

验收条件：缺失尺度、朝向、坐标空间或 digest 的记录必须直接失败，不能使用隐式猜测值。

## 阶段 2：实现 M4 CLI 和几何准备

修改：

- `src/lightconstruction/cli.py`

建议新增：

- `src/lightconstruction/geometry_prepare.py`
- `src/lightconstruction/render_jobs.py`

实现命令：

```text
lightconstruction prepare-geometry
lightconstruction build-render-jobs
lightconstruction render
```

`prepare-geometry` 负责：

- 读取 `object.json`
- 应用类别尺寸配置
- 应用对象级 override
- 计算规范化尺度、朝向和包围盒
- 将无法确定尺度或方向的对象放入 quarantine

`build-render-jobs` 负责：

- 联合读取：
  - `annotation_construction.json`
  - `object.json`
  - `scene.json`
  - prepared geometry
- 解析 place/replace
- 确定目标实体
- 生成固定且可复现的组合 transform
- 选择固定相机
- 生成确定性 `job_id`
- 写出 `render_jobs.jsonl`

验收条件：相同输入和配置重复执行，job 顺序、ID、transform 和 digest 完全一致。

## 阶段 3：实现场景组合 renderer

建议新增：

- `Lumina-T2X/tools/tokenlight_data/render_compositions.py`

不要直接复用 `render_assets.py` 的整体场景归一化逻辑，只复用其中的：

- EXR 输出规则
- canonical 坐标计算
- 灯光组件 metadata 格式
- fixture mask 渲染方式

组合 renderer 应负责：

1. 加载 Bistro `.blend`
2. 记录基础场景 fingerprint
3. 固定相机
4. 执行 place 或 replace
5. 检查对象可见性、碰撞和接触关系
6. 渲染 TokenLight 分量
7. 写出 metadata
8. 恢复场景或重新加载干净场景

`replace` 需要记录：

- 被替换实体
- 原 transform
- 原可见状态
- 新对象 transform

`place` 不得改变未授权的场景实体。

验收条件：连续渲染多个 job 后，基础场景 fingerprint 不变；前一个样本的修改不能污染后一个样本。

## 阶段 4：冻结分量光照和灯具策略

修改：

- `Lumina-T2X/tools/tokenlight_data/render_compositions.py`
- `Lumina-T2X/lumina_next_t2i/config.yaml`
- `Lumina-T2X/lumina_next_t2i/TOKENLIGHT_GUIDE.md`

为每种输出冻结光照状态：

| 分量 | World | 原生解析灯 | 自发光材质 | 当前受控灯 |
|---|---:|---:|---:|---:|
| ambient | 开 | 关 | 按 profile | 关 |
| dark | 关 | 关 | 关 | 关 |
| point component | 关 | 关 | 关 | 单灯开 |
| diffuse component | 关 | 关 | 关 | 对应 diffuse 开 |
| fixture-on | 关 | 关 | 关 | fixture 对应灯开 |

这样才能保持现有公式成立：

```text
global_diffuse = ambient + diffuse_component - dark
add_light = ambient + max(point_component - dark, 0)
```

### `in_scene_light` 决策

固定场景组合和相机后：

1. 收集真实灯具候选。
2. 检查候选是否：
   - 在相机视锥内
   - 有足够可见像素
   - 能生成有效 mask
   - 可绑定到解析灯或自发光贡献
3. 若有合格候选：
   - 选择一个确定性候选
   - `fixture_source = "scene_native"`
   - 记录 `fixture_entity_id`
   - 不创建球形 fixture
4. 若没有合格候选：
   - 创建程序化球形 fixture
   - `fixture_source = "procedural_fallback"`

候选选择必须有稳定排序，例如：

```text
可见面积降序 → 距画面中心升序 → entity_id 字典序
```

验收条件：

- 有合格真实灯具时绝不出现球体
- 无合格灯具时一定进入 fallback
- 遮挡、出视锥、无法绑定贡献的灯具不能冒充真实 fixture
- 两种来源分别统计

## 阶段 5：扩展严格验证

修改：

- `Lumina-T2X/tools/tokenlight_data/validate_components.py`
- `Lumina-T2X/tools/tokenlight_data/inspect_dataset.py`

建议新增：

- `tests/test_render_job_schema.py`
- `tests/test_manifest_validation.py`
- `tests/test_fixture_policy.py`

校验分成四层：

### 1. Schema 校验

- camera、canonical、transform
- 坐标空间和数组长度
- entity 和 relation
- fixture source
- digests、版本和 provenance

### 2. 图像分量校验

- EXR shape、NaN、Inf
- ambient/dark 差异
- point/diffuse contribution
- fixture mask 非空且与贡献区域对齐

### 3. 场景组合校验

- 插入对象可见
- place 接触关系合理
- replace 目标正确
- base fingerprint 未被污染

### 4. 真实 reader 冒烟测试

- 用生成样本实例化 `TokenLightDataset`
- 分别强制采样所有启用任务
- 检查 tensor、token、mask 和 target
- 配置启用了某任务但没有合格样本时直接失败

这一阶段原则上不改 `tokenlight/dataset.py`；只有 reader 冒烟测试证明现有格式无法表达所需语义时才修改。

## 阶段 6：切分、溯源和许可证

修改：

- `Lumina-T2X/tools/tokenlight_data/build_manifests.py`
- `Lumina-T2X/tools/tokenlight_data/validate_components.py`

提供两个明确的切分 profile：

### `object-held-out`

- `object_uid` 跨 split 交集必须为空
- 可以复用 base scene
- 不能声称场景泛化

### `scene-held-out`

- `base_scene_id` 跨 split 交集必须为空
- 场景数量不足时拒绝生成，而不是退化为对象切分

最终 manifest 强制包含：

- annotation/object/scene digest
- schema/config/generator 版本
- base scene 来源与 fingerprint
- Objaverse UID、作者、来源和许可证
- Bistro license 和 attribution
- 许可证白名单决策
- `fixture_source`

任何 digest 不匹配、来源不完整或许可证不允许的样本都不能进入正式 split。

## 阶段 7：小规模端到端验收

先从服务器数据中选择一个小批次，至少覆盖：

- `place`
- `replace`
- 有真实灯具的场景
- 无真实灯具、需要球形 fixture 的场景
- 遮挡或不可用真实灯具
- category 默认几何
- object override
- 至少两个不同 base scene

完整执行：

```text
annotation_construction.json
        ↓
prepare-geometry
        ↓
build-render-jobs
        ↓
Blender composition render
        ↓
validate-components
        ↓
build-manifests
        ↓
TokenLightDataset smoke test
        ↓
train_tokenlight.py fixed-manifest training smoke
        ↓
checkpoint resume verification
```

### 最终训练验收门

`TokenLightDataset` 能读取样本只证明格式兼容，不能把数据版本标记为 `train-ready`。在批量生成或正式训练前，必须使用现有的 `Lumina-T2X/lumina_next_t2i/train_tokenlight.py` 完成固定小 manifest 的 mini-overfit/训练冒烟；不增加第二个训练入口。

#### 操作者前置条件

训练冒烟只能在服务器环境执行。运行前由操作者显式提供并核对：

- 可用的 CUDA GPU 和与正式训练一致的 Python/Flash Attention 环境
- Lumina-Next-T2I 上游 checkpoint 路径
- VAE 路径
- 固定小数据集根目录
- 固定 train/validation manifest 路径
- 独立的 smoke `output_root` 和 `logging.run_id`
- 单进程、单 GPU 运行方式，以便精确验证 sampler 与 checkpoint 恢复位置

这些环境、GPU 和 checkpoint 路径不得由代码猜测，也不得把显存检查、smoke 和正式训练写入同一个 run 目录。

#### 固定 smoke 数据与配置

固定小 manifest 应来自阶段 7 已通过组件验证的样本，并满足：

- 覆盖配置中所有启用任务：`ambient_scale`、`global_diffuse`、`add_light`、`in_scene_light`
- `in_scene_light` 至少包含一个有效 fixture mask，并覆盖 `scene_native`；若该批数据存在 fallback，也覆盖 `procedural_fallback`
- manifest、所有组件文件及训练配置均记录 SHA256
- 固定 `train.seed`、`data.validation_seed`、任务概率、样本数和 manifest 顺序
- `data.num_workers: 0`，减少 smoke 期间的异步变量
- `micro_batch_size: 1`，使用足以执行至少一次 optimizer update 的固定 `gradient_accumulation_steps`
- `checkpoint_every_steps` 小于或等于 smoke 的 `max_steps`，确保第一段运行一定产生 checkpoint

现有 `ResumableRandomSampler` 会从固定 seed、样本位置和 dataset size 生成确定性的 `(dataset_index, sample_seed)`。在冻结 smoke manifest 前，应先记录该序列实际产生的任务；只有序列覆盖全部启用任务，才能把 manifest、seed 和步数作为正式 smoke fixture 固定下来。任务覆盖不能依靠概率上的“应该出现”。

#### `train_tokenlight.py` 需要增加的 smoke 断言

在现有训练入口中增加由配置显式启用的 smoke 断言，不建立平行训练循环：

- 每次 optimizer update 前后参数或 optimizer state 确实发生变化
- `loss_total`、每个已启用任务的 loss 和 `grad_norm` 均为有限值
- 完成 smoke 后，每个启用任务的累计 `task_count` 都大于零
- 每个 batch 的 `lighting_values`、`lighting_known` 和 `lighting_valid` 已传入模型，并且有效 lighting token 数量非零
- `in_scene_light` batch 的 `fixture_present` 为真、fixture mask 非空，且 mask 已传入模型
- 非 `in_scene_light` batch 不得错误标记 fixture 为存在
- 断言结果写入独立的机器可读 smoke summary，包含 run ID、manifest/config digest、逐任务计数和 loss、checkpoint 及恢复结果

这些断言只在显式 smoke 配置下启用，避免改变正式训练的性能和采样语义；普通训练继续使用同一个 `train_tokenlight.py --config ...` 入口。

#### 两段式 checkpoint 恢复验证

训练冒烟分为两个连续运行：

1. 第一段从上游 checkpoint 启动，完成至少一次 forward、backward 和 optimizer update，保存 TokenLight checkpoint，并记录 `global_step`、`samples_seen`、dataloader state 和 resume signature。
2. 第二段仍调用同一个 `train_tokenlight.py`，仅把 `paths.resume_checkpoint` 指向第一段输出并提高 `max_steps`。恢复后检查：
   - `global_step` 从保存值继续，而不是归零
   - `samples_seen` 与 checkpoint 一致
   - `validate_dataloader_resume` 接受当前 manifest、配置和 sampler state
   - 恢复后的第一个 `(dataset_index, sample_seed)` 是未消费的下一项
   - optimizer state 已恢复，并且能继续完成下一次有限 loss 的 optimizer update

两段都使用现有入口：

```bash
python train_tokenlight.py --config /path/to/tokenlight_smoke.yaml
```

第一段配置保持 `paths.resume_checkpoint: null`；第二段保持相同的 `output_root`、`logging.run_id`、manifest、seed 和训练语义配置，只设置第一段 checkpoint 并提高 `max_steps`。`train.smoke.enabled` 必须为 `true`，`train.smoke.required_tasks` 必须与 `data.tasks` 完全一致，`data.num_workers` 必须为 `0`，且 `runtime.gpu_ids` 只能包含一个 GPU。

修改 manifest、任务列表、任务概率、关键数据配置或 world size 后，resume signature/dataloader state 校验必须拒绝恢复，不能静默从错误位置继续。

#### `train-ready` 决策

数据版本只有同时满足以下条件才能标记为 `train-ready`：

- 数据生产与 `TokenLightDataset` smoke 全部通过
- 固定序列覆盖所有启用任务
- 所有任务 loss、总 loss 和 grad norm 均为有限值
- lighting token 与 fixture mask smoke 断言全部通过
- checkpoint 保存和第二段恢复验证通过
- smoke summary 中的 manifest/config digest 与待发布数据版本一致

任一检查失败、未运行，或者服务器环境、GPU、checkpoint 路径尚未由操作者提供时，状态必须保持 `not-ready` 或 `unverified`；不得创建成功标记，也不得沿用旧数据版本的 smoke 结果。

只有以下条件全部满足后再批量生成：

- 所有启用任务都有实际样本
- real fixture 与 fallback 数量可分别报告
- 没有环境光双计数
- 没有场景状态污染
- split 交集符合对应 profile
- 任一训练样本都能反查完整输入和许可证
- 相同 job 重跑得到相同 metadata 和几何状态
- 固定小 manifest 的训练冒烟完成至少一次 forward、backward 和 optimizer update
- 所有启用任务都有有限 loss，且 lighting token/fixture mask 断言通过
- checkpoint 保存与下一样本位置的恢复验证通过
- 当前数据版本具有 digest 匹配的 `train-ready` smoke summary

## 建议的首批实现范围

第一批完成阶段 1–5，先产出一个能够被 `TokenLightDataset` 真实读取的最小样本；切分、许可和大规模调度在这条纵向链路通过以后实施。但 `TokenLightDataset` 可读仅是中间里程碑，在上述固定 manifest 训练与恢复门通过前，任何数据版本都不能标记为 `train-ready`。
