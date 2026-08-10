# 基于 Lumina-Next-T2I 复现 TokenLight 的实施计划

## 0. 协作边界与开始条件

以下准备工作由用户完成：

- 下载或 clone Lumina-T2X 代码；
- 创建并配置 Python/Conda 环境；
- 安装 PyTorch、CUDA 相关依赖和 flash-attn；
- 下载 Lumina-Next-T2I、VAE 等预训练权重；
- 确保原始 Lumina 推理所需文件可用。
- 负责后续所有测试、训练验证、推理验证和阶段验收。

代码修改阶段不执行上述下载、安装或环境创建操作。开始修改前，用户需要提供：

```text
Lumina-T2X 项目绝对路径:/mnt/afs_fangwenqi/Lumina-T2X
需要使用的 Conda 环境名称或 Python 解释器路径:conda环境名lum，python解释器就在conda环境中
Lumina-Next-T2I checkpoint 路径
配套 VAE 路径（如果不包含在 checkpoint 目录中）
可使用的 GPU 编号：0
```

收到这些信息后，代码修改工作的范围是：

- 阅读并审计用户准备好的 Lumina 代码；
- 新增 TokenLight 模型、数据、训练、推理和评测模块；
- 修改必要的 Next-DiT 接口；
- 编写配置和文档；
- 提供建议的测试命令、预期输入输出和关键 tensor shape，但不编写或运行测试；
- 不覆盖用户已有修改，不擅自升级或重装依赖。

### 0.1 测试职责约定

本计划中后续出现的“测试”“验证”“通过条件”和“验收门槛”均作为用户自行执行的检查清单，不属于代码修改工作的执行范围。代码修改阶段：

- 不新增或修改 `tests/` 下的测试代码；
- 不运行单元测试、GPU 测试、训练测试或推理测试；
- 不执行单样本过拟合、回归训练和正式效果评测；
- 实现完成后明确标注未经执行测试，并向用户提供建议的测试方法。

## 1. 目标与边界

### 1.1 最终目标

基于公开的 `Alpha-VLLM/Lumina-Next-T2I` 和 Next-DiT 2B，实现一个可训练、可推理、可评测的 TokenLight 复现版本，支持：

1. `ambient_scale`：全局环境光强度控制；
2. `global_diffuse`：全局阴影软硬控制；
3. `add_light`：附加虚拟光源的位置、颜色、强度和扩散控制；
4. `in_scene_light`：可见场景灯具的 mask、颜色、强度和开关控制；
5. 多光源组合；
6. 只对光照条件做 classifier-free guidance；
7. 线性 RGB 光照分量的在线组合；
8. 独立、可复现的定量评测。

### 1.2 实现名称

由于 TokenLight 论文没有公开其 Adobe 内部 text-to-video 预训练 checkpoint，本项目不能声称是权重级别的精确复现。建议项目和 checkpoint 使用明确名称：

```text
TokenLight
```

报告结果时需要区分：

- `paper-faithful method reproduction`：数据公式、token 定义、联合注意力、flow matching、CFG 和训练设置尽量对齐；
- `backbone substitution`：用公开的 Lumina Next-DiT 2B 替代论文未公开的 text-to-video DiT。

### 1.3 第一阶段不做的内容

- 不把 Objaverse object 自动语义摆放到复杂室内 scene；
- 不训练新的 VAE；
- 不先做 LoRA 并把 LoRA 结果当作论文复现；
- 不在旧 `latent-diffusion` Python 3.8 / PyTorch 1.7 环境中继续堆叠实现；
- 不在模型最小验证通过前渲染 1000 个 object 的 960 px 正式数据。

## 2. 固定的技术决策

| 项目 | 决策 |
| --- | --- |
| 上游仓库 | `https://github.com/Alpha-VLLM/Lumina-T2X` |
| 目标子项目 | `lumina_next_t2i` |
| 初始权重 | `Alpha-VLLM/Lumina-Next-T2I` |
| 主干 | Next-DiT 2B |
| VAE | Lumina checkpoint 配套的 SDXL VAE，冻结 |
| 文本编码器 | TokenLight 训练路径中不使用；Lumina 原始推理基线保留 |
| 训练目标 | Linear-path flow matching，预测 velocity |
| 正式输入分辨率 | 960 x 960 |
| 调试分辨率 | 256，然后 512 |
| 数值精度 | BF16 |
| 正式训练方式 | 全参数微调；VAE 冻结 |
| CFG dropout | 10%，只丢弃 lighting edit tokens |
| CFG scale | 默认 2.0 |
| 优化器 | AdamW，`lr=1e-5`、`weight_decay=0.01`、`betas=(0.9, 0.95)` |
| 训练步数 | 15,000 optimizer steps |
| 目标全局 batch | 160 |
| 单卡策略 | micro-batch 1 起步，通过梯度累积得到有效 batch |
| 显存技术 | Flash Attention、activation checkpointing、冻结 VAE、BF16 |

所有上游代码和权重必须固定 revision，并在 `REPRODUCIBILITY.md` 中记录：

```text
Lumina-T2X git commit
Hugging Face model revision
CUDA driver/runtime
Python/PyTorch/flash-attn versions
GPU 型号
训练配置 SHA256
数据 manifest SHA256
```

## 3. 新项目建议结构

在新的 Lumina 仓库内新增独立模块，不把 TokenLight 逻辑散落到原始 T2I 路径：

```text
Lumina-T2X/
  lumina_next_t2i/
    models/
      model.py                    # 尽量少改的上游 Next-DiT
    tokenlight/
      __init__.py
      model.py                    # TokenLightNextDiT wrapper/variant
      tokens.py                   # 属性 schema 和 Fourier encoder
      dataset.py                  # 线性 RGB 分量组合
      collate.py                  # 变长 token/mask 和任务 batch
      flow.py                     # flow matching loss
      sampler.py                  # 50-step 推理与 lighting-only CFG
      checkpoint.py               # 权重迁移与新增参数初始化
      metrics.py                  # PSNR/SSIM/LPIPS/control metrics
    config.yaml                    # TokenLight 唯一参数配置入口
    train_tokenlight.py
    infer_tokenlight.py
    evaluate_tokenlight.py
    train.sh                       # 训练启动脚本
    infer.sh                       # 可输入光照参数的推理脚本
    TOKENLIGHT_GUIDE.md            # 从数据准备到正式运行的全流程指南
  tools/
    tokenlight_data/
      render_assets.py
      validate_components.py
      build_manifests.py
      inspect_dataset.py
```

原则：Lumina 原始 T2I 推理必须始终可以运行，以便判断回归来自环境、上游修改还是 TokenLight 分支。

### 3.1 单一配置文件

TokenLight 只新增一个配置文件：

```text
lumina_next_t2i/config.yaml
```

所有需要用户调整的参数统一放入该文件，包括：

```text
paths        # checkpoint、VAE、manifest、输出目录
data         # 分辨率、裁剪、曝光、采样和 worker
model        # Next-DiT、patch、最大灯数和 mask 设置
lighting     # token schema、Fourier encoding 和属性范围
flow         # 时间方向、prediction 和 loss
train        # batch、梯度累积、优化器、精度、步数和 checkpoint
infer        # seed、采样步数、solver、CFG 和输出设置
evaluate     # manifest、指标和结果目录
logging      # run_id、记录频率和日志目录
runtime      # device、GPU 编号、Flash Attention 和 checkpointing
```

`train_tokenlight.py`、`infer_tokenlight.py` 和 `evaluate_tokenlight.py` 均通过 `--config lumina_next_t2i/config.yaml` 读取同一份配置。除 `--config` 外不重复定义业务参数的命令行默认值；代码内部也不硬编码可调训练、数据或推理参数。缺少必需字段时应立即报错并指出完整字段路径。`infer_tokenlight.py` 允许接收 source、输出路径和本次光照编辑值，这些属于单次推理输入，不构成第二套持久配置或默认参数。

调试、512 px 验证和 960 px 正式运行通过修改同一份 `config.yaml` 中的分辨率及相关参数完成，不再维护三份独立 YAML。Lumina 原有的 `configs/infer/settings.yaml` 和 `configs/data/JourneyDB.yaml` 保持不变，仅供原始 T2I 路径使用。

### 3.2 训练、推理脚本与全流程指南

完成代码修改后必须提供：

```text
lumina_next_t2i/train.sh
lumina_next_t2i/infer.sh
lumina_next_t2i/TOKENLIGHT_GUIDE.md
```

`train.sh` 作为统一训练入口，默认读取 `lumina_next_t2i/config.yaml` 并调用 `train_tokenlight.py`。脚本需要正确设置工作目录、Python 入口和配置路径，不在 shell 中复制 YAML 已有的训练参数。断点续训 checkpoint 由 `config.yaml` 指定。

`infer.sh` 作为统一推理入口，默认读取同一个 `config.yaml` 并调用 `infer_tokenlight.py`。若现有代码没有可满足要求的推理 Python 入口，则实现 `infer_tokenlight.py`；不得为了复用而强行调用语义不匹配的原始文本生成脚本。

`infer.sh` 至少支持以下单次推理输入：

```text
source image
output path
task: ambient_scale | global_diffuse | add_light | in_scene_light
ambient scale
global diffuse
一个或多个 add-light: x,y,z,r,g,b,intensity,diffuse
fixture mask
fixture color: r,g,b
fixture intensity
fixture transition/on-off
```

多灯参数必须支持重复输入或读取结构化参数文件，不能只支持固定一盏灯。`infer.sh` 必须提供 `--help` 和清晰的参数错误信息。模型、VAE、checkpoint、采样步数、CFG、seed、精度、分辨率和 device 等持久参数仍从 `config.yaml` 读取；source、output 和光照编辑值由本次命令输入，并在输出 metadata 中完整保存以便复现。

`TOKENLIGHT_GUIDE.md` 必须按实际执行顺序提供从零开始的全流程说明，不能只罗列参数。至少覆盖：

1. 项目目录、环境、checkpoint、VAE、GPU 和数据路径准备；
2. `config.yaml` 每个 section 和必填字段说明；
3. 原始 Lumina T2I 基线由用户如何运行和记录结果；
4. 10-object smoke 阶段如何填写数据路径、使用 256 px 配置、生成或检查 manifest；
5. 数据处理脚本的输入目录、执行命令、输出目录和 manifest 结构；
6. 用户如何检查 linear RGB、EXR、source/target 对齐和数据完整性；
7. 如何运行 `train.sh`、观察 train/validation loss、定位日志和 checkpoint；
8. 如何停止并从 checkpoint 恢复训练；
9. 如何使用 `infer.sh` 分别执行 ambient、diffuse、单灯、多灯和 in-scene-light 推理；
10. 如何从 10-object/256 px 切换到 50-object/512 px；
11. 正式 960 px 训练前需要修改哪些配置字段及如何运行；
12. 正式 validation/test 如何修改 manifest、checkpoint、seed 和输出路径；
13. 如何运行评测、绘制 train/validation loss、导出 CSV 和查找 best checkpoint；
14. 常见错误处理，包括路径、checkpoint key、显存、NaN/Inf、EXR、shape 和断点续训问题；
15. 明确标注哪些验证由用户自行执行，以及每个阶段的预期产物和进入下一阶段的条件。

指南中的所有命令、参数名、路径和配置字段必须与最终代码一致。不得保留无法执行的占位命令；用户需要填写的机器相关路径使用明确占位符并逐项解释。

## 4. 模型架构设计

### 4.1 输入序列

TokenLight forward 接收：

```text
source image I
target image Ir（训练时）或 target noise（推理时）
flow time tau
lighting edit Delta L
optional fixture mask m
```

在 960 px 下，SDXL VAE 空间下采样 8 倍：

```text
RGB:                 [B, 3, 960, 960]
VAE latent:          [B, C, 120, 120]
2x2 patchify:        [B, 3600, 4C]
target tokens:       3600
source tokens:       3600
lighting tokens:     约 10 至 30
```

联合序列定义为：

```text
[noisy target patches]
[clean source patches]
[lighting attribute tokens]
[optional fixture-mask patches]
```

所有 token 必须在 Next-DiT block 中进行联合注意力。不能退化为旧项目中“target UNet 通过 cross-attention 读取 source/light”的实现。

### 4.2 Source 与 target patch

- Source 和 target 使用同一个冻结 VAE；
- Source 和 target 共享 patch embedding 权重；
- 对应空间位置使用相同的二维 RoPE 坐标；
- 只对 target patch 加 flow noise；
- 只从 target token 读取输出并 unpatchify 为 velocity；
- Source、light 和 mask token 不进入最终图像输出头；
- 不对 source 图像施加 CFG dropout。

需要首先审计 `lumina_next_t2i/models/model.py` 中的 `JointAttention`、patchify、RoPE 和 output head，然后选择最小侵入式实现。验收标准是 attention 内部真实的 Q/K/V 序列包含 target、source 和 light，而不是仅在 block 外做特征相加。

### 4.3 Lighting token schema

所有连续标量采用 Gaussian Fourier features：

```text
phi(x) = [sin(Bx), cos(Bx)]
B ~ N(0, sigma^2), sigma = 5
```

每个标量独立投影为一个 Next-DiT hidden-dimension token。

建议 schema：

| Task | Token |
| --- | --- |
| `ambient_scale` | `ambient` |
| `global_diffuse` | `global_diffuse` |
| `add_light` | 每盏灯的 `x,y,z,r,g,b,intensity,diffuse` |
| `in_scene_light` | `r,g,b,intensity,transition` + fixture mask patches |

多附加灯使用固定 slot，默认 `max_lights=3`。每个 slot 有 `valid` 标志；无效 slot 不能与数值零混淆。实现前需要再次依据 TokenLight 论文 PDF/补充材料确认最大灯数，若原文不是 3，以原文为准。

缺失属性使用显式 `known/valid mask`，不能只使用 `-1` 而没有有效位，因为 `-1` 可能处于合法连续控制范围。

### 4.4 Task token 与 modality embedding

论文公开描述的核心是 lighting attribute tokens，没有充分证据要求额外 task token 或 modality embedding。因此：

- 第一版不擅自增加 learned task token；
- task 由出现的 token 类型和 valid mask 表达；
- 第一版不增加 source/target modality embedding；
- 如果训练证明 source/target 歧义，再将 modality embedding 作为明确记录的 ablation，而不是默认宣称为论文设计。

### 4.5 Flow matching

训练目标：

```text
epsilon ~ N(0, I)
tau ~ Uniform(0, 1)
z_tau = (1 - tau) * epsilon + tau * z_target
velocity_target = z_target - epsilon
loss = MSE(model(z_tau, tau, source, Delta L), velocity_target)
```

必须确认 Lumina 原生 rectified-flow 时间方向与上述方向一致。若 Lumina 上游使用相反方向，只能在一个边界层完成映射，并为 `tau=0`、`tau=1` 编写单元测试，避免训练和采样方向相反。

### 4.6 权重初始化

- 加载完整 Lumina-Next-T2I Next-DiT 权重；
- target patch embed、Transformer blocks、time embed 和 output head从预训练权重加载；
- source patch 复用 target patch embed，不新建随机 source patch embed；
- lighting Fourier matrix固定随机种子，作为 buffer 保存进 checkpoint；
- 新增 light projections 使用与原模型线性层一致的初始化尺度；
- 加载时输出 missing/unexpected keys 完整清单；
- 只有预期新增的 lighting/mask 参数允许 missing；
- VAE 全程冻结并处于 eval 模式。

## 5. 数据契约

### 5.1 可直接用于最小验证的现有数据

当前 10-object smoke 数据：

```text
/mnt/afs_fangwenqi/latent-diffusion/data/tokenlight_test
```

其状态：

```text
8 train scenes
2 validation scenes
4 point-light positions per scene
3 diffuse levels per scene
256 px EXR
no HDRI
```

这些数据只用于 dataset、模型 forward、单 scene 过拟合和推理回归，不用于正式效果结论。

当前 Objaverse 子集：

```text
/mnt/afs_fangwenqi/data/Objaverse
```

其中本地有 1000 个 GLB；类别是否存在不影响第一阶段的单物体程序化场景渲染。

### 5.2 Manifest schema

每个 scene 至少包含：

```json
{
  "id": "scene_id",
  "asset": "relative/path.glb",
  "ambient": "components/scene_id/ambient.exr",
  "dark": "components/scene_id/dark.exr",
  "point_lights": [
    {
      "path": "components/scene_id/point_lights/light_000.exr",
      "position": [0.0, 0.0, 1.0],
      "base_energy": 500.0,
      "diffuse": 0.1
    }
  ],
  "diffuse": [
    {
      "path": "components/scene_id/diffuse/spread_00.exr",
      "level": 0.0,
      "size": 0.05
    }
  ],
  "camera": {},
  "canonical": {}
}
```

Train/validation/test 必须按 asset UID 切分，不能让同一 object 的不同视角跨 split。

### 5.3 线性 RGB 组合公式

记：

```text
D   = dark baseline
A   = ambient render
P_i = 第 i 个点光源 render
G_j = 第 j 个 diffuse render
```

Ambient：

```text
I  = A
Ir = D + (A - D) * a
```

Add light：

```text
I  = A
C_i = max(P_i - D, 0)
Ir = A + sum_i(C_i * color_i * intensity_i)
```

Global diffuse：

```text
I  = A + G_source - D
Ir = A + G_target - D
Delta d_g = level_target - level_source
```

组合后：

```text
linear RGB clamp-to-nonnegative
-> exposure
-> Reinhard tone mapping x / (1 + x)
-> resize/crop（source、target、mask 必须完全同步）
-> normalize to model range
```

禁止随机执行会破坏 source/target 对齐的几何增强。

注意：不能先分别把线性光照分量编码成 VAE latent 后再在线性组合，因为 VAE 是非线性的。必须先在线性 RGB 中组合目标图像，再进行 VAE 编码。

### 5.4 正式合成数据目标

第一版正式合成数据：

```text
objects:              先 1000 个本地 GLB，质量过滤后使用
resolution:           960 x 960
views_per_asset:      至少 2
point lights:         64 per scene
diffuse levels:       6
HDRI:                 目标约 600 张 PolyHaven HDRI
renderer:             Blender Cycles
format:               linear RGB OpenEXR
split:                by asset UID
```

推荐 diffuse spreads 初始候选：

```json
[0.05, 0.2, 0.4, 0.7, 1.0, 1.5]
```

最终数值应在少量场景中可视化确认阴影变化近似均匀后固定。

### 5.5 数据质量过滤

在正式渲染前自动检查：

- GLB 可导入且至少有一个 mesh；
- bounding box 非零且长宽高比例不过度异常；
- 相机画面非空，主体占比处于设定范围；
- 材质和纹理可加载；
- NaN/Inf 像素为零；
- EXR 为线性 RGB；
- dark/ambient/light 分量路径完整；
- 点光贡献不是全黑或全饱和；
- diffuse 六级具有可测量差异；
- ground contact 和主体朝向合理；
- 同一 scene 的所有分量分辨率、相机矩阵完全一致。

每个失败项写入机器可读 JSONL，不能静默跳过。

## 6. 分阶段实施与验收门槛

### Phase 0：接收并验证用户准备好的 Lumina 基线

任务：

1. 记录用户提供的 Lumina 项目路径、git commit 和工作区状态；
2. 记录用户提供的 Python、PyTorch、CUDA 和 flash-attn 版本；
3. 记录 checkpoint 与 VAE 路径和 revision；
4. 检查用户准备的原始 Lumina 是否已有可复现推理结果；
5. 向用户提供官方原始 T2I smoke test 的建议命令；
6. 由用户运行并保存输入 prompt、seed、配置、输出图和显存峰值；

通过条件：

- 原始 Lumina 权重严格加载；
- 固定 seed 可以重复生成；
- BF16 推理无 NaN/Inf；
- 未修改的官方路径仍能工作。

未通过前不得开始 TokenLight 模型改造。

### Phase 1：移植数据集，不改模型

任务：

1. 将现有 linear-component Dataset 移植到新仓库；
2. 支持 10-object manifest；
3. 实现四种 task schema；
4. 实现多光源组合；
5. 实现 known/valid masks；
6. 实现 deterministic validation sampling；
7. 保存可视化 source/target/delta 网格。

通过条件：

- `ambient a=1` 时 source 和 target 数值一致；
- `add_light intensity=0` 时 source 和 target一致；
- 多灯结果等于各线性贡献之和；
- source/target/mask 几何完全对齐；
- Dataset 重启后 validation 样本完全可复现；
- 所有单元测试通过。

### Phase 2：实现 lighting token encoder

任务：

1. 固定 Fourier feature schema；
2. 每个连续分量输出一个 token；
3. 支持最多 `max_lights` 个灯光 slot；
4. 支持 valid mask；
5. 支持 fixture mask latent token；
6. 实现 lighting-only dropout。

通过条件：

- token 数、顺序和有效位有快照测试；
- 相同输入输出完全确定；
- 不同 position/color/intensity 会改变对应 token；
- drop light 后 source token 完全不变；
- Fourier buffer被 checkpoint 保存并恢复。

### Phase 3：改造 Next-DiT 联合注意力

任务：

1. 支持 target/source/light/mask 联合序列；
2. 给 source/target 对应 patch 设置相同 RoPE 坐标；
3. 对属性 token 定义稳定的 position 处理；
4. 只输出 target velocity；
5. 从 Lumina checkpoint 加载已有权重；
6. 输出参数加载审计报告；
7. 启用 Flash Attention 和 activation checkpointing。

通过条件：

- 256 px、batch 1 forward/backward 成功；
- attention 中可确认 source/light token真实参与 Q/K/V；
- 输出 shape 与 target latent 一致；
- 除新增模块外没有意外 missing keys；
- source/light 梯度路径存在；
- BF16 forward/backward 无 NaN/Inf；
- 关闭 TokenLight 分支时 Lumina 原始推理不回归。

### Phase 4：Flow loss 与单样本过拟合

任务：

1. 实现论文 straight-path flow loss；
2. 对齐 Lumina time convention；
3. 在一个固定 scene、一个固定 edit 上训练；
4. 禁用随机数据采样；
5. 每隔固定步数生成相同 seed 的预测。

通过条件：

- 训练 loss 明显下降；
- 一个样本可以被高质量重建；
- 输出方向不是反向去噪；
- 改变 light attribute 能造成方向正确的可见变化；
- 不改变 source 结构和主体身份。

未完成单样本过拟合前，不启动大规模渲染或正式训练。

### Phase 5：采样器和 CFG

任务：

1. 实现与 flow 时间方向一致的 50-step sampler；
2. 无条件分支只删除 lighting edit；
3. source 和 fixture mask在两条 CFG 分支中一致；
4. 支持固定 seed attribute sweep；
5. 支持多灯推理。

CFG 定义：

```text
v = v_source_only + w * (v_source_and_light - v_source_only)
w = 2
```

通过条件：

- `w=1` 与纯 conditional forward 一致；
- 两个分支 source token逐元素一致；
- position sweep 产生连续移动的光照和阴影；
- intensity sweep 总体亮度单调变化；
- color sweep 不应明显改变非光照语义内容。

论文写作时采样器名称必须按实际实现记录。若无法严格复现论文所称 DDIM，不应把普通 Euler ODE sampler 标成 DDIM。

### Phase 6：10-object 回归训练

任务：

1. 在现有 8/2 scene 数据上训练短程模型；
2. 覆盖 ambient、diffuse、add-light；
3. 保存 last 和 top-k validation checkpoint；
4. 运行三种 control sweep；
5. 记录显存、吞吐和 checkpoint 恢复。

通过条件：

- 连续训练和 resume 后 loss/global step一致；
- checkpoint 能独立推理；
- 三种 task 均表现出正确控制方向；
- 训练显存和吞吐可预测；
- 没有把 10-object 结果作为正式论文效果结论。

### Phase 7：扩大程序化合成数据

按以下顺序扩容：

```text
50 objects x 512 px
-> 1000 objects x 512 px 质量审计
-> 选定高质量子集 x 960 px 正式渲染
-> 64 point lights + 6 diffuse levels + HDRI
```

每一级都要先检查：

- 渲染失败率；
- 数据体积；
- 单 scene 平均渲染时间；
- EXR 动态范围；
- 光照贡献有效率；
- 主体可见率；
- train/validation/test UID 泄漏。

禁止直接用 256 px 旧数据上采样到 960 px 作为正式训练数据。

### Phase 8：正式训练

推荐从以下配置开始：

```yaml
precision: bf16
resolution: 960
micro_batch_size: 1
gradient_accumulation_steps: 160
max_optimizer_steps: 15000
learning_rate: 1.0e-5
weight_decay: 0.01
betas: [0.9, 0.95]
cfg_drop_rate: 0.1
guidance_scale: 2.0
gradient_checkpointing: true
flash_attention: true
vae_trainable: false
ema: false
lr_scheduler: constant
```

注意：论文未明确要求 EMA 或学习率调度时，不应自行添加并宣称对齐。可以作为后续 ablation。

单卡 A100-80GB 的最终 micro-batch 需要通过实测确定。若 960 px 联合序列无法稳定 full fine-tune：

1. 先确认 Flash Attention 和 checkpointing 确实启用；
2. 减少 micro-batch 至 1；
3. 冻结 VAE 并在训练 step 外避免保留其计算图；
4. 使用 gradient accumulation；
5. 必要时使用 CPU optimizer offload；
6. 不以 LoRA 替代正式全参数复现，除非明确将结果标为 LoRA baseline。

### Phase 9：正式评测

至少建立：

1. 100 至 200 个完全留出的 Objaverse object；
2. 固定的单光位置测试集；
3. 六条灯光位置轨迹，每条若干连续位置；
4. 强度、颜色、diffuse 和 ambient 单变量 sweep；
5. 多灯组合测试；
6. 后续真实室内图像测试。

指标：

```text
PSNR
SSIM
LPIPS
object-mask PSNR/SSIM/LPIPS
input identity/content preservation
light-position trajectory consistency
intensity monotonicity
color-control error
```

评测脚本必须读取固定 manifest 和 seed，不能手工挑图。

### Phase 10：补齐 in-scene light 和真实数据

完整论文复现最后需要：

- 艺术家制作的室内 scene；
- 可见灯具 mesh 标注；
- 灯具投影 mask；
- 每个灯具单独的线性光照贡献；
- 灯具颜色、强度和 on/off transition；
- 真实室内开灯/关灯配对照片；
- 未知属性对应的 known mask。

没有这部分时，项目只能声称完成三个程序化合成子任务，不能声称完成完整 TokenLight。

## 7. Checkpoint 与日志要求

每个 checkpoint 至少包含：

```text
model state
optimizer state
global optimizer step
gradient accumulation state（框架支持时）
random states
Fourier matrices
model/data config
upstream revisions
dataset manifest hash
```

保存策略：

```text
last checkpoint
validation loss top 3
固定 step 间隔的轻量权重 checkpoint
```

Validation 必须覆盖每个 task，不能只报告一个混合 loss。建议记录：

```text
val/loss_total
val/loss_ambient
val/loss_diffuse
val/loss_add_light
val/loss_in_scene
```

### 7.1 Loss 历史与后续绘图

每次训练必须同时保留完整的 train loss 和 validation loss 变化历史，并使用同一 `global_step` 时间轴对齐；后续绘图时需要支持在同一张图中同时展示 train/validation loss，不能只记录或绘制其中一条曲线。

不能只把 loss 写进终端或 TensorBoard event 文件。每次正式训练同时保留以下四种记录：

```text
logs/<run_id>/tensorboard/              # TensorBoard events
logs/<run_id>/metrics/train.jsonl       # 原始训练指标
logs/<run_id>/metrics/validation.jsonl  # 原始验证指标
logs/<run_id>/checkpoints/index.jsonl   # checkpoint 与 loss 的对应关系
```

训练指标按 **optimizer step** 记录，而不是按 gradient-accumulation micro-step 记录。`train.jsonl` 每条至少包含：

```json
{
  "run_id": "tokenlight_lumina_001",
  "global_step": 1200,
  "epoch": 3,
  "samples_seen": 192000,
  "wall_time_sec": 14520.4,
  "learning_rate": 0.00001,
  "loss_total": 0.1842,
  "loss_ambient": 0.1710,
  "loss_diffuse": null,
  "loss_add_light": 0.1974,
  "loss_in_scene": null,
  "grad_norm": 0.83,
  "max_memory_allocated_gb": 71.2
}
```

某个 optimizer step 没有对应 task 时写 `null`，不能写 `0`，否则绘图和均值会被错误拉低。保存原始 loss，不在文件中只保存平滑后的数值；移动平均或 EMA 曲线在绘图阶段计算。

`validation.jsonl` 每次验证至少记录：

```text
global_step
val/loss_total
val/loss_ambient
val/loss_diffuse
val/loss_add_light
val/loss_in_scene
PSNR/SSIM/LPIPS（启用后）
```

`checkpoints/index.jsonl` 将 checkpoint 和当时指标绑定：

```json
{
  "global_step": 1200,
  "checkpoint": "checkpoints/step_000001200.safetensors",
  "val_loss_total": 0.1628,
  "is_best": true,
  "created_at": "ISO-8601 timestamp"
}
```

断点续训要求：

- 从 checkpoint 恢复后继续使用单调递增的 `global_step`；
- 不覆盖已有 JSONL；
- 新 run 记录 `parent_run_id` 和 `resumed_from_checkpoint`；
- 合并曲线时按 `global_step` 去重；
- 每行写入后及时 flush，降低异常退出造成的日志损失；
- rank 0 独占写日志，避免多卡重复或交错写入。

同时提供独立绘图脚本：

```text
tools/plot_loss.py
```

脚本输入一个或多个 run 目录，输出：

```text
loss_total_raw.png
loss_total_smoothed.png
loss_by_task.png
validation_loss.png
learning_rate.png
```

绘图脚本必须支持：

- 按 global step 对齐多个续训 run；
- 可配置 moving-average window；
- 同时绘制 train 和 validation；
- 标出 checkpoint 保存位置和 best checkpoint；
- 导出用于论文作图的合并 CSV；
- 不依赖 GPU 和训练环境即可运行。

## 8. 用户自行执行的测试矩阵

以下测试均由用户在代码交付后自行执行：

| 测试 | CPU | GPU | 必须通过阶段 |
| --- | --- | --- | --- |
| Token schema/shape | 是 | 否 | Phase 2 |
| Linear RGB composition | 是 | 否 | Phase 1 |
| Split UID leakage | 是 | 否 | Phase 1/7 |
| RoPE source-target mapping | 是 | 可选 | Phase 3 |
| Flow endpoint convention | 是 | 是 | Phase 4 |
| BF16 forward/backward | 否 | 是 | Phase 3 |
| Lighting-only CFG | 否 | 是 | Phase 5 |
| Checkpoint exact resume | 否 | 是 | Phase 6 |
| Single-sample overfit | 否 | 是 | Phase 4 |
| 10-object regression | 否 | 是 | Phase 6 |
| 960 px memory smoke | 否 | 是 | Phase 8 |

## 9. 主要风险与应对

### 风险 1：公开 backbone 与论文 backbone 不同

应对：始终使用 `TokenLight-Lumina` 名称，清楚报告 backbone substitution，不伪称使用论文原始 checkpoint。

### 风险 2：联合序列在 960 px 下显存过高

Source 和 target 合计约 7200 个空间 token，full attention 成本高。必须优先验证 Flash Attention、activation checkpointing 和 batch 1；在 256/512 验证正确性后才进入 960。

### 风险 3：移除文本条件破坏预训练能力

应对：保留 Lumina 原始 T2I 路径；对新增 TokenLight 模型做单样本过拟合和小数据比较。必要时将空文本 token 保留为 ablation，但不能在无证据时把它写成论文设计。

### 风险 4：合成数据规模大但有效变化不足

应对：渲染前后做贡献能量、阴影变化和主体可见性过滤；统计有效 point-light 和 diffuse 样本比例。

### 风险 5：单卡训练时间过长

应对：先获得每 optimizer step 的实测耗时，再估算 15,000 steps；不要根据旧 LDM 速度估算 Next-DiT。多卡可用后再启用 FSDP。

### 风险 6：论文补充材料中的细节不完整

应对：所有无法从论文确认的选择写入 `ASSUMPTIONS.md`，并设计为配置项和 ablation，不把推测写成论文事实。

## 10. 审核后的第一批执行任务

计划通过审核后，只执行以下第一批工作：

1. 读取用户提供的 Lumina-T2X 路径、环境和 checkpoint 信息；
2. 检查并记录 git commit、工作区状态和依赖版本，不执行安装或升级；
3. 提供原始 T2I 固定 seed 推理的建议命令，由用户执行验证；
4. 审计 `models/model.py` 的 JointAttention、patchify、RoPE 和 output head；
5. 输出一份具体到函数和 tensor shape 的 `architecture-audit.md`；
6. 暂不修改模型，等待架构审计确认后再进入 Phase 1 至 Phase 3。

## 11. 完成定义

只有同时满足下列条件，才称为完成 TokenLight-Lumina 复现：

- 使用 Next-DiT 2B 预训练权重初始化；
- source、noisy target、lighting 和 mask tokens 进行联合注意力；
- 使用论文形式的 flow matching；
- lighting-only CFG 正确；
- 支持 ambient、diffuse、add-light 和 in-scene-light；
- 支持多光源；
- 使用正式 960 px 线性 RGB 数据训练；
- 有独立 held-out object 测试集；
- 有 PSNR/SSIM/LPIPS 和控制连续性评测；
- checkpoint 可恢复，配置和数据 revision 可追踪；
- 提供可直接启动训练的 `train.sh`；
- 提供支持四类光照控制和多灯输入的 `infer.sh` 及对应 `infer_tokenlight.py`；
- 提供与最终代码一致的 `TOKENLIGHT_GUIDE.md` 全流程操作指南；
- README 明确披露 Lumina backbone substitution 和与论文的剩余差异。

若暂时只完成前三个合成任务，应使用以下描述：

```text
TokenLight-Lumina synthetic-task reproduction
(ambient scale, global diffuse, and virtual added lights)
```

而不是“完整 TokenLight 复现”。

## 12. 参考资料

- TokenLight paper: https://arxiv.org/abs/2604.15310
- Lumina-T2X repository: https://github.com/Alpha-VLLM/Lumina-T2X
- Lumina-Next paper: https://arxiv.org/abs/2406.18583
- Lumina-Next-T2I model: https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I
- 当前 LDM baseline 差距说明：`diff.md`
- 当前论文数据构造摘录：`conversation.md`

## 13. 后续计划：Objaverse 与 Scene 自动组合

本节仅记录后续扩展方向，不属于当前阶段的实现、数据准备、测试或完成条件。当前阶段先使用不加 Scene 的单物体程序化场景完成数据生成和 TokenLight 训练链路。

后续目标是利用已有 80 个 LVIS 类别标注，将大量 Objaverse object 与 `.blend`、`.fbx`、`.glb` Scene 自动合理组合，并生成可用于重光照训练的合成数据。建议 pipeline：

```text
读取 Objaverse LVIS 标注
-> 建立 asset UID 到 object category 的反向索引
-> 将 Scene 导入并缓存为标准化 .blend
-> 自动检测 floor/table/shelf/wall/ceiling 等候选锚点
-> 按 object category 和放置规则匹配锚点
-> 自动缩放、定向和摆放 object
-> 执行支撑、碰撞、悬空、遮挡和相机可见性检查
-> 渲染低分辨率 preview 并自动过滤
-> 固定有效组合的 Scene、object transform 和 camera
-> 渲染正式光照分量并生成 manifest
```

### 13.1 Object 类型与放置策略

- 优先使用现有 LVIS 标注，不对已有标签的 object 重新执行视觉分类；
- 将 `category -> asset UID` 转换为 `asset UID -> primary/all categories`；
- 为 80 个类别配置允许的锚点类型、真实尺寸范围、默认朝向、倾斜范围和空间间距；
- 无标签 object 才考虑使用图像分类或视觉语言模型作为兜底；
- 明显不适合当前 Scene 的类别允许标记为 unsupported，不强制组合。

### 13.2 Scene 标准化与锚点检测

- `.blend` 直接读取，`.fbx` 和 `.glb` 导入后统一缓存为标准 `.blend`；
- 统一米制单位、Z-up、坐标原点、材质和 mesh 层级；
- 根据法线、共面连通区域、面积、相对高度和净空检测水平支撑面；
- 根据法线和 Scene 边界检测 wall 与 ceiling 候选区域；
- mesh 名称中存在 `floor`、`table`、`shelf`、`wall` 等语义时作为额外证据；
- 自动检测结果允许通过少量人工审核修正，但不要求逐 object 手工摆放。

### 13.3 自动摆放与过滤

- 根据 object 包围盒、底面、主轴和类别尺寸范围完成缩放与定向；
- 在锚点边界按 object footprint 收缩后的区域中采样位置；
- 使用 BVH/raycast 检查碰撞、支撑比例、穿模和悬空；
- 必要时使用短时间刚体模拟完成稳定落位；
- 使用 object mask、深度和 preview 检查可见性、遮挡比例和画面占比；
- 每个候选设置有限重试次数，失败后切换锚点；
- 所有失败原因写入机器可读 JSONL，不能静默跳过。

### 13.4 组合采样与 TokenLight 渲染

- 不生成全部 `Scene x Object` 笛卡尔积；
- 按 Scene 类型、锚点类型、object category 和配额进行均衡采样；
- 每个 object 分配少量兼容 Scene，每个 Scene 限制各类别样本数；
- 先固定组合 manifest，再渲染不同光照分量；
- 同一组合的 Scene 几何、object transform、相机和材质必须保持不变，只改变光照；
- 后续 manifest 需要记录 `scene_asset`、`inserted_asset`、category、anchor、object transform、camera 和 lights；
- Scene 内可控灯具应进一步记录 fixture mesh、fixture mask、颜色、强度、开关状态及独立光照贡献。

该扩展开始前，先选择 3 至 5 个不同格式的代表 Scene，审计单位、坐标轴、mesh 层级、材质、灯光、相机和命名质量，再确定自动锚点检测能够利用的语义信息。
