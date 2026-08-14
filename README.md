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
- 非目标场景几何先经过 AABB broad phase，再由 world-space BVH 三角面重叠给出最终碰撞结论；AABB 重叠本身不会拒绝样本。
- 初始尺寸发生真实碰撞或 place footprint 超限时，renderer 按配置进行有下限的等比缩小；每次缩放后重新贴合 target 底面中心。达到最小比例仍不合格时拒绝 job，不会无限缩小对象。
- 每个 Blender job 都重新打开只读 base `.blend`，不会把上一个样本的状态带入下一个样本。

### 自建 composition camera

- 正式路径完全忽略原 scene 的相机；每个 job 都创建独立的 `LC_CompositionCamera`。
- 相机围绕最终插入对象中心采样方位、俯仰、距离和小幅画面偏移，因此目标不必固定在画面正中心。
- `replace` 要求原 target 中心仍在视锥内；`place_on` 要求对象与 support 的接触区域在视锥内，并要求 support mesh 有实际可见像素，但不强制整张大型支撑物入镜。
- 候选相机必须同时通过主体占比、NDC 边界、无裁切、正深度、插入对象 mask 和 target/support mask 门；第一个合格候选被固定到 metadata。
- 相机确定后才建立 canonical 坐标和采样 point/diffuse/fixture 灯光，保证 lighting token 与实际画面一致。

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
   - 围绕插入对象按其 world 尺度采样多个球形 fixture 候选
   - 球体大小也按对象尺度计算并限制最小/最大值，避免小物体使用过大的固定球体
   - 只有通过最终自建相机的视锥和可见像素门后才接受
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

`metadata.json` 包含 composition transform、初始/最终 asset scale、碰撞缩放决策、相机与 canonical 坐标、fixture 来源、base scene fingerprint、输入 digests、许可证决策和生成器版本。只有通过 validator 的 metadata 才能进入正式 manifest。

## 4. 环境

本流程面向 Linux GPU 服务器。不要用一次无版本约束的 `pip install -e .` 修改已经能运行 Stage 1 的 `lum` 环境。推荐把职责拆为四部分：

| 环境或程序 | 用途 | 是否修改现有 `lum` |
|---|---|---|
| `lightconstruction` Conda 环境 | M1–M4 外层 Python、annotation client、manifest 与数据验证 | 否 |
| Blender 4.5 可执行文件 | FBX 归一化、组合渲染和 Blender focused checks | 不属于 Conda |
| 独立 `vllm` Conda 环境 | 仅在需要重新生成 annotation 时提供 OpenAI-compatible 服务 | 否 |
| 已验证的 `lum` 环境 | TokenLight smoke、8 卡训练和评估 | 保持不变 |

`run_stage2.sh` 每次 action 只读取一个 `PYTHON_BIN`，所以不要用 `all` 跨越两个 Python 环境。分别准备 data 与 train 参数文件，并按阶段执行。

### 4.1 服务器、驱动和编译工具预检

先确认 GPU 驱动、8 张目标 GPU 和基础工具：

```bash
nvidia-smi
gcc --version
which conda
which bash
```

PyTorch wheel/Conda 包携带的是 CUDA runtime；编译 `flash-attn` 时通常还需要含 `nvcc` 的 CUDA Toolkit。只有需要编译扩展的环境才检查：

```bash
nvcc --version
echo "${CUDA_HOME:-CUDA_HOME is not set}"
```

不要仅根据 `nvidia-smi` 顶部显示的“CUDA Version”选择 PyTorch；它表示驱动支持上限，不表示当前 Python 中 PyTorch 的 CUDA runtime。最终以以下输出为准：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 4.2 LightConstruction 数据环境

先读取已经跑通 Stage 1 的 `lum` 环境版本；不要根据 README 猜版本，也不要修改 `lum`：

```bash
conda activate lum
python - <<'PY'
import torch
import torchvision
import torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
conda list | grep -E '^(pytorch|torch|torchvision|torchaudio|pytorch-cuda|cuda)\s'
```

从仓库根目录建立全新环境，并使用与上述 `lum` **完全相同**的 PyTorch、torchvision、torchaudio 和 CUDA runtime 组合。下面只是在 `lum` 恰好为 PyTorch 2.1.0/CUDA 12.1 时的示例；实际命令必须替换成 `lum` 的版本：

```bash
cd /path/to/lightconstruction
conda create -n lightconstruction python=3.11 -y
conda activate lightconstruction
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install numpy opencv-python-headless Pillow
```

根项目的 `pyproject.toml` 会安装 `openai`、`orjson`、`pydantic`、`PyYAML` 和 `tqdm`。额外的 NumPy/OpenCV 用于 component validator，Pillow 和 PyTorch 用于 `inspect_dataset.py`。虽然当前 inspection 的张量计算主要发生在 CPU，数据环境仍与 `lum` 使用同一 CUDA PyTorch 组合，避免依赖和 ABI 分叉。验证版本和 CUDA 可用性：

```bash
python - <<'PY'
import cv2
import numpy
import openai
import orjson
import pydantic
import torch
import yaml
from PIL import Image
assert torch.cuda.is_available()
print("LightConstruction dependencies OK")
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("visible GPUs:", torch.cuda.device_count())
PY
python -m lightconstruction.cli --help
```

M1–M4 的 Python 命令均从仓库根运行。`lightconstruction.cli` 只是仓库自己的薄命令入口，不是外部服务。

### 4.3 Blender 4.5

Blender 使用自己的 Python，不要安装进 Conda。`configs/default.yaml` 的示例路径是 `/opt/blender-4.5/blender`，但 `scripts/stage2.env` 中的 `BLENDER_BIN` 必须改成服务器真实可执行文件：

```bash
export BLENDER_BIN=/actual/path/to/blender-4.5/blender
test -x "${BLENDER_BIN}"
"${BLENDER_BIN}" --version
```

输出应为 Blender 4.5.x。M2 使用该程序归一化 FBX；M4 composition worker 通过 `--factory-startup` 启动，每个 worker 保持一个 Blender 进程并连续处理其 job，但每个 job 都会重新打开对应 base blend。`RENDER_PERSISTENT_DATA=true` 启用 Cycles persistent data，不代表跨 worker 或跨脚本常驻 Blender。

### 4.4 独立部署 vLLM

只有 `REUSE_ANNOTATION=false` 或没有可复用的 `data/annotation_construction.json` 时才需要 vLLM。不要为了 M3 修改现有 `lum`，建立独立服务环境：

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
python -m pip install -U pip
python -m pip install vllm
python -c "import vllm; print(vllm.__version__)"
which vllm
```

如果服务器不能在线下载 Hugging Face 模型，先把 Qwen3-14B 放到本地共享存储，并将后续 `VLLM_MODEL` 和启动命令都改成同一个本地目录。单卡手动部署：

```bash
conda activate vllm
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 8000
```

另一个终端验证服务，而不是只检查进程存在：

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

此时 data 参数文件使用：

```bash
REUSE_ANNOTATION=false
START_VLLM=false
VLLM_MODEL=Qwen/Qwen3-14B
VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

也可以让脚本启动并在 annotation 结束后关闭服务。把独立环境入口的绝对路径写入参数文件：

```bash
REUSE_ANNOTATION=false
START_VLLM=true
VLLM_BIN=/actual/conda/envs/vllm/bin/vllm
VLLM_MODEL=Qwen/Qwen3-14B
VLLM_GPU_IDS=0
```

当前自动启动入口只设置 `CUDA_VISIBLE_DEVICES`，没有传 `--tensor-parallel-size`；因此 `VLLM_GPU_IDS=0,1` 不等于启用两卡张量并行。需要多卡时保持 `START_VLLM=false`，手动运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2
```

脚本自动启动失败时查看 `${RUNTIME_DIR}/vllm.log`。手动服务不会由脚本关闭；完成 M3 后应由操作者停止，释放 Blender/训练需要的 GPU。

### 4.5 TokenLight、PyTorch、CUDA 和 FlashAttention

如果现有 `lum` 已经跑通 Stage 1，优先原样复用，先保存环境清单，不执行任何安装或升级。它可以直接用于 smoke/train/eval；若 Stage 2 缺包，则使用克隆环境，不要修补原环境：

```bash
conda activate lum
python -m pip freeze > /path/outside/repo/lum-environment-freeze.txt
python -m pip check
```

验证 Stage 2 的实际必需项：

```bash
cd /path/to/lightconstruction/Lumina-T2X
python - <<'PY'
import accelerate
import cv2
import diffusers
import fairscale
import flash_attn
import safetensors
import tensorboard
import torch
import transformers
import yaml

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
print("flash-attn:", flash_attn.__version__)
print("TokenLight dependencies OK")
PY
```

当前 Next-DiT 配置强制 `runtime.flash_attention=true`，所以 `flash_attn` 不是可选依赖；Apex 则是可选项，不要安装 Python-only Apex。若现有 `lum` 缺少任何依赖，或者希望使用统一的 `run_stage2.sh tests`，先克隆环境并只修改克隆：

```bash
conda create -n lum-stage2 --clone lum -y
conda activate lum-stage2
cd /path/to/lightconstruction
python -m pip install -e .
python -m pip check
```

仅当没有可复用训练环境时，才按 Lumina-Next-T2I 上游文档建立新环境。仓库当前给出的参考组合是 Python 3.11、PyTorch 2.1.0、torchvision 0.16.0、torchaudio 2.1.0 和 CUDA runtime 12.1；不要用它覆盖一个已经验证的环境：

```bash
conda create -n tokenlight python=3.11 -y
conda activate tokenlight
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  pytorch-cuda=12.1 -c pytorch -c nvidia -y

cd /path/to/lightconstruction/Lumina-T2X
python -m pip install -r requirements.txt
python -m pip install ninja packaging
python -m pip install flash-attn --no-build-isolation
python -m pip install -e . --no-deps
python -m pip check
```

`flash-attn` 必须在 PyTorch 安装后构建；若无兼容预编译包，构建使用的 `nvcc`/`CUDA_HOME` 必须与该 PyTorch CUDA runtime 兼容。仓库没有锁定 FlashAttention 版本，因此新环境必须先通过 import、单卡 smoke 和 8 卡 DDP smoke，再冻结版本，不能仅凭安装命令成功认定可用。

检查8张训练卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    print(index, torch.cuda.get_device_name(index))
PY
```

### 4.6 模型、VAE 和 checkpoint 文件

权重必须位于仓库外，由操作者在 env 文件中提供绝对路径：

```text
/path/to/stage1/checkpoints/step_XXXXXXXXX/   # STAGE1_CHECKPOINT
/path/to/sdxl-vae/                           # VAE_PATH
  config.json
  diffusion_pytorch_model.safetensors
```

`STAGE1_CHECKPOINT` 是已完成 Stage 1 的 TokenLight checkpoint 目录，不是任意单个权重文件。首次 Stage 2 加载时 optimizer/global step/sampler 会重新创建；只有 `FORMAL_RESUME_CHECKPOINT` 才恢复完整 Stage 2 训练状态。VAE 必须是可由 Diffusers `AutoencoderKL.from_pretrained()` 读取的目录。上游权重加载只允许 TokenLight 新增的 `lighting_encoder` 和 `fixture_mask_embedder` keys 缺失。

运行前检查：

```bash
test -d "${STAGE1_CHECKPOINT}"
test -f "${VAE_PATH}/config.json"
test -f "${VAE_PATH}/diffusion_pytorch_model.safetensors"
```

### 4.7 用两份 env 文件隔离 data 与 train

不要提交包含服务器路径的实际 env 文件。建立两份本地参数文件：

```bash
cp scripts/stage2.env.example scripts/stage2.data.env
cp scripts/stage2.env.example scripts/stage2.train.env
```

`scripts/stage2.data.env`：

```bash
PYTHON_BIN=/actual/conda/envs/lightconstruction/bin/python
VLLM_BIN=/actual/conda/envs/vllm/bin/vllm
```

`scripts/stage2.train.env`：原 `lum` 已通过全部 import 时可以直接指向它；需要补依赖或运行统一 tests 时，指向 `lum-stage2` 克隆。

```bash
PYTHON_BIN=/actual/conda/envs/lum/bin/python
```

两份文件中的 `RUNTIME_DIR`、`DATASET_ROOT`、`TOKENLIGHT_OUTPUT_ROOT`、模型路径、许可证和训练参数必须保持一致。执行：

```bash
STAGE2_ENV_FILE=scripts/stage2.data.env bash scripts/run_stage2.sh data

STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh smoke
STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh train
STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh eval
```

统一的 tests action 同时运行根项目和 Lumina-T2X 测试，需要 `lum-stage2` 这类兼具两边依赖的克隆环境：

```bash
STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh tests
```

如果坚持让 `stage2.train.env` 指向完全不改动的原 `lum`，则分别使用环境绝对路径执行三段检查，不使用统一 tests action：

```bash
/actual/conda/envs/lightconstruction/bin/python -m unittest discover -s tests -p 'test_*.py'
/actual/path/to/blender-4.5/blender --background --factory-startup \
  --python tests/blender_composition_focus.py
(cd Lumina-T2X && /actual/conda/envs/lum/bin/python -m unittest discover -s tests -p 'test_*.py')
```

不要在分离环境模式下执行 `bash scripts/run_stage2.sh all`，因为一次 shell 运行只能使用一个 `PYTHON_BIN`。

## 5. 正式运行前必须填写的配置

代码会拒绝猜测对象尺度和许可证。以下配置为空时不能进入正式 M4。

### 第二阶段自动运行脚本（不包含 Stage 1）

仓库提供 [`scripts/run_stage2.sh`](scripts/run_stage2.sh)，用于从 object/scene 预处理一直执行到组合渲染、数据门、固定-manifest smoke、8 卡正式训练和评估。它**不会生成 Stage 1 简易数据，也不会训练 Stage 1**；选定的 Stage 1 TokenLight checkpoint 是操作者提供给 Stage 2 的只读输入。

按第 4.7 节建立 data/train 两份服务器参数文件；如果明确选择单环境运行，也可以使用默认文件名：

```bash
cp scripts/stage2.env.example scripts/stage2.env
vim scripts/stage2.env
```

必须填写实际 `OBJECT_ROOT`、Blender、Stage 1 checkpoint、VAE、许可证政策和 GPU 编号。包含空格的值必须像模板中的 attribution 一样加引号。正式训练固定使用配置中的 BF16；模板的 micro batch 1、gradient accumulation 4 和 8 卡对应 global batch 32。`FORMAL_ACTIVATION_CHECKPOINTING` 必须显式填写：显存足够的 A100 建议设为 `false` 以避免重算开销，出现 OOM 时再设为 `true` 并使用新 `FORMAL_RUN_ID`。固定-manifest smoke 为降低单卡显存压力会独立强制开启 activation checkpointing，不受该变量控制。

在仓库根目录按需执行：

| 命令 | 执行范围 |
|---|---|
| `STAGE2_ENV_FILE=scripts/stage2.data.env bash scripts/run_stage2.sh data` | M1 object、M2 scene、M3 annotation、M4 组合渲染、manifest、组件验证和 Dataset inspection |
| `STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh tests` | 仅在 `PYTHON_BIN` 指向兼具两边依赖的 `lum-stage2` 克隆时运行全部检查 |
| `STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh smoke` | 固定小 manifest，单卡执行 forward/backward/update/checkpoint 和精确恢复 |
| `STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh train` | smoke 与数据摘要仍匹配后，启动单机 8 卡 Stage 2 DDP |
| `STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh eval` | 使用指定或自动选择的正式 checkpoint 单卡评估 |
| `bash scripts/run_stage2.sh all` | 仅适用于一个 Python 同时具备 data 与 train 全部依赖的环境；分离环境时禁用 |

已有且 digest 仍有效的 `data/annotation_construction.json` 可通过 `REUSE_ANNOTATION=true` 复用。需要重新标注时，可以先自行启动 vLLM；也可以设置 `START_VLLM=true`，让脚本启动并在标注结束后关闭它。正式渲染不使用 `--allow-partial`，每个 render worker 只启动一次 Blender 并连续消费分配给它的 job；`RENDER_PERSISTENT_DATA=true` 同时启用 Cycles persistent data。

脚本根据模板生成以下运行时文件，不需要手工修改：

```text
outputs/stage2_runtime/project.yaml
outputs/stage2_runtime/tokenlight_stage2.yaml
outputs/stage2_runtime/tokenlight_smoke_phase1.yaml
outputs/stage2_runtime/tokenlight_smoke_phase2.yaml
outputs/stage2_runtime/tokenlight_evaluate.yaml
```

`train` 会拒绝以下情况：smoke 未通过、checkpoint resume 未验证、完整 train/validation manifest 或 `dataset_release.json` 在 smoke 后变化、8 卡列表数量不等于 8，或者未设置 resume 却复用了已有正式 run 目录。正式断点续训时填写 `FORMAL_RESUME_CHECKPOINT`，并保持原来的8卡 world size和训练签名。

#### `stage2.env` 变量完整清单

可执行文件和配置：

| 变量 | 是否可空 | 作用与约束 |
|---|---|---|
| `PYTHON_BIN` | 否 | 当前 action 使用的 Python；data 指向 `lightconstruction`，tests/smoke/train/eval 指向 `lum` |
| `BLENDER_BIN` | 否 | Blender 4.5.x 可执行文件的绝对路径 |
| `PROJECT_BASE_CONFIG` | 否 | LightConstruction 基础配置，通常为 `configs/default.yaml` |
| `TOKENLIGHT_BASE_CONFIG` | 否 | TokenLight 基础配置，通常为 `Lumina-T2X/lumina_next_t2i/config.yaml` |
| `RUNTIME_DIR` | 否 | 生成的 project/formal/smoke/eval YAML 和 vLLM 日志目录；两份 env 必须一致 |
| `OBJECT_ROOT` | 否 | Objaverse canonical asset 根目录；`object.json` 中保存的是相对路径 |
| `STAGE1_CHECKPOINT` | 否 | 已完成 Stage 1 的 TokenLight checkpoint 目录；不是 Stage 2 resume |
| `VAE_PATH` | 否 | SDXL VAE 的 Diffusers 目录 |
| `DATASET_ROOT` | 否 | composition 分量、manifest、validator 和 release 输出根目录 |
| `TOKENLIGHT_OUTPUT_ROOT` | 否 | smoke 之外的正式 Stage 2 训练输出根目录 |

许可证：

| 变量 | 是否可空 | 作用与约束 |
|---|---|---|
| `OBJECT_LICENSE_ALLOWLIST` | 否 | 与 `object.json` 规范名称匹配的逗号分隔白名单，比较时忽略大小写 |
| `LICENSE_POLICY_VERSION` | 否 | 操作者批准的策略版本；不能使用 `unconfigured` |
| `BASE_SCENE_LICENSE_NAME` | 否 | Bistro 的已核验许可证名称 |
| `BASE_SCENE_SOURCE_URI` | 否 | 场景规范来源 URL |
| `BASE_SCENE_ATTRIBUTION` | 否 | 必需署名；含空格时加引号 |
| `BASE_SCENE_LICENSE_DECISION` | 否 | 正式数据必须严格为 `allowed` |

annotation/vLLM：

| 变量 | 是否可空 | 作用与约束 |
|---|---|---|
| `REUSE_ANNOTATION` | 否 | `true` 时仅在目标文件存在时复用；M4 仍校验 object/scene digest |
| `START_VLLM` | 否 | `true` 由脚本启动/关闭服务；`false` 要求 `/v1/models` 已可访问 |
| `VLLM_BIN` | 否 | 自动启动时使用的 `vllm` 命令，推荐独立环境绝对路径 |
| `VLLM_MODEL` | 否 | server 与 annotation request 共用的模型名或本地模型目录 |
| `VLLM_BASE_URL` | 否 | OpenAI-compatible 基址，默认 `http://127.0.0.1:8000/v1` |
| `VLLM_GPU_IDS` | 否 | 自动启动时写入 `CUDA_VISIBLE_DEVICES`；不自动启用 tensor parallel |
| `VLLM_CONCURRENCY` | 否 | annotation 并发请求数；显存/服务不稳时降低 |
| `VLLM_WAIT_SECONDS` | 否 | 自动启动健康检查超时秒数 |

Blender composition：

| 变量 | 是否可空 | 作用与约束 |
|---|---|---|
| `RENDER_GPU_IDS` | 否 | Blender worker 可见的物理 GPU，逗号分隔且不能重复 |
| `RENDER_WORKERS` | 否 | Blender 常驻 worker 数；保守设置为每 GPU 一个 |
| `RENDER_RESOLUTION` | 否 | 渲染与 TokenLight data resolution，共用同一整数 |
| `RENDER_SAMPLES` | 否 | Cycles samples |
| `RENDER_PERSISTENT_DATA` | 否 | 严格为 `true` 或 `false` |
| `MAX_RENDER_JOBS` | 是 | 空表示全部；真实小 batch 验收填写 `1`，不可把该产物当正式全量数据 |

smoke 与正式训练：

| 变量 | 是否可空 | 作用与约束 |
|---|---|---|
| `SMOKE_GPU_ID` | 否 | 单卡 smoke 使用的物理 GPU |
| `SMOKE_MAX_MANIFEST_ROWS` | 否 | 固定 smoke manifest 最大行数 |
| `SMOKE_MAX_SCHEDULE_SAMPLES` | 否 | 为覆盖全部任务搜索 sampler 前缀的上限 |
| `SMOKE_RUN_ID` | 否 | smoke 唯一名称；同名输出存在时拒绝覆盖 |
| `TRAIN_GPU_IDS` | 否 | 正式单机 DDP 必须恰好列出8张物理 GPU |
| `FORMAL_RUN_ID` | 否 | 正式 run 唯一名称；新训练不得复用已有目录 |
| `FORMAL_MICRO_BATCH_SIZE` | 否 | 每个 rank 的 micro batch |
| `FORMAL_GRADIENT_ACCUMULATION_STEPS` | 否 | 梯度累积次数；global batch 为该值 × micro batch × 8 |
| `FORMAL_MAX_STEPS` | 否 | 正式 optimizer update 总步数 |
| `FORMAL_LEARNING_RATE` | 否 | 正式 AdamW 学习率 |
| `FORMAL_NUM_WORKERS` | 否 | 每个 DDP rank 的 DataLoader worker 数；总进程压力约为该值 × 8 |
| `FORMAL_ACTIVATION_CHECKPOINTING` | 否 | `false` 更快但占显存；`true` 省显存但重算；resume 时不得改变 |
| `FORMAL_RESUME_CHECKPOINT` | 是 | 首次训练留空；恢复时填同一个 Stage 2 run 的 checkpoint 目录 |
| `EVAL_GPU_ID` | 否 | 单卡评估使用的物理 GPU |
| `EVAL_CHECKPOINT` | 是 | 空时自动选择正式 run 中编号最大的 checkpoint |

#### 首次运行指南

建议不要第一次执行 `all`；使用分离环境逐段确认产物，才能准确定位路径、许可证、vLLM、Blender、数据和显存问题。

1. 分别检查数据与训练环境：

   ```bash
   cd /path/to/lightconstruction
   conda activate lightconstruction
   python -c "import openai, orjson, pydantic, numpy, cv2, torch, yaml; print('data env OK')"

   conda activate lum
   python -c "import torch, flash_attn, diffusers, fairscale, cv2, yaml; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

   /actual/path/to/blender-4.5/blender --version
   nvidia-smi
   ```

2. 复制参数模板并填写服务器实际值：

   ```bash
   cp scripts/stage2.env.example scripts/stage2.data.env
   cp scripts/stage2.env.example scripts/stage2.train.env
   vim scripts/stage2.data.env
   vim scripts/stage2.train.env
   ```

   先做无数据副作用的语法和路径预检：

   ```bash
   bash -n scripts/run_stage2.sh

   set -a
   source scripts/stage2.data.env
   set +a
   "${PYTHON_BIN}" -m lightconstruction.cli --help
   test -x "${BLENDER_BIN}"
   test -d "${OBJECT_ROOT}"
   test -d "${STAGE1_CHECKPOINT}"
   test -f "${VAE_PATH}/config.json"

   set -a
   source scripts/stage2.train.env
   set +a
   "${PYTHON_BIN}" -c "import torch, flash_attn; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
   ```

   最少需要确认：

   | 参数 | 含义 |
   |---|---|
   | `PYTHON_BIN` | data 文件填 `lightconstruction` Python，train 文件填已验证的 `lum`/`lum-stage2` Python；推荐绝对路径 |
   | `BLENDER_BIN` | Blender 4.5 可执行文件 |
   | `OBJECT_ROOT` | Objaverse canonical asset 根目录 |
   | `STAGE1_CHECKPOINT` | 已完成的 Stage 1 TokenLight checkpoint 目录；脚本不会生成它 |
   | `VAE_PATH` | SDXL VAE 目录 |
   | `OBJECT_LICENSE_ALLOWLIST` | 与 `object.json` 一致的规范许可证名，逗号分隔 |
   | `BASE_SCENE_*` | Bistro 来源、署名和已审核的许可证结论 |
   | `RENDER_GPU_IDS` | Blender 渲染使用的物理 GPU 列表 |
   | `TRAIN_GPU_IDS` | 正式 DDP 使用的8张物理 GPU |
   | `FORMAL_RUN_ID` | Stage 2 正式 run 的唯一名称 |

3. 生成和验证组合数据：

   ```bash
   STAGE2_ENV_FILE=scripts/stage2.data.env bash scripts/run_stage2.sh data
   ```

   若已有当前 object/scene digest 对应的 annotation，保持 `REUSE_ANNOTATION=true`。如果需要重新调用 Qwen 标注，设置 `REUSE_ANNOTATION=false`，并选择以下一种方式：

   ```bash
   # 方式一：先在 scripts/stage2.data.env 中设置 START_VLLM=true，
   # 然后让脚本自己启动和关闭 vLLM
   STAGE2_ENV_FILE=scripts/stage2.data.env bash scripts/run_stage2.sh data

   # 方式二：另一个终端提前启动，stage2.data.env 中保持 START_VLLM=false
   vllm serve Qwen/Qwen3-14B --host 127.0.0.1 --port 8000
   curl --fail http://127.0.0.1:8000/v1/models
   ```

   `data` 成功的终点不是只有渲染文件存在，而是 manifest 构建、组件验证和 Dataset inspection 全部返回成功。

4. 运行仓库级检查。只有 `stage2.train.env` 指向已安装根项目依赖的 `lum-stage2` 克隆时才使用统一入口；否则使用第 4.7 节的三段分离命令：

   ```bash
   STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh tests
   ```

5. 使用组合数据执行训练 smoke：

   ```bash
   STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh smoke
   ```

   脚本会确定性选择小 manifest 和覆盖全部任务的 sampler 前缀，自动连续调用两次现有 `train_tokenlight.py`：第一次执行 forward/backward/update 并保存 checkpoint，第二次精确恢复后再更新一次。这里的“smoke 两段”只是恢复验收，不是 Stage 1/Stage 2 课程训练。

6. 启动 Stage 2 正式8卡训练：

   ```bash
   STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh train
   ```

   首次 Stage 2 训练必须保持：

   ```bash
   FORMAL_RESUME_CHECKPOINT=
   ```

   此时模型从 `STAGE1_CHECKPOINT` 加载权重，但 optimizer、global step 和 sampler 都新建。有效 global batch 的计算是：

   ```text
   FORMAL_MICRO_BATCH_SIZE × FORMAL_GRADIENT_ACCUMULATION_STEPS × 8
   ```

7. 评估 checkpoint：

   ```bash
   STAGE2_ENV_FILE=scripts/stage2.train.env bash scripts/run_stage2.sh eval
   ```

   `EVAL_CHECKPOINT` 为空时会选择 `${TOKENLIGHT_OUTPUT_ROOT}/${FORMAL_RUN_ID}/checkpoints/` 下编号最大的 checkpoint；需要评估指定权重时，在 `stage2.train.env` 中填写其完整目录。

#### 中断恢复与失败重跑

- **正式训练中断**：把 `FORMAL_RESUME_CHECKPOINT` 设置为同一个 Stage 2 run 的 checkpoint 目录，再执行 `train`。不得在恢复时改变 manifest、world size、学习率、batch、任务概率或 lighting/flow 语义。
- **想重新开始正式训练**：清空 `FORMAL_RESUME_CHECKPOINT`，并更换 `FORMAL_RUN_ID`；脚本不会覆盖已有 run。
- **smoke 失败或需要重跑**：修复原因后更换 `SMOKE_RUN_ID`。旧 summary 不会被当作新数据版本的通过结果。
- **渲染失败**：查看 `${DATASET_ROOT}/render_errors.jsonl`、`${DATASET_ROOT}/runtime/composition_worker_*.log` 和 `${DATASET_ROOT}/components/<job_id>.partial/failure.json`。默认重跑会保留已有 `.partial` 并返回其诊断路径；确认失败原因后，只清理需要重做的单个 job，再执行渲染：

  ```bash
  set -a
  source scripts/stage2.data.env
  set +a

  "${PYTHON_BIN}" -m lightconstruction.cli clear-render-partial \
    --config outputs/stage2_runtime/project.yaml \
    --job-id <job_id>
  OBJECT_ROOT="${OBJECT_ROOT}" "${PYTHON_BIN}" -m lightconstruction.cli render \
    --config outputs/stage2_runtime/project.yaml \
    --blender-bin "${BLENDER_BIN}" \
    --workers "${RENDER_WORKERS}"
  ```

  清理命令只接受当前 `render_jobs_output` manifest 中唯一存在且带 `failure.json` 的 job；不会清理其他 partial，也不会删除已完成的 `metadata.json`。
- **数据在 smoke 后变化**：重新执行 `data` 和 `smoke`。`train` 会比较完整 manifest 与 `dataset_release.json` 的 SHA256，并拒绝复用过期 smoke。
- **只想完整串行运行**：只有同一个 Python 已经同时通过 data 与 train 的全部依赖检查时，才能在逐段验证后执行 `bash scripts/run_stage2.sh all`；采用推荐的分离环境时必须按 action 分开运行。

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

完整的独立环境安装、单卡/多卡部署、自动启动限制和日志位置见第 4.4 节。手动服务示例：

```bash
vllm serve Qwen/Qwen3-14B --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/v1/models
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

必须确认 `unresolved_pairs` 符合预期，且文件中的 object/scene digest 与当前输入一致。低于
`annotation.confidence_review_threshold` 的 LLM 关系只进入 review，不会生成正式 target；M4
复用旧 annotation 时也会依据 `class_rules` 再执行同一置信度门。M4 会拒绝 stale annotation。

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
资产解析优先使用 `object.json` 的 `canonical_path`；若旧 inventory 的该字段带有并不存在的
`glbs/` 前缀，则只在同一个 `OBJECT_ROOT` 内回退到 `inventory_path`。不需要创建
`glbs -> OBJECT_ROOT` 的自引用软链接，任何解析到 `OBJECT_ROOT` 外的路径都会被拒绝。

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

相同输入、配置和 seed 必须产生相同 job ID、target、yaw 和 transform 目标。job ID 还绑定
实际 GLB/normalized blend 摘要、几何契约、fixture 候选、许可证和渲染配置；其中任一项变化
都会生成新的 job ID，避免复用旧渲染。

## 10. M4：组合渲染

先用小批次设置：

```yaml
m4:
  max_render_jobs: 8
  render_gpu_ids: [0]
  render_workers: 1
  overwrite: false
  camera_strategy: generated_target_visible
  camera_candidate_count: 24
  camera_subject_fill_range: [0.25, 0.60]
  camera_ndc_x_range: [0.15, 0.85]
  camera_ndc_y_range: [0.15, 0.85]
  render:
    resolution: 256
    samples: 16
    require_gpu: true
    collision_bvh_epsilon: 0.0
    collision_shrink_enabled: true
    collision_shrink_factor: 0.90
    collision_min_scale_ratio: 0.60
    collision_max_attempts: 6
```

服务器上保留一条真实小 Blender batch 验收路径。复制一份独立环境文件，将其中
`MAX_RENDER_JOBS=1`、`RENDER_GPU_IDS=0`、`RENDER_WORKERS=1`，并把
`DATASET_ROOT` 改为独立验收目录；确认输入路径和许可证后运行：

```bash
cp scripts/stage2.env scripts/stage2.blender-acceptance.env
STAGE2_ENV_FILE=scripts/stage2.blender-acceptance.env \
  bash scripts/run_stage2.sh data
```

该命令必须在装有项目 Python 依赖、真实 Blender、GPU、Objaverse 资产和 scene blend 的服务器上执行；检查生成 metadata 的相机 NDC、主体可见像素、`fixture_source`，并让后续 composition validator 通过。本地 focused 测试不等价于此真实 batch，也不得替代服务器验收结果。

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

组件目录先写为 `<job_id>.partial`，只有 metadata 完整后才原子改名。缺文件、摘要变化和许可证
失败会在创建 partial 前终止。`overwrite: false` 仅在已有 `metadata.json` 的
`render_job_digest` 与当前 job 完全一致时复用；不一致时拒绝覆盖。无论 `overwrite` 配置如何，
已有 partial 都不会被 renderer 自动删除。检查 `<job_id>.partial/failure.json` 后，使用
`python -m lightconstruction.cli clear-render-partial --config configs/default.yaml --job-id <job_id>`
明确清理该单个失败 job，再重新执行渲染。正式 composition 默认要求 Cycles GPU 初始化成功，
不会静默回退 CPU。

## 11. 构建 split 和严格验证

编辑 `Lumina-T2X/lumina_next_t2i/config.yaml`，让 TokenLight 指向 M4 输出：

```yaml
paths:
  dataset_root: /absolute/path/to/outputs/tokenlight_dataset
  render_output_root: /absolute/path/to/outputs/tokenlight_dataset
  render_jobs_manifest: /absolute/path/to/outputs/manifests/render_jobs.jsonl
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

`require_composition_contract: true` 时，manifest builder 只读取 `render_jobs_manifest` 中列出的
精确 job，并核对 metadata 的 job ID、对象、场景和 `render_job_digest`；同一输出根下的历史完成
目录不会混入当前数据集，其 ID 会记录在 `ignored_foreign_completed_outputs`。每次构建还会写
`dataset_release.json`，保存 split、fixture 来源、许可证政策、lineage digest 和三个 manifest digest。

## 12. 两阶段训练：简易数据预训练，再用组合数据微调

推荐把两类数据分成两个独立训练 run，而不是混在同一个 manifest 中：

1. **Stage 1（简易数据）**：切回原分支/原代码，使用 Lumina 原有的简易场景搭建与渲染数据训练 TokenLight。
2. **Stage 2（组合数据）**：切到当前代码，使用本 README 生成的 object + scene 组合数据继续训练。

Stage 1 正常从 Lumina 上游权重开始，示意配置如下：

```yaml
paths:
  upstream_checkpoint: /path/to/Lumina-Next-T2I
  resume_checkpoint: null
  dataset_root: /path/to/simple-tokenlight-dataset
  train_manifest: /path/to/simple-tokenlight-dataset/manifests/train.jsonl
  validation_manifest: /path/to/simple-tokenlight-dataset/manifests/validation.jsonl

train:
  learning_rate: 1.0e-5

logging:
  run_id: tokenlight_stage1_simple_v1
```

Stage 1 训练完成并选定 checkpoint 后，Stage 2 把该 **checkpoint 目录**作为上游权重，新建 run：

```yaml
paths:
  upstream_checkpoint: /path/to/stage1-output/checkpoints/step_XXXXXXXXX
  resume_checkpoint: null
  dataset_root: /path/to/composition-tokenlight-dataset
  train_manifest: /path/to/composition-tokenlight-dataset/manifests/train.jsonl
  validation_manifest: /path/to/composition-tokenlight-dataset/manifests/validation.jsonl

train:
  # 组合数据微调建议先用低于 Stage 1 的学习率；具体值由真实 smoke 决定。
  learning_rate: 2.0e-6

logging:
  run_id: tokenlight_stage2_composition_v1
```

这里必须区分两种加载语义：

- **Stage 1 → Stage 2 是权重初始化**：设置 `upstream_checkpoint`，保持 `resume_checkpoint: null`；Stage 2 使用新的 optimizer、global step、sampler、输出目录和 run ID。
- **同一 Stage 2 run 的中断续训才是精确恢复**：设置 `resume_checkpoint`；manifest、seed、任务概率、batch、world size 和关键模型配置必须与保存 checkpoint 时一致。

Stage 1 与 Stage 2 的主干模型结构必须兼容。若 Stage 1 没有启用 fixture mask 分支，Stage 2 可以容许 `fixture_mask_embedder.*` 缺失并随机初始化，但这部分能力会从 Stage 2 才开始学习；如果简易数据能够提供有效 fixture mask，优先在 Stage 1 就启用它。Stage 2 开始前仍需使用其组合数据完成下一节的固定-manifest smoke；此时 smoke 的 `upstream_checkpoint` 应指向选定的 Stage 1 checkpoint，而不是重新指向原始 Lumina 权重。

两个阶段应分别保留验证结果。Stage 2 至少同时观察组合数据验证集和简易数据验证集，确认组合能力提升时没有出现不可接受的基础能力遗忘；只有确实观察到遗忘时，再考虑在 Stage 2 加入少量简易数据 replay。

## 13. 固定 manifest 训练 smoke

Dataset 可读不等于训练可用。任何数据版本在两阶段 smoke 完成前都不是 `train-ready`。

### 13.1 Smoke 配置

复制正式配置为服务器本地 smoke 配置，并显式修改：

```yaml
paths:
  # Stage 2 smoke 使用选定的 Stage 1 checkpoint 目录；单阶段训练可填原始 Lumina checkpoint。
  upstream_checkpoint: /path/to/stage1-output/checkpoints/step_XXXXXXXXX
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

### 13.2 第一阶段

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

### 13.3 第二阶段恢复

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

## 14. 正式训练与评测

Smoke 通过后，恢复正式 `max_steps`、worker、DDP 和独立 `logging.run_id`。采用两阶段策略时，Stage 2 正式训练从已选定的 Stage 1 TokenLight checkpoint 初始化；单阶段训练时才直接从上游 Lumina 权重初始化。两者首次启动都保持：

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

## 15. 测试命令

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

## 16. 常见阻断条件

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

## 17. 可声明结果的边界

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
