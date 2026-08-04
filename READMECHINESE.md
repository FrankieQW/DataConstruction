# SceneCompose

[English README](README.md)

三阶段的代码级算法、数据流、产物契约和当前限制见 [技术实现细节](docs/technical-implementation-details.md)。

SceneCompose 按 Inline Execution 组织为三个大步骤：

1. 基于观察锚点的 Scene 局部切分；
2. 对局部 Scene 进行语义/实例分割；
3. Object 目录构建、放置、约束求解、物理验证和 GLB 导出。

当前已实现三个步骤。第三步输出 Object 与 partition scene 的组合 GLB 和参考相机参数，但不执行图片渲染。

## 前置条件

- Linux
- Python 3.10 或更高版本
- Blender 4.5 LTS，可通过 `blender` 调用，或由 `SCENECOMPOSE_BLENDER` 指定
- SceneCompose 环境和 Blender 内置 Python 均需提供 NumPy

以下源码仓库需直接位于 SceneCompose 项目根目录：

```bash
git clone https://github.com/NVlabs/WarpConvNet.git
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/VinAIResearch/Open3DIS.git
git clone https://github.com/JinLi998/CoSMo3D.git
git clone https://github.com/NVlabs/Mosaic3D.git
```

观察切分本身不依赖模型权重。SceneCompose 不会自动下载权重。

### 查看 Objaverse metadata

当 `data/obj` 中的 Objaverse GLB 使用 `NNN-NNN/<uid>.glb` 结构时，可根据文件清单下载对应的官方 metadata。该命令不会下载或修改 GLB，也不会下载模型权重：

```bash
bash scripts/run_objaverse_metadata.sh data/obj/Objaverse.md data/obj/metadata
```

也可以直接运行：

```bash
python scripts/download_objaverse_metadata.py \
  --manifest data/obj/Objaverse.md \
  --output data/obj/metadata
```

所有官方 gzip 文件都缓存在 `data/obj/metadata/cache`，不会写入 `~/.objaverse`。建议先查看 `data/obj/metadata/sample.json`；完整结果是 `annotations.json` 和 `annotations.jsonl`，`lvis_categories.json` 提供后续组合阶段优先使用的类别信号。字段定义、断点续传和缺失 UID 行为见 [Objaverse metadata 产物说明](docs/objaverse-metadata-artifacts.md)。

## 安装

Blender 独立于 Python 环境管理。可选择 Mamba 或 Pixi。

### 方法一：Mamba

```bash
mamba create -n scenecompose -c conda-forge python=3.10 pip numpy
mamba activate scenecompose
python -m pip install -e .
export SCENECOMPOSE_BLENDER=/opt/blender/blender
```

语义分割还需在同一环境安装 Mosaic3D 与 SAM3 依赖：

```bash
python -m pip install -r Mosaic3D/requirements.txt
python -m pip install -e sam3
python -m pip install -e .
```

### 方法二：Pixi

```bash
pixi init
pixi add "python=3.10.*"
pixi install
pixi run python -m pip install -r Mosaic3D/requirements.txt
pixi run python -m pip install -e sam3
pixi run python -m pip install -e .
export SCENECOMPOSE_BLENDER=/opt/blender/blender
```

已有 `pixi.lock` 后只需执行 `pixi install`，无需再次执行 `pixi init`。安装 CUDA 依赖前应核对 `Mosaic3D/requirements.txt` 与服务器驱动是否匹配。

## 自适应切分

### 默认方案：基于观察锚点的局部切分

程序从朝上的连续大面积几何中寻找候选地面，检查观察点附近的地面连续性、头部净空和身体净空，然后选择参考观察方向。输出范围由半径、水平视角、垂直范围以及相机移动 Context 共同确定。

切分不会按遮挡删除几何。墙后几何可以保留在 partition 中；后续分割视图的深度图负责统计真实可见性，最终 Object 放置还需针对最终相机执行 BVH 射线检查。

单个 Scene：

```bash
bash scripts/run_observation_partition.sh \
  data/scene/example.fbx \
  data/work/example/observations
```

等价命令：

```bash
scenecompose sample-observations \
  --scene data/scene/example.fbx \
  --output data/work/example/observations \
  --config configs/observation_partition.json \
  --blender "$SCENECOMPOSE_BLENDER"
```

批量处理 `data/scene` 下所有 FBX：

```bash
scenecompose sample-observations-all \
  --scene-root data/scene \
  --output-root data/work \
  --config configs/observation_partition.json \
  --workers 4
```

`--workers` 表示并发 Blender 进程数，应按 CPU 内存设置，而不是按 GPU 数量设置。

每个 observation 的主要产物为：

```text
data/work/<scene-id>/observations/observation_<hash>/
|-- observation.json
|-- partition/
|   |-- scene_partition.glb
|   `-- source_faces.npz
`-- segmentation/
```

几何保持原 Scene 世界坐标。`source_faces.npz` 保存输出三角面到原始对象、实例和 polygon 的映射，以及 `is_core/is_context`。详细契约见 [观察切分产物说明](docs/observation-partition-artifacts.md)。

### Legacy：XY 自适应切分

旧的 `partition` 和 `partition-all` 命令继续保留，用于调试和兼容已有结果，但不再作为默认分割和组合入口。

```bash
bash scripts/run_partition.sh \
  data/scene/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx \
  data/work/EmeraldSquare_Day/partition
```

```bash
scenecompose partition-all \
  --scene-root data/scene \
  --output-root data/work \
  --config configs/partition.json \
  --workers 1
```

## Scene 语义分割

分割采用 Mosaic3D 开放词汇 3D 特征、SAM3 多视图提示分割，以及 Open3DIS 风格的 2D 到 3D 几何提升和跨视图关联。默认入口递归发现 `data/work` 下的 `observation.json`，一个 observation 对应一个 GPU 工作项。

权重路径由 `configs/segmentation.json` 配置，默认是：

```text
weights/mosaic3d.ckpt
weights/sam3.pt
weights/recap-clip/
├── open_clip_config.json
├── open_clip_pytorch_model.bin
├── added_tokens.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

ReCap-CLIP 从 `mosaic3d.text_model_path` 配置的平铺本地目录直接加载，默认目录为 `weights/recap-clip`。运行时不再使用 Hugging Face 模型 ID 或缓存结构，也不会自动下载缺失文件。下载目录中的 `configuration.json` 可以保留，但不是当前加载器的必需文件。

先做只读预检和任务分配检查：

```bash
scenecompose segment-observations \
  --observation-root data/work \
  --config configs/segmentation.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 8 \
  --dry-run
```

正式运行时移除 `--dry-run`，或使用：

```bash
bash scripts/run_observation_segmentation.sh data/work 0,1,2,3,4,5,6,7 8
```

默认启用 `--resume`。可用 `--force-stage sam3` 使 SAM3 及后续阶段失效并重算。有效阶段名为 `geometry`、`views`、`mosaic3d`、`sam3`、`fusion` 和 `export`。

每个 observation 的结果写入自身 `segmentation/`。关键文件包括：

- `fusion/point_labels.npz`：点语义、实例、`visibility_count`、`is_observed` 和 `is_core`；
- `fusion/face_labels.npz`：面语义、实例、`visibility_count`、`is_observed` 和 `is_core`；
- `fusion/instances.json`：实例类别、置信度、包围盒、视图和可见点比例；
- `visualization/semantic.ply` 与 `visualization/instances.ply`：人工检查用点云。

全 Scene 调试入口 `segment-scenes` 仍然保留，但不读取 observation 锚点，也不作为默认组合流程。

## Scene 与 Object 组合

组合分为两个独立命令。`build-object-catalog` 将本地 GLB 与 `annotations.json` 按 UID 关联，先执行确定性的 metadata/LVIS 规则，可选使用单次加载的本地 Qwen 模型处理无法确定的对象，最后由 Blender 生成几何档案。`compose-observations` 从语义面标签中提取朝上、已观察且属于 Core 的支撑面，再执行尺寸归一化、边界、碰撞和锚点可见性检查。每个 observation 最多放置一个 Object。

### 本地部署 Qwen（不使用 API）

`configs/composition.json` 默认设置 `llm.enabled: false`，此时只接受高置信规则命中的对象。启用本地 Qwen 前，需要先安装与服务器 CUDA 驱动匹配的 PyTorch，再安装 Transformers 依赖。

Mamba 环境：

```bash
mamba activate scenecompose
python -m pip install "transformers>=4.51" accelerate safetensors
```

Pixi 环境：

```bash
pixi run python -m pip install "transformers>=4.51" accelerate safetensors
```

将 Transformers 格式的指令模型放在项目目录内，例如 `weights/Qwen3-8B`，然后修改配置：

```json
"llm": {
  "enabled": true,
  "backend": "transformers",
  "model_path": "weights/Qwen3-8B",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "local_files_only": true
}
```

模型只在对象目录构建阶段加载一次，代码要求非思考模式输出，并将结果限制在配置中的标准类别内。分类缓存写入 `data/work/object_catalog/llm_classification_cache.json`，不会启动 HTTP 服务，也不会调用外部 API。用法依据 [Qwen 官方 Transformers 指南](https://qwen.readthedocs.io/en/stable/inference/transformers.html)。

### 构建 Object 目录

确保 Objaverse GLB 已位于 `data/obj` 后执行：

```bash
bash scripts/run_object_catalog.sh \
  data/obj \
  data/obj/metadata/annotations.json \
  data/work/object_catalog
```

可先给 `scenecompose build-object-catalog` 增加 `--dry-run`，只检查发现结果和规则分类，不运行 Blender 或 LLM。正式结果为 `data/work/object_catalog/catalog.json`，几何档案和 LLM 分类均支持断点复用。

### 执行组合

```bash
bash scripts/run_composition.sh \
  data/work \
  data/work/object_catalog/catalog.json \
  data/composed
```

成功目录包含 `combined_scene.glb`、`placement.json`、`camera.json` 和 `support_patches.json`。`camera.json` 保存从 observation 锚点朝向 Object 的参考相机，但本阶段不渲染图片。失败任务仍会保留 `placement.json`，其中逐项记录候选对象、支撑面和失败原因。
