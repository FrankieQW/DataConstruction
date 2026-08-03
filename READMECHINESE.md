# SceneCompose

[English README](README.md)

SceneCompose 整体分为三个主要阶段：

1. Scene 自适应切分；
2. 对各个 Region 进行语义/实例分割，并完成跨 Region 融合；
3. Region 选择、Object 放置、约束求解和物理验证。

## Scene 自适应切分

当前仓库实现了第一阶段。代码面向 Linux 服务器，通过 Blender 后台模式运行。项目根目录中的第三方代码仓库与 SceneCompose 编排包保持隔离，不会被直接导入。

### 环境要求

- Linux
- Python 3.10 或更高版本
- Blender 4.5 LTS，可通过 `blender` 命令调用，或由 `SCENECOMPOSE_BLENDER` 环境变量指定
- SceneCompose Python 环境需要安装 NumPy；Blender 内置的 Python 也必须提供 NumPy

Scene 切分阶段不需要任何模型权重。

### 安装

可以选择 Mamba 或 Pixi 中的任意一种方式。无论使用哪种方式，Blender 都独立于 Python 环境管理。

#### 方式一：使用 Mamba

创建并激活命名环境，然后以 editable 模式安装 SceneCompose：

```bash
mamba create -n scenecompose -c conda-forge python=3.12 pip numpy
mamba activate scenecompose
python -m pip install -e .
export SCENECOMPOSE_BLENDER=/opt/blender/blender
```

Shell 初始化和环境管理方法参见 [Mamba 官方用户指南](https://mamba.readthedocs.io/en/stable/user_guide/mamba.html)。

#### 方式二：使用 Pixi

当前仓库已经包含 `pyproject.toml`。首次配置时，在仓库根目录运行一次 `pixi init`；Pixi 会加入 workspace 配置，并将当前 Python 项目注册为 editable 依赖。随后安装环境并进入 Pixi Shell：

```bash
pixi init
pixi install
pixi shell
```

进入 Pixi Shell 后设置 Blender 路径：

```bash
export SCENECOMPOSE_BLENDER=/opt/blender/blender
```

后续重新拉取项目，或者仓库中已经存在 `pixi.lock` 时，只需运行 `pixi install` 和 `pixi shell`，不需要重复执行 `pixi init`。详细说明参见 Pixi 官方的 [`pyproject.toml` 指南](https://pixi.prefix.dev/latest/python/pyproject_toml/)。

### 切分单个 Scene

```bash
bash scripts/run_partition.sh \
  data/scene/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx \
  data/work/EmeraldSquare_Day/partition
```

对应的直接调用命令如下：

```bash
scenecompose partition \
  --scene data/scene/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx \
  --output data/work/EmeraldSquare_Day/partition \
  --config configs/partition.json \
  --blender "$SCENECOMPOSE_BLENDER"
```

### 批量切分全部 Scene

```bash
scenecompose partition-all \
  --scene-root data/scene \
  --output-root data/work \
  --config configs/partition.json \
  --blender "$SCENECOMPOSE_BLENDER" \
  --workers 1
```

也可以使用封装脚本：

```bash
bash scripts/run_partition_all.sh data/scene data/work 1 configs/partition.json
```

`--workers` 控制同时运行的 Blender 进程数量，应根据服务器可用 CPU 内存设置，而不是根据 GPU 数量设置。完成切分后，后续语义分割阶段可以将不同 Region 独立分配到八张 A100 GPU 上处理。

### 切分行为

只有同时满足以下全部条件时，Scene 才会保持完整而不进行切分：

- evaluated mesh 的三角形数量不超过 `max_triangles_per_region`；
- 估算采样点数量不超过 `max_estimated_points_per_region`；
- XY 平面上的最长边不超过 `max_xy_extent_m`。

满足这些条件的 Scene 会成为名为 `region_full` 的 identity Region。该 Region 直接引用原始 FBX，不会复制 Scene 几何。

较大的 Scene 会按照确定性的空间中位数递归切分。各 Region 的 Core 区域在面积上互不重叠，Context 区域则会在 Core 外增加可配置的 Halo，用于提供边界上下文。所有 Region 都保留 Scene 的完整 Z 轴范围，不会沿高度方向切开地面、桌面、墙体等空间关系。

切分归属由三角形在世界坐标中的 XY 质心决定。跨越 Region 边界的三角形会保持完整，不进行几何裁断，从而保留原始拓扑、UV 和材质关系。

### 中间结果格式

默认中间格式是 `.blend`。这种格式可以保留材质并继续引用外部纹理，避免在每个 Region 中重复嵌入大型纹理。

只有确实需要自包含交换文件时，才建议将配置中的 `export_format` 设置为 `glb`。GLB 可能在多个 Region 中重复写入纹理数据，显著增加存储占用。

切分输出的主要文件包括：

```text
<partition-output>/
|-- regions.json
|-- summary.json
`-- regions/
    |-- region_full/
    |   `-- manifest.json
    `-- region_<content-hash>/
        |-- manifest.json
        |-- source_faces.npz
        `-- scene_region.blend
```

其中：

- `regions.json` 是后续语义分割和组合阶段的主要输入契约；
- `summary.json` 记录原始 Scene、实例化几何、规模和切分统计；
- `scene_region.blend` 保存一个自适应切分 Region 的 Core 与 Halo 几何；
- `source_faces.npz` 保存导出三角形到原始 Scene 对象、实例和 polygon 的映射；
- `manifest.json` 记录 Region 边界、所有权、输出对象顺序和相关文件路径。

更完整的中间结果定义参见 [自适应切分产物说明](docs/partition-artifacts.md)。
