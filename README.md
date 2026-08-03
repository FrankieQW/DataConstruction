# SceneCompose

SceneCompose is organized as three large stages:

1. adaptive scene partitioning;
2. per-region semantic/instance segmentation and cross-region fusion;
3. region selection, object placement, constraint solving, and physical validation.

## Adaptive Scene Partitioning

This repository currently implements stage 1. It targets Linux servers and runs Blender in background mode. The third-party repositories in the project root remain isolated from the orchestration package.

### Requirements

- Linux
- Python 3.10 or newer
- Blender 4.5 LTS available as `blender` or through `SCENECOMPOSE_BLENDER`
- NumPy in the orchestration environment; Blender's bundled Python must also provide NumPy

#### Third-party source repositories

The complete pipeline expects the following repositories to exist directly under the SceneCompose project root. Run these commands from the project root so the generated directory names match the configured paths:

```bash
git clone https://github.com/NVlabs/WarpConvNet.git
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/VinAIResearch/Open3DIS.git
git clone https://github.com/JinLi998/CoSMo3D.git
git clone https://github.com/NVlabs/Mosaic3D.git
```

These repositories are reserved for the later segmentation and composition stages and remain isolated from the core orchestration package. Adaptive partitioning itself does not import them.

No model weights are needed for partitioning.

### Install

Choose either Mamba or Pixi. Blender is managed separately from the Python environment in both methods.

#### Option A: Mamba

Create and activate a named environment, then install SceneCompose as an editable package:

```bash
mamba create -n scenecompose -c conda-forge python=3.12 pip numpy
mamba activate scenecompose
python -m pip install -e .
export SCENECOMPOSE_BLENDER=/opt/blender/blender
```

See the [Mamba user guide](https://mamba.readthedocs.io/en/stable/user_guide/mamba.html) for shell initialization and environment management.

#### Option B: Pixi

This repository already contains `pyproject.toml`. Run `pixi init` once in the repository root; Pixi adds its workspace configuration and registers the current Python project as an editable dependency. Then install the environment and enter its shell:

```bash
pixi init
pixi install
pixi shell
```

Set the Blender path inside the Pixi shell:

```bash
export SCENECOMPOSE_BLENDER=/opt/blender/blender
```

On later checkouts or after `pixi.lock` is available, use `pixi install` followed by `pixi shell`; `pixi init` is not repeated. See the official [Pixi `pyproject.toml` guide](https://pixi.prefix.dev/latest/python/pyproject_toml/) for details.

### Partition one scene

```bash
bash scripts/run_partition.sh \
  data/scene/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx \
  data/work/EmeraldSquare_Day/partition
```

The equivalent direct command is:

```bash
scenecompose partition \
  --scene data/scene/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx \
  --output data/work/EmeraldSquare_Day/partition \
  --config configs/partition.json \
  --blender "$SCENECOMPOSE_BLENDER"
```

### Partition all scenes

```bash
scenecompose partition-all \
  --scene-root data/scene \
  --output-root data/work \
  --config configs/partition.json \
  --blender "$SCENECOMPOSE_BLENDER" \
  --workers 1
```

or use the wrapper:

```bash
bash scripts/run_partition_all.sh data/scene data/work 1 configs/partition.json
```

`--workers` controls concurrent Blender processes and should be selected from available CPU memory, not GPU count. Each later segmentation process can independently consume a completed Region on one of the eight A100 GPUs.

### Behavior

A scene is kept intact only when all of these conditions hold:

- evaluated triangle count does not exceed `max_triangles_per_region`;
- estimated sampled point count does not exceed `max_estimated_points_per_region`;
- the longest XY extent does not exceed `max_xy_extent_m`.

Such a scene becomes the identity Region `region_full`; it references the original FBX and does not duplicate its geometry. Larger scenes are recursively split along a deterministic spatial median. Core bounds never overlap in area, while context bounds add a configurable halo. All Regions retain the full Z range.

The default intermediate format is `.blend`, which preserves materials while referencing external textures. This avoids embedding the same large textures in every Region. Set `export_format` to `glb` only when a self-contained exchange artifact is required; GLB may duplicate texture payloads across Regions.

See `docs/partition-artifacts.md` for the output contract used by later pipeline stages.
