# TokenLight 数据上的 FLUX.1-Kontext-dev LoRA

这套入口与原来的 Lumina-Next-T2I/TokenLight 训练并存，不覆盖 `config.yaml`、
`train_tokenlight.py` 或已有 checkpoint。Blender 数据生成也不需要重跑。

## 1. 实际训练关系

每条 TokenLight 样本被转换成 FLUX Kontext 的三元组：

```text
condition image = TokenLight source_image
instruction     = 光照任务及数值组成的英文编辑指令
target image    = TokenLight target_image
```

四类任务仍使用原有线性 EXR 合成公式：`ambient_scale`、`global_diffuse`、
`add_light` 和 `in_scene_light`。EXR 仍经过原配置中的 exposure、Reinhard、中心裁剪和
缩放。包装数据集用固定的 `(seed, index)` 取样，因此多卡重跑和断点恢复不会改变同一
索引对应的 source/target/instruction。

标准 Kontext 只接收 condition image 和文本指令，不能直接接收 TokenLight 的 fixture
mask。当前 `in_scene_light` 依靠 source 中可见的灯具和指令学习；mask 保留在原始数据中，
但没有送入 FLUX。若以后必须显式使用 mask，需要增加一个图像条件 adapter，这已超出
纯 LoRA 微调的结构范围。

训练时冻结 FLUX transformer 的原权重、VAE、CLIP 和 T5，只训练 YAML 中列出的 LoRA
层。target latent 按 rectified-flow 路径加噪，condition latent 作为另一段 Kontext 图像
token，监督目标为 `noise - clean_target_latent`。

## 2. 权重唯一推荐位置

把 Hugging Face 上 `black-forest-labs/FLUX.1-Kontext-dev` 的完整 Diffusers 格式
snapshot 下载到：

```text
/mnt/afs_fangwenqi/models/FLUX.1-Kontext-dev/
```

最终至少应为：

```text
/mnt/afs_fangwenqi/models/FLUX.1-Kontext-dev/
  model_index.json
  scheduler/
  transformer/
  vae/
  text_encoder/
  text_encoder_2/
  tokenizer/
  tokenizer_2/
```

不要只下载 BFL 原生推理仓库使用的 `flux1-kontext-dev.safetensors` 和 `ae.safetensors`；
本训练入口还需要两套 tokenizer/text encoder 及 Diffusers 配置。模型是 gated 模型，下载前
需在 Hugging Face 页面接受访问条件。代码固定使用 `local_files_only=True`，训练时不会
联网补文件。

已经下载的 `flux` 文件夹是 BFL 参考推理源码；它可以保留，但当前训练入口不需要修改
它，也不把权重放进源码目录。

## 3. 单独环境

不要修改已有 `lum` 环境：

```bash
conda create -n flux-kontext python=3.11 -y
conda activate flux-kontext
cd /mnt/afs_fangwenqi/Lumina-T2X
pip install -r lumina_next_t2i/requirements-flux-kontext.txt
```

安装与服务器 CUDA 匹配的 PyTorch wheel 时，以服务器现有 CUDA/驱动为准。代码没有自动
安装或下载依赖。

## 4. 配置

独立配置是 `lumina_next_t2i/flux_kontext_lora.yaml`。默认从原
`lumina_next_t2i/config.yaml` 读取 TokenLight dataset root、manifest、exposure 和四类
光照范围。若要使用不同 manifest，可在 `paths.train_manifest` 等字段直接覆盖。

默认先用 512 分辨率、LoRA rank 16、BF16、gradient checkpointing 验证链路。8 卡时：

```text
每卡 micro batch = 1
梯度累积          = 4
GPU 数            = 8
global batch       = 32
```

512 smoke 稳定后再考虑 1024。960 虽能被 16 整除，但不是 Kontext 推荐的方形训练尺寸；
切换分辨率应新建 run，不要从不同分辨率 checkpoint 恢复 optimizer 状态。

## 5. 由你在服务器执行的验收

只检查配置及权重目录契约，不加载模型：

```bash
cd /mnt/afs_fangwenqi/Lumina-T2X
conda activate flux-kontext
python lumina_next_t2i/train_flux_kontext_lora.py \
  --config lumina_next_t2i/flux_kontext_lora.yaml \
  --check-config
```

读取 train/validation 的第一条确定性样本，核对 shape、范围和编辑指令：

```bash
python lumina_next_t2i/train_flux_kontext_lora.py \
  --config lumina_next_t2i/flux_kontext_lora.yaml \
  --check-data
```

先做单卡少步 smoke 时，把 YAML 的 `max_steps` 临时改小，并运行：

```bash
CUDA_DEVICES=0 NUM_PROCESSES=1 bash lumina_next_t2i/train_flux_kontext_lora.sh
```

正式八卡：

```bash
bash lumina_next_t2i/train_flux_kontext_lora.sh
```

输出目录为：

```text
/mnt/afs_fangwenqi/Lumina-T2X/outputs/flux_kontext_tokenlight/
  flux_kontext_tokenlight_lora/
    config.yaml
    checkpoint-500/
    checkpoint-1000/
    final/
```

恢复时把 `paths.resume_checkpoint` 设为 `latest`，或指向一个完整的 `checkpoint-N` 目录。
不要指向其中的单个 safetensors 文件。

## 6. 推理

下面的例子读取 TokenLight `ambient.exr`，使用与训练相同的 exposure/Reinhard 预处理：

```bash
python lumina_next_t2i/infer_flux_kontext_lora.py \
  --config lumina_next_t2i/flux_kontext_lora.yaml \
  --lora /mnt/afs_fangwenqi/Lumina-T2X/outputs/flux_kontext_tokenlight/flux_kontext_tokenlight_lora/final \
  --source /path/to/component/ambient.exr \
  --ambient-scale 0.5 \
  --output /mnt/afs_fangwenqi/tokenlighttest/flux_ambient_05.png
```

其他任务可通过 `--prompt` 传入完整编辑指令。推理会在 PNG 旁写一个 `.png.json`，记录
base model、LoRA、prompt、seed、steps、guidance、resolution 和 exposure。

## 7. 当前验收边界

本地工作区未运行 Python/GPU 检查，且权重尚未提供，因此这里只完成代码接线，没有宣称
训练已通过。服务器上至少应依次通过 `--check-config`、`--check-data`、单卡少步 smoke，
再启动八卡正式训练。首次 smoke 若遇到 Diffusers API 版本报错，应记录完整 traceback 和
`python -c "import diffusers; print(diffusers.__version__)"` 的输出。
