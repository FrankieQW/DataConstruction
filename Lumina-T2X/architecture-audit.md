# Lumina-Next-T2I 架构审计

## 1. 审计范围与结论

本审计对应 `plan.md` 的第一批执行任务，仅阅读以下路径：

```text
lumina_next_t2i/models/model.py
lumina_next_t2i/models/__init__.py
lumina_next_t2i/train.py
lumina_next_t2i/sample.py
lumina_next_t2i/utils/cli.py
lumina_next_t2i/transport/
```

本批次未修改模型、训练或推理代码，也未运行测试、训练或推理。

核心结论：

1. 上游没有名为 `JointAttention` 的类，实际核心类是 `Attention`。
2. `Attention` 对图像序列 `x` 执行非因果 self-attention，并另外执行 `x query -> Gemma key/value` 的 gated cross-attention。
3. TokenLight 应将 target、source、lighting 和 mask token 拼入同一个 `x` 序列，使其共享 self-attention 的 Q/K/V；不能把 lighting token 填入原文本 cross-attention 接口。
4. source 和 target 可以复用现有 `x_embedder`，无需新增 source patch embedding。
5. 现有 output head 会处理整个 `x` 序列；TokenLight 必须在 output head 前只切出 target token。
6. 上游 linear transport 的时间方向与计划一致：`t=0` 是噪声，`t=1` 是数据，velocity target 是 `x_data - x_noise`。
7. 为保留原始 T2I 路径，建议只给上游 attention/block 增加“不使用文本分支”的可选入口，TokenLight 的序列构造放在独立 wrapper 中。

## 2. 仓库与环境记录

### 2.1 仓库

```text
project:       /mnt/afs_fangwenqi/Lumina-T2X
branch:        main
remote:        https://github.com/Alpha-VLLM/Lumina-T2X
commit:        1c606962f95899da711633ee3a333d21c753e2d9
commit date:   2025-02-16T19:34:43+08:00
plan SHA256:   ef58b4c85b58f5b7766b9e6c5570300862542600ed0a1e77587661c74a9ce262
```

工作树中大量 tracked 文件显示 modified。`git ls-files --eol` 显示索引为 LF、工作区为 CRLF；`git diff --ignore-space-at-eol --quiet` 返回无内容差异。本任务必须继续避免格式化或重写这些无关文件。

### 2.2 Conda 环境

Conda 注册了环境：

```text
/mnt/afs_fangwenqi/miniconda3/envs/lum
```

但该目录目前只有 `conda-meta/` 和 `etc/`，没有 `bin/python`。环境历史仅记录：

```text
conda create -n lum -y
```

因此当前 `lum` 是空环境，尚不能记录 Python、PyTorch、CUDA、flash-attn、diffusers、transformers 和 fairscale 的有效版本。`conda run -n lum` 当前会错误地解析到 base Python，不能用于项目运行。

### 2.3 GPU 与 CUDA

当前执行容器中：

```text
nvidia-smi: 无法连接 NVIDIA driver
nvcc:       未找到
```

这不等价于宿主机没有 GPU，但表示本次会话无法完成 GPU 基线验证。按协作约定，相关验证由用户执行。

### 2.4 Checkpoint 与 VAE

尚未提供 Lumina-Next-T2I checkpoint 和配套 VAE 的绝对路径。在 `/mnt/afs_fangwenqi` 的常见目录及 Hugging Face 缓存中未定位到对应资产；只发现与本项目无关的 LoRA safetensors。

继续实现前需要明确：

```text
Lumina-Next-T2I checkpoint 目录
checkpoint revision 或下载 commit
SDXL VAE 目录
VAE revision
checkpoint 的 model_args.pth
consolidated 文件数量及格式（.pth 或 .safetensors）
```

## 3. Next-DiT 2B 固定结构

模型工厂位于 `models/model.py:994`：

```text
model:       NextDiT_2B_patch2
patch_size:  2 latent pixels
dim:         2304
n_layers:    24
n_heads:     32
head_dim:    72
in_channels: 4（SDXL VAE）
out_channels:8 when learn_sigma=true
```

上游 `forward()` 最终在 channel 维将 8 通道分成两半，只返回前 4 通道，所以 transport 实际收到与 latent 相同 shape 的预测。

### 3.1 分辨率与 token shape

SDXL VAE 空间下采样为 8，Next-DiT latent patch 为 2，因此一个 DiT patch 对应 16 x 16 RGB pixels。

| RGB | VAE latent | patch grid | 单图 token | source + target |
| --- | --- | --- | ---: | ---: |
| 256 x 256 | `[B,4,32,32]` | 16 x 16 | 256 | 512 |
| 512 x 512 | `[B,4,64,64]` | 32 x 32 | 1024 | 2048 |
| 960 x 960 | `[B,4,120,120]` | 60 x 60 | 3600 | 7200 |

patchify 前每个 patch 包含：

```text
patch_size * patch_size * in_channels = 2 * 2 * 4 = 16
```

`x_embedder` 将 `[B,L,16]` 投影为 `[B,L,2304]`。

## 4. Patchify 与 padding

实现位于 `models/model.py:770`。

Tensor 输入路径：

```text
x                         [B,4,H,W]
view/permute/flatten      [B,H/2,W/2,16]
x_embedder                [B,H/2,W/2,2304]
flatten spatial           [B,L,2304]
mask                      [B,L]，全部为 1
freqs_cis                 [1,L,36] complex64
```

List 输入路径允许同一 batch 中的 latent 尺寸不同。每张图独立 patchify，随后用 learned `pad_token` 补到最大 L，并生成 `[B,Lmax]` mask。

TokenLight 第一版使用同分辨率、对齐的 source/target，因此建议使用 Tensor 路径分别调用共享的 `x_embedder`，再手工拼接联合序列。不要把 source 和 target 在 channel 维拼接，否则无法共享预训练 patch embedding，也不满足联合 token 定义。

## 5. Attention 的真实行为

实现位于 `models/model.py:137` 和 `models/model.py:337`。

### 5.1 图像 self-attention

输入：

```text
x          [B,L,2304]
x_mask     [B,L]
freqs_cis  [B or 1,L,36] complex64
```

投影后：

```text
Q [B,L,local_heads,72]
K [B,L,local_kv_heads,72]
V [B,L,local_kv_heads,72]
```

Q/K 应用二维 RoPE，V 不应用。BF16/FP16 使用 `flash_attn_varlen_func`，FP32 使用 PyTorch scaled-dot-product attention。两条路径均为 `causal=False`。

只要 TokenLight 构造：

```text
x_joint = [target, source, light, optional mask]
```

现有 `wq/wk/wv` 就会对所有 token 生成同一套 Q/K/V，target 可以直接关注 source/light/mask，source/light/mask 之间也能相互注意。这符合计划要求的联合注意力。

### 5.2 Gemma cross-attention

当 attention 存在 `wk_y/wv_y` 时，上游还会计算：

```text
Q = image Q
K/V = projected Gemma features
output = image_self_attention + tanh(gate) * text_cross_attention
```

文本 token 没有与图像 token 拼成同一 Q/K/V 序列。因此不能把 lighting token 传给 `y` 并宣称完成 TokenLight 联合注意力。

### 5.3 最小接口改造建议

保留原 `NextDiT.forward(x,t,cap_feats,cap_mask)` 的行为不变。给 `Attention` 和 `TransformerBlock` 增加显式的 optional text 分支：

```text
y is None -> 只执行联合 x self-attention
y exists  -> 保持原图像 self-attention + Gemma cross-attention
```

TokenLight wrapper 调用 `y=None`；原始 Lumina T2I 继续传 `cap_feats/cap_mask`。这样无需加载 Gemma，同时不会破坏原 T2I API 或已有文本权重。

## 6. AdaLN 条件

上游在 `models/model.py:846` 执行：

```text
t_emb          [B,1024]
pooled caption [B,cap_feat_dim]
cap_emb        [B,1024]
adaln_input = t_emb + cap_emb
```

每个 Transformer block 将 `adaln_input` 投影为：

```text
scale_msa, gate_msa, scale_mlp, gate_mlp
```

final layer 也使用同一个 `adaln_input` 做 scale modulation。

TokenLight 第一版不使用文本编码器，因此建议：

```text
adaln_input = t_embedder(t)
```

lighting 条件通过联合 lighting tokens 提供，不再池化后加入 AdaLN。原 checkpoint 中 `cap_embedder` 及 text K/V 权重保留用于严格加载和原始 T2I 路径，但 TokenLight forward 不调用它们。

这是 backbone substitution 中的重要行为变化，后续需要由用户通过单样本过拟合验证。若效果不足，只能把空文本条件作为明确的 ablation，不能悄悄恢复 Gemma 并称为默认设计。

## 7. RoPE 映射

实现位于 `models/model.py:916`。2D RoPE 将 head_dim 72 分成高度与宽度两部分，最终每个 token 获得 36 个 complex64 旋转因子。

TokenLight 联合序列建议：

```text
target patch (row,col) -> rope(row,col)
source patch (row,col) -> 同一个 rope(row,col)
mask patch   (row,col) -> 同一个 rope(row,col)
lighting attribute     -> identity rotation（complex ones，等价于零坐标）
```

因此在 960 px 下：

```text
target_rope [B,3600,36]
source_rope [B,3600,36]  # target_rope clone/reuse
light_rope  [B,Nlight,36] # complex ones
mask_rope   [B,Nmask,36]
joint_rope  [B,7200+Nlight+Nmask,36]
```

属性 token 使用 identity rotation，避免伪造二维空间位置；token 类型和 slot 由 lighting encoder 的独立 projection 与固定顺序表达。

上游 RoPE 存在两个实现注意点：

1. `precompute_freqs_cis()` 内部直接调用 `.cuda()`，模型构造依赖 CUDA，无法纯 CPU 初始化。
2. `self.freqs_cis` 是普通 tensor 属性，不是 registered buffer，不进入 state dict；forward 中手工迁移 device。

TokenLight 改造时不应顺手重构无关上游行为，但新建的 Fourier matrix 必须按计划注册为 buffer。

## 8. Output head

`ParallelFinalLayer` 位于 `models/model.py:627`，输入 `[B,L,2304]`，输出：

```text
[B,L,patch_size*patch_size*out_channels]
= [B,L,32] when learn_sigma=true
```

上游随后将整个 L unpatchify。TokenLight 不得对 joint sequence 直接 unpatchify，正确边界是：

```text
x_joint after 24 blocks
-> x_target = x_joint[:, :target_len]
-> final_layer(x_target, t_emb)
-> unpatchify target only
-> if learn_sigma: channel chunk and keep first 4
-> velocity [B,4,Hlatent,Wlatent]
```

source、lighting 和 mask token 不进入 output head。

## 9. 推荐的 TokenLight tensor 流

以同分辨率 batch 为例：

```text
z_target_tau [B,4,H,W]
z_source     [B,4,H,W]
light_values + known/valid masks
fixture_mask optional
```

建议 wrapper 流程：

```text
1. target/source 分别用同一个 x_embedder -> [B,L,2304]
2. lighting encoder -> [B,Nlight,2304] + [B,Nlight] mask
3. mask embedder（启用时）-> [B,Nmask,2304]
4. 拼接 x_joint、joint_mask、joint_rope
5. t_embedder(t) -> [B,1024]
6. 24 个 block 执行 y=None 的联合 self-attention
7. 只切 target token
8. final_layer + unpatchify -> velocity [B,4,H,W]
```

第一版不增加 source/target modality embedding，也不增加 learned task token。对应空间 patch 依靠相同 RoPE 对齐；source/target 的角色由序列区间和“只有 target 进入 output head”定义。

## 10. Flow matching 时间方向

`transport/path.py:25-31` 定义：

```text
alpha(t) = t
sigma(t) = 1 - t
```

`transport/path.py:116-139` 因此得到：

```text
x_t = t * x_data + (1-t) * x_noise
u_t = x_data - x_noise
```

`transport/transport.py:130-164` 使用模型预测与 `u_t` 的 MSE。采样器默认从 `t=0` 正向积分到 `t=1`，输入标准高斯噪声，得到数据 latent。

结论：计划中的公式与 Lumina 原生 Linear + velocity 完全同向，无需时间反转边界层：

```text
tau=0 -> noise
tau=1 -> target data
velocity_target = z_target - epsilon
```

TokenLight 可以复用 transport 的数学约定，但建议在独立 `tokenlight/flow.py` 中显式实现和记录，避免原训练器的文本接口、list batch 和日志行为泄漏到新路径。

## 11. CFG 差异

原 `forward_with_cfg()`：

1. 复制 latent batch；
2. 使用 conditional/empty-caption 两组 Gemma feature；
3. 只对输出的前三个 channel 做 CFG；
4. 将 guided 结果复制回两半。

该实现不适合 TokenLight。TokenLight CFG 必须建立两个除 lighting edit 外完全一致的分支：

```text
conditional:   target noise + source + lighting + fixture mask
unconditional: target noise + source + dropped lighting + fixture mask
```

并对完整 4-channel velocity 使用：

```text
v = v_source_only + w * (v_source_and_light - v_source_only)
```

因此后续应在 `tokenlight/sampler.py` 独立实现，不修改或复用原 `forward_with_cfg()` 的三通道语义。

## 12. Checkpoint 兼容策略

上游 checkpoint 包含：

```text
model_args.pth
consolidated[ _ema].RR-of-WW.pth
```

CLI 路径也支持对应 `.safetensors`。原推理使用 `strict=True`。

推荐迁移顺序：

1. 按 `model_args.pth` 构造与 checkpoint 完全一致的原 NextDiT 2B。
2. 严格加载所有上游参数，先确认 checkpoint 自身无 missing/unexpected key。
3. TokenLight wrapper 复用已加载的 `x_embedder`、`t_embedder`、24 blocks 和 final layer。
4. 保留但在 TokenLight forward 中不调用 `cap_embedder`、`wk_y`、`wv_y`、text norm 和 gate。
5. 新增 lighting/mask 参数单独初始化，并输出精确的新增参数清单。
6. checkpoint loader 同时支持 `.pth` 与 `.safetensors`，但一次只接受一种明确格式。
7. 记录 checkpoint shard 数；model-parallel world size 必须与 shard 布局兼容。

如果直接用不含文本模块的新类加载上游 state dict，会产生大量 unexpected text keys，不符合计划中的加载审计要求。因此第一版应保留上游文本参数以获得可解释、可逆的迁移。

## 13. 上游实现风险

### 13.1 Activation checkpointing 属性名不一致

`train.py:402` 引用：

```text
model.transformer_blocks
```

但 `NextDiT` 实际模块名为：

```text
model.layers
```

模型已提供 `get_checkpointing_wrap_module_list()` 返回 `layers`。TokenLight 训练入口应调用该方法，不能复制原属性名，否则开启 checkpointing 会报 `AttributeError`。

### 13.2 VAE 未显式冻结

原训练代码在 `torch.no_grad()` 中编码图像，但没有明确执行：

```text
vae.eval()
vae.requires_grad_(False)
```

TokenLight 必须显式执行二者，满足计划的冻结要求。

### 13.3 CPU 构造不可用

RoPE 预计算硬编码 `.cuda()`。这会影响 CPU schema/checkpoint 工具。后续若需要 CPU-only checkpoint 审计，应把 device 选择做成兼容改动，但必须保证原 GPU行为不回归。

### 13.4 960 px attention 长度

960 px 时仅 target+source 已有 7200 token。加入 mask patch 若再增加 3600 token，会显著放大 attention 成本。fixture mask token 的压缩方式必须在 Phase 2/3 前确认；不能默认把全分辨率 mask patch 全量追加后直接进入正式训练。

## 14. 第一批建议的原始 T2I 基线命令

以下命令由用户在修复 `lum` 环境并填写实际路径后执行。本次审计未执行。

先确认 checkpoint 至少包含：

```bash
ls <LUMINA_CHECKPOINT>/model_args.pth
ls <LUMINA_CHECKPOINT>/consolidated_ema.00-of-01.pth
```

如果实际为 `.safetensors`，应使用 `lumina_next` CLI 路径；不要把扩展名直接替换后交给 `sample.py`。

准备一个单行 prompt 文件后，运行原始开发推理：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n lum python -u lumina_next_t2i/sample.py \
  --ckpt <LUMINA_CHECKPOINT> \
  --image_save_path <BASELINE_OUTPUT> \
  --caption_path <PROMPTS_TXT> \
  --precision bf16 \
  --num_gpus 1 \
  --num_sampling_steps 50 \
  --cfg_scale 4.0 \
  --seed 12345 \
  --resolution 1024:1024x1024 \
  --time_shifting_factor 1.0 \
  --batch_size 1
```

用户需要保存：

```text
prompt 文本
seed
完整命令
checkpoint revision
生成图片
nvidia-smi GPU/driver 信息
峰值显存
第二次同 seed 输出及文件 hash
```

注意：原 `sample.py` 会加载 `google/gemma-2b` 和 `stabilityai/sdxl-vae`。离线运行前必须保证二者已在本地缓存，或改用原 CLI 的本地 checkpoint 参数；本阶段不下载依赖或模型。

## 15. 第一批停止条件

架构审计已经完成，但 Phase 0 尚未通过，当前不得开始 TokenLight 模型改造。继续 Phase 1 至 Phase 3 前需要用户完成：

1. 使 `lum` 环境具备有效 Python 和依赖；
2. 提供 Lumina-Next-T2I checkpoint 绝对路径；
3. 提供 SDXL VAE 绝对路径；
4. 在可访问 GPU 的环境中运行原始 T2I baseline；
5. 确认 baseline 严格加载、BF16 无 NaN/Inf、固定 seed 可复现；
6. 审阅并确认本架构审计中的最小改造方向。

## 16. 后续状态

上述停止条件记录的是架构审计完成时的状态。用户随后已明确认可本审计并确认继续执行 `plan.md`，因此代码实现已进入后续阶段。checkpoint、VAE、GPU baseline 和全部测试仍由用户准备并执行；它们尚未在本次代码修改环境中验证。
