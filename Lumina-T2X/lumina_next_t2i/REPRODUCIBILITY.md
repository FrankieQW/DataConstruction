# TokenLight-Lumina 可复现性记录

每次正式 run 复制本模板并填写：

```text
run_id:
parent_run_id:
created_at:

Lumina-T2X git commit: 1c606962f95899da711633ee3a333d21c753e2d9
Lumina checkpoint path/revision:
VAE path/revision:

Python:
PyTorch:
CUDA runtime:
CUDA driver:
flash-attn:
GPU:

config.yaml SHA256:
train manifest SHA256:
validation manifest SHA256:
test manifest SHA256:

start checkpoint:
selected checkpoint:
seed:
resolution:
```

训练入口会自动把 git commit、PyTorch/CUDA/GPU、config SHA256、train/validation manifest SHA256 和恢复来源写入 run 目录的 `run_metadata.json`。Driver、flash-attn、模型 revision 与 test manifest hash 仍需用户补充。

