# TokenLight-Lumina 实现假设

1. 本实现是 TokenLight 方法在公开 Lumina-Next-T2I 2B backbone 上的替代实现，不是未公开 Adobe text-to-video 权重的精确复现。
2. 第一版 `max_lights=3` 是可配置工程选择；训练和推理必须使用相同值。
3. 没有额外 task token 或 source/target modality embedding。任务由有效的属性 token 类型表达。
4. Source、target 共用 Lumina 的 `x_embedder`，对应 patch 共用二维 RoPE 坐标。
5. Lighting scalar 使用 `sin(Bx), cos(Bx)`，固定 Gaussian Fourier buffer；每个 scalar 有独立投影。
6. Flow 方向固定为 noise at `t=0`、data at `t=1`，预测 `data-noise` velocity，Euler 从 0 积分到 1。
7. CFG 只屏蔽 lighting token 和 fixture-mask token，始终保留 source。
8. 当前运行入口只支持单卡 `model_parallel_size=1`。多卡 FSDP/TP 尚未在 TokenLight 入口实现。
9. 当前单物体 Blender renderer 使用程序化球形可见灯具生成 in-scene fixture 数据；艺术家制作的真实室内灯具仍不在当前范围内。
10. 正式控制连续性指标需要固定轨迹数据；当前通用评测只计算 PSNR、SSIM 和可选 LPIPS。
