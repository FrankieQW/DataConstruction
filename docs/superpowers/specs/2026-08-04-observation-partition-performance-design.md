# Observation Partition 性能优化设计

## 目标

解决大型 FBX 在第若干 observation 导出时长时间停留于 Blender glTF exporter 的问题，同时保持 `observation.json`、`partition/scene_partition.glb` 与 `partition/source_faces.npz` 契约不变。

## 已确认根因

当前实现先提取一次全 Scene 几何，随后对每个 observation、每个 source 再次调用 evaluated mesh 提取，并为每个命中的 source 创建一个 Blender object。密集局部区域会让 glTF exporter 处理大量 object/primitive。输出 mesh 还复制 source 的全部材质槽，包括本 observation 未使用的材质。无阶段进度日志和末尾统一 publish 会放大“卡住”的观感。

## 设计

1. FBX 导入后为每个 evaluated source 建立不可变 NumPy 缓存。世界顶点和统计量使用 float32，索引使用 int32；锚点采样和 observation 选择共用该缓存。
2. 三角形扇形测试分七次执行质心、三个顶点与三个边中点，不构造 `[F, 7, 3]` 大数组。
3. 一个 observation 的所有选中 source 合并为单个 Blender mesh，同时保持每行输出三角形到 source object/instance/polygon 的映射。
4. 只挂载实际命中 polygon 使用的材质。正常材质直接复用；仅对包含 `size == (0, 0)` Image Texture 的材质建立私有副本并移除无效节点，其他纹理和 UV 保持不变。
5. 日志记录 import、geometry cache、floor/anchor、每个 observation build/export/mapping 的耗时及三角形、source、材质数量。
6. 失败行为与原实现一致：staging 目录在异常时删除，成功后原子发布。

## 非目标

- 不改为 separate glTF 或 `.blend`。
- 不改变 observation 采样结果。
- 不实现跨 observation 共享 GLB 内嵌纹理。
- 不改变分割和组合代码。

## 验证

按用户要求不在本地运行 Blender 或数据测试。执行 Python AST、配置 JSON、文档与静态契约检查；服务器端用同一命令比较日志、耗时和产物。
