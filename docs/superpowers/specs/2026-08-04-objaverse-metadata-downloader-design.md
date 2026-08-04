# Objaverse Metadata Downloader Design

## 目标

根据 `Objaverse.md` 中已有 GLB 清单，下载 Objaverse 1.0 官方 metadata 分片和 LVIS 类别文件，仅保留清单所引用对象的 annotation，并生成便于人工检查和后续 Scene/Object 组合使用的本地元数据索引。

脚本不下载 GLB、不下载模型权重、不调用 Sketchfab API，所有缓存与输出均位于用户指定的项目目录。

## 输入

命令入口：

```bash
python scripts/download_objaverse_metadata.py \
  --manifest /path/to/Objaverse.md \
  --output data/obj/metadata
```

清单按行解析。符合以下形式的行视为对象记录：

```text
000-000/0000ecca9a234cae994be239f6fec552.glb
```

规则：

- 分片必须匹配 `NNN-NNN`。
- 文件扩展名不区分大小写，但只接受 `.glb`。
- UID 取文件名去掉扩展名后的完整字符串，不限定为 32 位十六进制。
- 单独的分片标题、空行、Markdown 标记和其他文本忽略。
- 相同 `(shard, uid)` 只处理一次。
- 同一 UID 若出现在多个分片，立即报错，不猜测正确分片。
- 清单不存在或没有有效 GLB 记录时退出失败。

## 官方数据源

metadata 分片 URL：

```text
https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/<shard>.json.gz
```

LVIS annotation URL：

```text
https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz
```

URL 基地址可通过命令行覆盖，用于镜像或离线 HTTP 服务，但默认只使用上述官方数据源。

## 输出结构

```text
data/obj/metadata/
|-- cache/
|   |-- metadata/
|   |   `-- 000-xxx.json.gz
|   `-- lvis-annotations.json.gz
|-- annotations.json
|-- annotations.jsonl
|-- sample.json
|-- lvis_categories.json
|-- missing_uids.txt
`-- summary.json
```

### `annotations.json`

以 UID 为键的完整 annotation 字典。原始 Objaverse 字段不裁剪，另外加入 `_scenecompose` 字段：

```json
{
  "<uid>": {
    "name": "...",
    "description": "...",
    "tags": [],
    "categories": [],
    "license": "...",
    "_scenecompose": {
      "uid": "<uid>",
      "shard": "000-000",
      "glb_relative_path": "000-000/<uid>.glb",
      "lvis_categories": []
    }
  }
}
```

若官方 annotation 已存在 `_scenecompose` 字段，脚本退出失败，避免静默覆盖原始数据。

### `annotations.jsonl`

每行一个对象，内容与 `annotations.json` 中对应值一致，额外将 `uid` 放在顶层。记录按清单相对路径稳定排序，便于流式读取。

### `sample.json`

默认保存排序后前 20 条完整 annotation。`--sample-size` 可修改数量，设为 0 时生成空数组。该文件用于快速查看 metadata 字段，不影响完整索引。

### `lvis_categories.json`

保存清单 UID 到 LVIS 类别数组的反向映射。没有 LVIS 类别的 UID不写入该文件，但其 annotation 中 `_scenecompose.lvis_categories` 仍为空数组。

### `missing_uids.txt`

逐行列出未在对应官方 metadata 分片中找到的 UID。缺失对象不会阻止其他结果生成；命令默认以退出码 2 结束。使用 `--allow-missing` 时退出码为 0，并仍在 summary 中报告缺失数量。

### `summary.json`

记录 schema 版本、输入清单绝对路径、对象数、分片数、找到/缺失数量、LVIS 命中数量、缓存命中/下载数量、生成文件相对路径以及警告。为保证结果可复现，不写运行时间戳。

## 下载与缓存

- 使用 Python 标准库 `urllib.request`，不增加项目运行依赖。
- 下载先写同目录唯一 `.part` 临时文件。
- 下载完成后验证 gzip 可完整读取，顶层 JSON 必须是对象，再原子替换缓存文件。
- 已存在缓存也必须验证；无效缓存移动为 `.invalid` 后重新下载。
- 每个文件默认最多尝试 3 次，使用有限指数退避；`--retries` 和 `--timeout` 可配置。
- 日志只打印分片、状态和错误，不打印 annotation 全文。
- 不自动删除有效缓存；重复运行只读取已有缓存。
- `--force-download` 强制重新下载清单涉及的分片和 LVIS 文件。

## 数据处理

1. 解析并验证清单。
2. 按分片分组 UID。
3. 串行下载/验证 metadata 分片。第一版不并发下载，避免对官方服务产生突发请求。
4. 从每个分片提取目标 UID，同时保留完整原始 annotation。
5. 下载/读取 LVIS 类别到 UID 列表，将其反向索引到清单 UID。
6. 注入 `_scenecompose` 路径和类别信息。
7. 先在输出目录写临时文件，全部序列化成功后逐个原子替换正式索引文件。
8. 根据缺失 UID 决定退出码。

## 命令行参数

```text
--manifest PATH             必填，Objaverse.md 或同格式清单
--output PATH               默认 data/obj/metadata
--sample-size INT           默认 20，必须非负
--timeout FLOAT             默认 60 秒，必须大于 0
--retries INT               默认 3，必须至少为 1
--base-url URL              默认官方 Objaverse Hugging Face 根 URL
--force-download            忽略有效缓存并重新下载
--allow-missing             缺失 UID 时仍返回退出码 0
--skip-lvis                 不下载 LVIS 文件，所有 LVIS 类别为空
```

## 模块边界

第一版使用单个脚本，但保持纯函数边界：

- `parse_manifest`：清单文本到不可变对象记录。
- `group_by_shard`：稳定分组并检查冲突。
- `ensure_cached_gzip_json`：下载、重试、校验和原子缓存。
- `extract_annotations`：按分片过滤 UID。
- `invert_lvis_annotations`：类别到 UID 映射反转。
- `build_outputs`：注入 SceneCompose 字段并构建所有输出 payload。
- `write_outputs_atomic`：输出序列化与原子替换。
- `main`：参数校验、日志、退出码。

网络和文件系统只由缓存及输出函数访问，其余逻辑保持纯 Python，方便后续由用户测试。

## 错误处理

以下情况立即失败且不发布新的正式索引：

- 清单格式无有效对象记录；
- UID 跨分片冲突；
- 下载重试耗尽；
- gzip 损坏或顶层 JSON 不是对象；
- annotation 不是 JSON object；
- 原始 annotation 包含保留字段 `_scenecompose`；
- 输出目录无法创建或原子替换失败。

单个 UID 不存在属于可报告的数据缺失，不影响其余 UID 的索引生成。

## 非目标

- 不下载、移动或修改 GLB。
- 不调用 LLM、CLIP、SAM3 或 Mosaic3D。
- 不从 GLB 渲染图推断类别。
- 不修改官方 annotation 文本。
- 不决定 Object 应放在桌面、地面或墙面；该推理属于后续组合阶段。

## 验收方式

由用户在可联网环境运行脚本并检查：

- 缓存只出现在指定 `--output/cache` 中；
- `annotations.json` 条数等于清单 UID 数减去 missing 数；
- `sample.json` 可直接看到 Objaverse 原始字段；
- 同一命令重复运行显示 cache hit 且不重新下载；
- 非标准 UID保留并在找到或缺失列表中明确出现；
- 中断下载不会留下被误认为有效缓存的正式 `.json.gz`。
