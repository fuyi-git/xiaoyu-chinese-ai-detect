# 本目录不含权重文件

这里只放**模型结构配置**（`model_cfg/`）和**分词器配置**（`tok/`）。
权重不进 git 仓库——二进制会让仓库历史永久膨胀，而且每换一版就多压一份。
原因与取舍见仓库根目录 `README.md`。

## 首次使用前拉一次

```bash
python scripts/download.py
```

默认从魔搭 `fuyi163/xiaoyu-chinese-ai-detect` 拉 `quant/int4_packed.pt`
到 `~/.cache/xiaoyu-ai-detect`，并校验 sha256：

```
80a2505fbc922ab33cc4e9875dd87b626225daba83bf980316c1ad07b045f2d9
```

## 三档权重（按需选，不必全下）

| 档 | 仓库内路径 | 体积 | 拉取命令 |
|---|---|---|---|
| int4 轻量档（默认） | `quant/int4_packed.pt` | 60.2 MB | `python scripts/download.py` |
| int8 平衡档 | `quant/int8_packed.pt` | 101.0 MB | `python scripts/download.py --bits 8` |
| fp32 原始精度档 | `model.safetensors` + `config.json` | 390.1 MB | `python scripts/download.py --bits 32` |

三档都是**已经打包好的成品**，不需要自己再打包或量化。fp32 是标准 safetensors
（`from_pretrained` 直读），int4 / int8 是本仓库自带的打包格式，由 `download.py` /
`detect.py` 内置的 `unpack_blob` 解包。

## detect.py 的查找顺序

1. `assets/` 下对应档位的文件（本目录，优先）
2. 环境变量 `XIAOYU_WEIGHTS` 指向的目录
3. `~/.cache/xiaoyu-ai-detect`（`download.py` 的默认落点）

所以下到缓存目录就能直接用，不需要额外配置。
不写 `--bits` 时按 **int4 → int8 → fp32** 取第一个找到的。

## 想改成离线自包含

把权重文件直接放到本目录即可，`detect.py` 会优先读它，之后完全离线可用。
