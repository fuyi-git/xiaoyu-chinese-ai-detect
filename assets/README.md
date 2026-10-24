# 本目录不含权重文件

这里只放**模型结构配置**（`model_cfg/`）和**分词器配置**（`tok/`）。
int4 权重 `int4_packed.pt`（约 63MB）不进 git 仓库——二进制会让仓库历史永久膨胀，
而且每换一版权重就多压一份。原因与取舍见仓库根目录 `README.md`。

## 首次使用前拉一次

```bash
python scripts/download.py
```

它会从魔搭 `yanxizhishang/xiaoyu-chinese-ai-detect` 拉 `quant/int4_packed.pt`
到 `~/.cache/xiaoyu-ai-detect`，并校验 sha256：

```
80a2505fbc922ab33cc4e9875dd87b626225daba83bf980316c1ad07b045f2d9
```

## detect.py 的查找顺序

1. `assets/int4_packed.pt`（本目录，优先）
2. 环境变量 `XIAOYU_WEIGHTS` 指向的目录
3. `~/.cache/xiaoyu-ai-detect`（`download.py` 的默认落点）

所以下到缓存目录就能直接用，不需要额外配置。

## 想改成离线自包含

把权重文件直接放到本目录即可，`detect.py` 会优先读它，之后完全离线可用。
