# 晓语 · 中文 AI 检测（技能包）

WorkBuddy 技能形态的中文「AI 生成 / 人写」检测器。当前模型版本 **V3.3**。

给一段中文文本，输出**一个整篇 AI 生成概率**和三档判决，并逐句给出**句头概率**（仅供参考）。

---

## 安装

把本目录放到 WorkBuddy 的技能目录下：

```
~/.workbuddy/skills/xiaoyu-chinese-ai-detect
```

Windows 对应 `C:\Users\<你的用户名>\.workbuddy\skills\xiaoyu-chinese-ai-detect`。

目录名不影响识别，**技能名取自 `SKILL.md` 里的 `name` 字段**。

## 首次使用：先拉一次权重

本仓库**不包含**权重文件（二进制不进 git，避免仓库历史被反复撑大）。
首次使用前跑一次：

```bash
python scripts/download.py
```

默认拉 **int4 轻量档**（60.2 MB，速度最快），从魔搭
[fuyi163/xiaoyu-chinese-ai-detect](https://modelscope.cn/models/fuyi163/xiaoyu-chinese-ai-detect)
拉到 `~/.cache/xiaoyu-ai-detect`，并做 sha256 校验。

**想要更高保真可以换一档**（用体积换精度，按需选，不必全下）：

| 档 | 体积 | 怎么拉 | 和 int4 比分数差别 |
|---|---|---|---|
| int4 轻量档（默认） | 60.2 MB | `python scripts/download.py` | 基准 |
| int8 平衡档 | 101.0 MB | `python scripts/download.py --bits 8` | ≤ 0.55 个百分点 |
| fp32 原始精度档 | 390.1 MB | `python scripts/download.py --bits 32` | 同上 |

拉完打分时用 `--bits` 指定档位（不写就是 int4）：

```bash
python scripts/detect.py "文本" --bits 8
```

`python scripts/download.py --check` 一次列出三档本地有没有、sha256 对不对。

之后**纯本地 CPU 推理，不联网，不上传任何文本**。

> **若下载报 404 / `record not found`**：是模型仓库暂不可用（尚未发布或正在迁移），不是你操作错了。
> 脚本会自动重试 3 次；仍失败就按提示手动下载对应文件，放到 `~/.cache/xiaoyu-ai-detect/`。
>
> 想改成离线自包含：把权重文件直接放进 `assets/` 即可，`detect.py` 会优先用它（三级查找：技能内置 → `XIAOYU_WEIGHTS` → 用户缓存）。

## 用法

**单段（人话版输出）**

```bash
python scripts/detect.py "要检测的中文文本"
```

**单段（JSON，便于程序处理）**

```bash
python scripts/detect.py "要检测的中文文本" --json
```

**换聚合方式 / 只要整篇 / 两种口径都给**

```bash
python scripts/detect.py "文本" --agg len      # 加权聚合（默认是等权）
python scripts/detect.py "文本" --no-sent      # 不算句头，长稿快很多
python scripts/detect.py "文本" --json --both  # 同时给出等权与加权两个整篇分
python scripts/detect.py "文本" --win 4        # 句头邻窗改成 ±4 句（默认 ±2）
python scripts/detect.py "文本" --bits 32      # 换 fp32 那档权重（默认 int4）
```

**批量（jsonl，每行 `{"id":"可选","text":"..."}`）**

```bash
python scripts/detect.py --file input.jsonl --out output.jsonl
```

**校验权重完整性**

```bash
python scripts/detect.py "任意文本" --verify
```

## 三档判决

| AI生成概率 | 判决 |
| --- | --- |
| ≥ 90% | **AI特征明显** |
| 50% – 90% | **疑似AI参与** |
| < 50% | **AI特征不明显** |

## 两条红线

1. **整篇分是倾向，不是证据。** 它回答「这篇读起来像不像 AI」，不回答「这篇是谁写的」。
2. **句头不能当定位器。** 拿句分排序找「哪几句是 AI 写的」接近瞎猜。句头只作辅助线索。

## 文档

- `SKILL.md` —— 技能说明、完整用法与执行纪律（agent 实际读取的那份）
- `references/口径与边界.md` —— 两种聚合怎么选、为什么切窗、句头和边界
- `assets/README.md` —— 权重为什么不在仓库里、怎么拉

## 环境

- Python ≥ 3.9，`torch` ≥ 2.0，`transformers` ≥ 4.40；首次下载权重还需 `modelscope`
- 纯 CPU 可跑，无需 GPU

## 许可

Apache-2.0，全文见 `LICENSE`。底座模型 `chinese-roberta-wwm-ext` 的出处与署名见 `NOTICE`。

使用时请保留出处，并**不要把这个概率当作作者身份的判定依据**。
