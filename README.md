# 晓语 · 中文 AI 检测（技能包）

agent 技能形态的中文「AI 生成 / 人写」检测器。当前模型版本 **V3.3**。

给一段中文文本，输出**一个整篇 AI 生成概率**和三档判决，并逐句给出**句头概率**（仅供参考）。

---

## 安装

把本目录放到 agent 的技能目录下（WorkBuddy 对应下面这个路径）：

```
~/.workbuddy/skills/xiaoyu-chinese-ai-detect
```

Windows 对应 `C:\Users\<你的用户名>\.workbuddy\skills\xiaoyu-chinese-ai-detect`。

目录名不影响识别，**技能名取自 `SKILL.md` 里的 `name` 字段**。

### 从魔搭一条命令装（推荐，不用自己找目录）

```bash
pip install modelscope
modelscope skills add fuyi163/xiaoyu-ai-detect-skill \
  --local_dir ~/.agents/skills/xiaoyu-chinese-ai-detect
```

`--local_dir` 后面那个目录名**建议自己带上**。CLI 是把包里文件**直接解到你给的目录根上**，
不会自动套一层同名子目录——所以装多个技能时一定要各给一个独立目录，
否则 `SKILL.md`、`LICENSE`、`README.md` 会互相覆盖。

装完进那个目录跑一句校验，15 行全 OK 就说明整包没缺件：

```bash
cd ~/.agents/skills/xiaoyu-chinese-ai-detect && sha256sum -c SHA256SUMS.txt
```

## 首次使用：装环境 + 拉权重

> **在 Agent 里用的，下面两步都不用你动手。**
> 把稿子直接发给 agent 说「检测这段是不是 AI 写的」就行。第一次跑它会自己发现依赖没装、
> 权重没下，然后装依赖、拉那 60MB 权重，再给你结果——**首次大概要等 6-7 分钟**
> （大头是装 CPU 版 torch 和拉权重，网速好能压到 3 分钟左右），之后每次都是几秒。
> **跳过下面两节，直接去看「用法」**。
>
> 只有**不用 agent、自己敲命令行跑脚本**的人，才需要照下面两步手动做一遍（做过一次就永久有效）。
> 万一是 agent 那边装不上（网络问题），再回来手动走这两步。

### 第 1 步：装 Python 环境（手动跑脚本时）

要求 **Python ≥ 3.9**（实测 3.13 通过），**不需要显卡，纯 CPU 跑**。

```bash
# ① CPU 版 torch —— 这条必须带 --index-url
pip install torch --index-url https://download.pytorch.org/whl/cpu
# ② 加载模型用
pip install transformers
# ③ 只有"从魔搭拉权重"这一步才需要
pip install modelscope
```

> **① 那条的 `--index-url` 千万别省**。不带的话 pip 会装 CUDA 版 torch，
> 体积从约 200MB 涨到 2GB 以上，而且这台机器没 GPU 的话装了也用不上。

装完自查一句，能打出两个版本号就是好了：

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

**下载慢**就把 ②③ 换成国内镜像（① 不要换，CPU 版 torch 只有官方源有）：

```bash
pip install transformers modelscope -i https://mirrors.cloud.tencent.com/pypi/simple
```

> 实测跑通的组合：Python 3.13.14 ／ torch 2.14.0+cpu ／ transformers 5.17.0 ／
> modelscope 1.40.2 ／ safetensors 0.8.0。装完整个环境约 500MB。

### 第 2 步：拉权重（手动跑脚本时）

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

> **字数建议：300 字以上。**
> 模型本身**没有最小字数限制**——非空就能跑，不会因为太短而报错或拒绝。
> 但太短的文本判不准：几十个字，模型拿不到足够的文风特征，出来的分数参考价值很低，
> **而且它不会告诉你「这次判得不准」**。一句话的稿子可以测，但别拿那个数字当结论。

### 在 Agent 里直接用（推荐）

装好之后根本不用敲命令，直接在对话里说人话：

> 帮我检测下这段是不是 AI 写的：<粘贴正文>

agent 会自己调 `scripts/detect.py`，回给你**整篇概率、三档判决和句头分布**，你不用记参数。

**能听懂的说法**（`SKILL.md` 里已声明的触发词）：

> 检测这段是不是 AI 写的 ｜ AI 味检测 ｜ AI 文本鉴别 ｜ 判 AI ｜ 这段稿子像不像 AI 写的 ｜ 人写的还是 AI 写的 ｜ AI生成概率

**常见想法对照**——按左边这么说，agent 会自己换算成右边的参数：

| 你想干嘛 | 就这么说 | agent 实际干的 |
|---|---|---|
| 判一篇稿 | 检测这段是不是 AI 写的：<正文> | 默认整篇 + 逐句 |
| 只要整篇结论（长稿快很多） | 只给整篇概率，不用逐句 | `--no-sent` |
| 换加权口径 | 用加权口径再算一遍 | `--agg len` |
| 两个口径都要 | 两种口径都给我 | `--both` |
| 批量判一堆 | 把这些逐条判一下，结果存 `out.jsonl` | `--file` / `--out` |
| 换精度档 | 用 int8 那档再判一遍 | `--bits 8` |

**三个坑先记住**：

1. **刚装完要刷新会话**（重启 agent——比如 WorkBuddy——或新开一个对话）才认得到新技能——技能列表是启动时加载的，光把目录放进去它不会自动发现。
2. **长文本别直接粘**。存成 `.txt`，把路径告诉 agent：「判一下 `C:\稿件\0824.txt`」。粘正文吃 token，超长还可能被截断。
3. **报「找不到权重」= 还没拉权重**，回去跑一次 `python scripts/download.py`。

> 技能名取自 `SKILL.md` 的 `name` 字段 **`晓语-中文AI检测`**，跟目录名 `xiaoyu-chinese-ai-detect` 不是一回事——两者不影响使用，但对话里叫它「晓语检测」最省事。

### 命令行

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

## 许可

Apache-2.0，全文见 `LICENSE`。底座模型 `chinese-roberta-wwm-ext` 的出处与署名见 `NOTICE`。

使用时请保留出处，并**不要把这个概率当作作者身份的判定依据**。
