# -*- coding: utf-8 -*-
"""晓语 · 中文 AI 检测 —— 输出一个整篇概率 + 每句话的句头概率。

判决口径（唯一一套尺子，三档）
  AI生成概率 ≥ 90%   → AI特征明显
  50% ≤ 概率 < 90%  → 疑似AI参与
  概率 < 50%        → AI特征不明显

整篇概率：模型一次只能吃 512 token，长文切窗（500 字/窗、步长 400）后聚合。
  等权 eq  ：Σp / n            每窗一票（默认）
  加权 len ：Σ(len·p) / Σlen   按窗口字数加权
  两种口径概率刻度不同，不要跨口径比大小。

句头概率：以该句为中心取上下文窗打分，**仅供参考**，不代表该句由 AI 生成，
也不具备篇内定位能力。

用法
  python detect.py "要检测的文本"
  python detect.py "要检测的文本" --agg len       # 用加权聚合
  python detect.py "要检测的文本" --json
  python detect.py --file in.jsonl --out out.jsonl
  python detect.py "要检测的文本" --no-sent        # 只要整篇分
"""
import argparse
import io
import json
import os
import re
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SKILL, "assets")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "xiaoyu-ai-detect")

WEIGHTS = "int4_packed.pt"
WEIGHT_SHA256 = "80a2505fbc922ab33cc4e9875dd87b626225daba83bf980316c1ad07b045f2d9"
MODEL_ID = "fuyi163/xiaoyu-chinese-ai-detect"

# 三档权重。默认用 int4（60.2MB，最快）；想要更高保真可以切 int8 或 fp32。
#   kind="packed" → 仓库自带打包件，需 unpack_blob 反量化
#   kind="hf"     → 标准 safetensors 目录，from_pretrained 直读
# 分数会随档位有细微差别（int4 边界样本的概率会漂移），同一批稿子换档请重跑、别跨档比大小。
TIERS = {
    4:  {"name": "int4_packed.pt", "kind": "packed", "mb": 60.2,
         "sha256": "80a2505fbc922ab33cc4e9875dd87b626225daba83bf980316c1ad07b045f2d9",
         "label": "int4 轻量档（默认）"},
    8:  {"name": "int8_packed.pt", "kind": "packed", "mb": 101.0,
         "sha256": "bf6456105bbdd92c38a41671cf53882d90731129c24a4fbb6a79197a0a8b9687",
         "label": "int8 平衡档"},
    32: {"name": "fp32", "kind": "hf", "mb": 390.1,
         "sha256": "75e56acd525cc7452f70741c898d4ad42dd9304d5ad6df8f45a4726f237d6ea7",
         "label": "fp32 原始精度档"},
}
TIER_ORDER = (4, 8, 32)

G = 128                      # 量化分组大小
MAXLEN = 512
LIM = MAXLEN - 2
FULL_WIN = 500
FULL_STRIDE = 400
WIN = 2                      # 句头邻窗 ±2 句
MAX_SENT = 200
SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")

T_AI = 0.9
T_SUS = 0.5
V_AI = "AI特征明显"
V_SUS = "疑似AI参与"
V_LOW = "AI特征不明显"

AGG_MODES = ("eq", "len")
AGG_DEFAULT = "eq"

HEAD_NOTE = ("句头概率仅供参考：它衡量的是「这一句所在的上下文读起来像不像 AI」，"
             "不代表该句由 AI 生成，也不能用来在一篇文章里排序定位。")


# ---------------------------------------------------------------- 权重

def _tier_path(base, bits):
    """某档在某个基准目录下的期望位置。fp32 是目录，其余是单文件。"""
    t = TIERS[bits]
    p = os.path.join(base, t["name"])
    return (os.path.isdir(p) if t["kind"] == "hf" else os.path.isfile(p)), p


def find_weights(bits=None):
    """定位权重。bits=None 时按 int4 → int8 → fp32 找第一个存在的。

    返回 (路径, 来源说明, 实际档位)。
    """
    order = [bits] if bits else list(TIER_ORDER)
    bases = [(ASSETS, "技能内置")]
    env = os.environ.get("XIAOYU_WEIGHTS")
    if env:
        bases.append((env, "环境变量 XIAOYU_WEIGHTS"))
    bases.append((CACHE, "用户缓存"))

    tried = []
    for b in order:
        for base, src in bases:
            ok, p = _tier_path(base, b)
            tried.append(p)
            if ok:
                return p, src, b

    if bits:
        detail = ("指定了 --bits %d（%s），但没找到 %s\n已查找：\n         %s"
                  % (bits, TIERS[bits]["label"], TIERS[bits]["name"],
                     "\n         ".join(tried)))
        hint = "  python %s --bits %d\n" % (os.path.join(SKILL, "scripts", "download.py"), bits)
    else:
        detail = ("三档权重都没找到（int4 / int8 / fp32）\n已查找：\n         %s"
                  % "\n         ".join(tried))
        hint = "  python %s\n" % os.path.join(SKILL, "scripts", "download.py")
    sys.stderr.write(
        "\n[晓语] 找不到权重\n%s\n\n"
        "拉取方式：\n"
        "  1) %s          ← 从魔搭自动下载（默认 int4；换档加 --bits 8 或 --bits 32）\n"
        "  2) 手动下载放到上面任一目录\n"
        "  3) 模型主页：https://modelscope.cn/models/%s\n"
        % (detail, hint.rstrip(), MODEL_ID))
    sys.exit(2)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def unpack_blob(blob, torch):
    """打包件 → fp32 state_dict。规则与魔搭模型仓库 fuyi163/xiaoyu-chinese-ai-detect
    里 quant/pack_quant.py 的打包逻辑逐位同源，本文件已内联实现，无需另装。"""
    out = {}
    for k, e in blob["pack"].items():
        if e["pk"] == "lin":
            if "bytes" in e:                      # INT4 位打包
                b, shape = e["bytes"], tuple(e["shape"])
                lo = (b & 0x0F).to(torch.int16) - 8
                hi = (b >> 4).to(torch.int16) - 8
                u = torch.empty(shape[0], shape[1], dtype=torch.int16)
                u[:, 0::2] = lo
                u[:, 1::2] = hi
                q = u.to(torch.int8)
            else:
                q = e["q"]
            o, i = q.shape
            ng = e["scale"].shape[1]
            out[k] = (q.float().reshape(o, ng, G) * e["scale"].unsqueeze(-1)).reshape(o, i)
        elif e["pk"] == "emb":
            q, sc = e["q"], e["scale"]
            r, c = q.shape
            out[k] = (q.float().reshape(r, sc.shape[1], G) * sc.unsqueeze(-1)).reshape(r, c)
        else:
            out[k] = e["w"]
    return out


_C = {}
_PENDING_BITS = None      # detect(bits=...) 设定，供内部无参 build() 继承


def build(verify=False, bits=None):
    global _PENDING_BITS
    if _C:
        return _C
    if bits is None:
        bits = _PENDING_BITS
    else:
        _PENDING_BITS = bits
    try:
        import torch
        from transformers import (AutoConfig, AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except ImportError as e:
        sys.stderr.write(
            "[晓语] 缺依赖：%s\n"
            "请先安装：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install transformers\n" % e)
        sys.exit(2)

    torch.set_grad_enabled(False)
    path, src, use_bits = find_weights(bits)
    tier = TIERS[use_bits]
    if verify:
        if tier["kind"] == "hf":
            import hashlib
            f = os.path.join(path, "model.safetensors")
            got = hashlib.sha256(open(f, "rb").read()).hexdigest() if os.path.isfile(f) else None
            if got is None:
                raise RuntimeError("fp32 目录里没有 model.safetensors：%s" % path)
        else:
            got = sha256(path)
        if got != tier["sha256"]:
            raise RuntimeError("权重 sha256 不符（%s）\n  期望 %s\n  实测 %s"
                               % (tier["label"], tier["sha256"], got))
        sys.stderr.write("[权重校验] %s sha256 OK（来源：%s）\n" % (tier["label"], src))

    tok = AutoTokenizer.from_pretrained(os.path.join(ASSETS, "tok"))
    probe = tok("测试一句话")["input_ids"]
    if probe.count(tok.unk_token_id) >= len(probe) / 2:
        raise RuntimeError("词表异常（全 UNK）—— assets/tok 需为原始词表")

    if tier["kind"] == "hf":
        # 标准 safetensors 目录：直接按 HF 方式加载，不需要 unpack_blob
        model = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        cfg = AutoConfig.from_pretrained(os.path.join(ASSETS, "model_cfg"))
        model = AutoModelForSequenceClassification.from_config(cfg)
        blob = torch.load(path, weights_only=False, map_location="cpu")
        missing, unexpected = model.load_state_dict(unpack_blob(blob, torch), strict=False)
        if missing or unexpected:
            raise RuntimeError("权重不匹配 missing=%s unexpected=%s"
                               % (list(missing), list(unexpected)))
    model.eval()

    _C.update(torch=torch, tok=tok, model=model, path=path, src=src, bits=use_bits)
    return _C


def _p(torch, logits):
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, -1)[:, 1]


# ---------------------------------------------------------------- 打分

def split_sents(text):
    out, pos = [], 0
    for piece in SENT_SPLIT.split(text):
        if piece:
            out.append((pos, pos + len(piece), piece))
            pos += len(piece)
    return [(s, e, t) for s, e, t in out if t.strip()]


def build_windows(text):
    segs, i, n = [], 0, len(text)
    while i < n:
        seg = text[i:i + FULL_WIN]
        if len(seg.strip()) < 30:
            break
        segs.append(seg)
        if i + FULL_WIN >= n:
            break
        i += FULL_STRIDE
    return segs


def aggregate(ps, lens, how="eq"):
    if not ps:
        return None
    if how == "len":
        w = float(sum(lens))
        if w > 0:
            return sum(l * p for l, p in zip(lens, ps)) / w
    return sum(ps) / len(ps)


def score_full(text, agg=AGG_DEFAULT, batch=8):
    if agg not in AGG_MODES:
        raise ValueError("未知聚合方式 %s（可选 %s）" % (agg, " / ".join(AGG_MODES)))
    c = build()
    torch, tok, model = c["torch"], c["tok"], c["model"]

    ntok = len(tok(text, add_special_tokens=False)["input_ids"])
    segs = [] if ntok <= LIM else build_windows(text)

    if len(segs) <= 1:
        with torch.no_grad():
            enc = tok(text, truncation=True, max_length=MAXLEN, return_tensors="pt")
            enc = {k: v for k, v in enc.items()
                   if k in ("input_ids", "attention_mask", "token_type_ids")}
            p = float(_p(torch, model(**enc).logits)[0])
        return {"p": p, "p_eq": p, "p_len": p, "n_win": 1,
                "scope": "single"}

    lens = [len(x) for x in segs]
    ps = []
    with torch.no_grad():
        for k in range(0, len(segs), batch):
            enc = tok(segs[k:k + batch], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt")
            enc = {kk: v for kk, v in enc.items()
                   if kk in ("input_ids", "attention_mask", "token_type_ids")}
            ps += _p(torch, model(**enc).logits).float().cpu().tolist()

    p_eq = aggregate(ps, lens, "eq")
    p_len = aggregate(ps, lens, "len")
    return {"p": p_eq if agg == "eq" else p_len,
            "p_eq": p_eq, "p_len": p_len, "n_win": len(ps), "scope": "full"}


def score_sents(text, sents, win=WIN, batch=16):
    c = build()
    torch, tok, model = c["torch"], c["tok"], c["model"]
    n = len(sents)
    ins, keep, dropped = [], [], 0
    for i, (s, e, _) in enumerate(sents):
        lo = sents[max(0, i - win)][0]
        hi = sents[min(n - 1, i + win)][1]
        pre = tok(text[lo:s], add_special_tokens=False)["input_ids"]
        if len(pre) >= LIM:        # 该句之前上下文已超窗 → 当前句必被截掉
            dropped += 1
            continue
        ins.append(text[lo:hi])
        keep.append(i)
    ps = [None] * n
    if ins:
        with torch.no_grad():
            for i in range(0, len(ins), batch):
                enc = tok(ins[i:i + batch], padding=True, truncation=True,
                          max_length=MAXLEN, return_tensors="pt")
                enc = {k: v for k, v in enc.items()
                       if k in ("input_ids", "attention_mask", "token_type_ids")}
                vals = _p(torch, model(**enc).logits).float().cpu().tolist()
                for k, v in enumerate(vals):
                    ps[keep[i + k]] = v
    return ps, dropped


def verdict_of(p):
    if p is None:
        return None
    if p >= T_AI:
        return V_AI
    if p >= T_SUS:
        return V_SUS
    return V_LOW


def pct(p, nd=2):
    if p is None:
        return None
    return ("%." + str(nd) + "f%%") % (100.0 * p)


# ---------------------------------------------------------------- 主入口

def detect(text, agg=AGG_DEFAULT, sent=True, max_sent=MAX_SENT, both=False, win=WIN,
           bits=None):
    """文本 → dict。整篇概率 + 每句句头概率。

    bits: 4 / 8 / 32，指定用哪一档权重；None = 按 int4 → int8 → fp32 取第一个可用的。
    """
    global _PENDING_BITS
    _PENDING_BITS = bits
    text = text if isinstance(text, str) else (text or "")
    if not text.strip():
        raise ValueError("文本为空")

    d = score_full(text, agg)
    res = {
        "agg": agg,
        "bits": _C.get("bits"),
        "doc": {"ai_prob": d["p"], "ai_prob_pct": pct(d["p"]),
                "verdict": verdict_of(d["p"]), "n_win": d["n_win"],
                "chars": len(text)},
        "sentences": [],
        "summary": {"n_sent": 0, "n_ai": 0, "n_sus": 0, "n_low": 0},
        "win": win,
    }
    if both:
        res["doc"]["ai_prob_eq"] = d["p_eq"]
        res["doc"]["ai_prob_len"] = d["p_len"]

    if not sent:
        return res

    all_sents = split_sents(text)
    if not all_sents:
        return res

    truncated = len(all_sents) > max_sent
    use = all_sents[:max_sent] if truncated else all_sents
    ps, dropped = score_sents(text, use, win=win)

    rows, n_ai, n_sus, n_low, n_skip = [], 0, 0, 0, 0
    for j, (s, e, raw) in enumerate(use):
        v = verdict_of(ps[j])
        if v is None:
            n_skip += 1
        elif v == V_AI:
            n_ai += 1
        elif v == V_SUS:
            n_sus += 1
        else:
            n_low += 1
        rows.append({"i": j + 1, "text": raw.strip(), "ai_prob": ps[j],
                     "ai_prob_pct": pct(ps[j]), "verdict": v})

    res["sentences"] = rows
    summary = {"n_sent": len(use), "n_ai": n_ai, "n_sus": n_sus, "n_low": n_low}
    if n_skip:
        summary["n_skipped"] = n_skip
    if dropped:
        summary["n_dropped"] = dropped
    if truncated:
        summary["n_sent_total"] = len(all_sents)
        summary["truncated_note"] = "长稿保护：共 %d 句，只对前 %d 句打分" % (
            len(all_sents), len(use))
    res["summary"] = summary
    res["note"] = HEAD_NOTE
    return res


MIN_CHARS = 10          # 短于此的碎片不单独加标记（仍参与邻窗上下文）


def _tier_tag():
    """当前用的档位短名，打在输出末尾，避免用户记不清自己在看哪档的分数。"""
    b = _C.get("bits")
    if not b:
        return "未加载"
    return {4: "int4", 8: "int8", 32: "fp32"}.get(b, str(b))


def render(res, min_chars=MIN_CHARS):
    """人话版：整篇一行 + 句头概率列表。"""
    d = res["doc"]
    lines = ["整篇 AI生成概率 %s —— %s" % (d["ai_prob_pct"], d["verdict"]), ""]

    sents = res["sentences"]
    s = res["summary"]
    mine = res.get("win", WIN)
    how = "加权" if res["agg"] == "len" else "等权"

    if sents:
        lines.append("句头概率（仅供参考，不代表该句由 AI 生成）")
        for r in sents:
            t = r["text"]
            if len(t) > 46:
                t = t[:46] + "…"
            lines.append("  %8s  %s" % (r["ai_prob_pct"] or "—", t))
        lines.append("")
        lines.append("─" * 60)
        tail = "共 %d 句 ｜ AI特征明显 %d ｜ 疑似AI参与 %d ｜ AI特征不明显 %d" % (
            s["n_sent"], s["n_ai"], s["n_sus"], s["n_low"])
        if s.get("n_skipped"):
            tail += " ｜ %d 句超长未打分" % s["n_skipped"]
        if s.get("truncated_note"):
            tail += "\n" + s["truncated_note"]
        lines.append(tail)
        lines.append("%s聚合 ｜ %d 窗 ｜ 句头邻窗 ±%d 句 ｜ 判决线 50%% / 90%% ｜ 权重 %s"
                     % (how, d["n_win"], mine, _tier_tag()))
        lines.append("句头仅供参考，不作篇内定位，也不代表该句由 AI 生成。")
    else:
        # 未算句头时不要打「共 0 句」和句头免责声明，那是在报告一件没做的事
        lines.append("─" * 60)
        lines.append("%s聚合 ｜ %d 窗 ｜ 判决线 50%% / 90%% ｜ 权重 %s ｜ 本次只出整篇分，未算句头"
                     % (how, d["n_win"], _tier_tag()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="晓语 · 中文 AI 检测：输出整篇概率与句头概率")
    ap.add_argument("text", nargs="?", default=None, help="单段待检文本")
    ap.add_argument("--file", default=None, help="批量 jsonl 输入（每行含 text 字段）")
    ap.add_argument("--out", default=None, help="输出 jsonl（缺省 stdout）")
    ap.add_argument("--text-key", default="text", help="批量输入的文本字段名")
    ap.add_argument("--id-key", default="id", help="批量输入的 id 字段名")
    ap.add_argument("--agg", default=AGG_DEFAULT, choices=list(AGG_MODES),
                    help="整篇聚合方式：eq 等权（默认）/ len 加权")
    ap.add_argument("--win", type=int, default=None, help="句头邻窗句数（默认 ±2）")
    ap.add_argument("--max-sent", type=int, default=MAX_SENT, help="句头最多算多少句")
    ap.add_argument("--no-sent", action="store_true", help="只出整篇概率")
    ap.add_argument("--json", action="store_true", help="单段时输出 JSON 而非人话版")
    ap.add_argument("--both", action="store_true", help="同时给出等权与加权两个整篇分")
    ap.add_argument("--verify", action="store_true", help="校验权重 sha256")
    ap.add_argument("--bits", type=int, default=None, choices=[4, 8, 32],
                    help="用哪一档权重：4 int4(默认) / 8 int8 / 32 fp32。"
                         "换档需先 download.py --bits N")
    a = ap.parse_args()

    if not a.text and not a.file:
        ap.error("需要位置参数文本或 --file jsonl")

    build(verify=a.verify, bits=a.bits)   # 先加载，把权重提示打在结果之前

    sent = not a.no_sent
    win = a.win if a.win is not None else WIN
    if win < 0:
        ap.error("--win 不能为负数（0 表示只看本句，窗内不含邻句）")
    if a.both and not a.json:
        # 人话版只显示选定的那一种聚合，--both 的另一个分只在 json 里
        sys.stderr.write("[晓语] 提示：--both 的两个整篇分只出现在 --json 输出中；"
                         "人话版仍按 --agg 显示一个\n")
    if a.text:
        r = detect(a.text, agg=a.agg, sent=sent, max_sent=a.max_sent, both=a.both,
                   win=win, bits=a.bits)
        if a.json:
            txt = json.dumps(r, ensure_ascii=False, indent=2) + "\n"
            if a.out:
                with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                    f.write(txt)
            else:
                sys.stdout.write(txt)
        else:
            sys.stdout.write(render(r) + "\n")
        return

    items = []
    for ln, line in enumerate(open(a.file, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        txt = o.get(a.text_key)
        if txt is None:
            raise KeyError("第 %d 行缺字段 %s" % (ln + 1, a.text_key))
        items.append({"id": o.get(a.id_key, "n%d" % ln), "text": txt})

    out_f = open(a.out, "w", encoding="utf-8", newline="\n") if a.out else sys.stdout
    for c in items:
        r = detect(c["text"], agg=a.agg, sent=sent, max_sent=a.max_sent,
                   both=a.both, win=win, bits=a.bits)
        rec = {"id": c["id"]}
        rec.update(r)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    if a.out:
        out_f.close()
        sys.stderr.write("[晓语] 完成 %d 条 → %s\n" % (len(items), a.out))


if __name__ == "__main__":
    main()
