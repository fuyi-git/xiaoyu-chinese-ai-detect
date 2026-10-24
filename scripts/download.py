# -*- coding: utf-8 -*-
"""从魔搭下载权重到本地缓存。技能自带权重时不需要跑这个。

用法
  python download.py                # 下载 int4（约 63MB，默认最快）
  python download.py --bits 8       # 下载 int8（约 101MB，精度更接近原始）
  python download.py --bits 32      # 下载 fp32（约 390MB，原始精度，标准 safetensors）
  python download.py --check        # 只看本地有哪几档，并逐档校验 sha256

说明
  - 下载后一定校验 sha256，不符即报错，不会静默通过
  - 网络失败重试 3 次（1s / 3s / 9s 退避）；仍失败给出人话提示，不抛裸 traceback
  - 若报 404 / NotFound，通常是模型仓库暂不可用（未发布或迁移），不是你的操作问题
  - 三档是同一份权重的不同体积，分数会有些微差别；同一批稿子请固定用同一档，别跨档比大小
"""
import argparse
import os
import shutil
import sys
import time

MODEL_ID = "fuyi163/xiaoyu-chinese-ai-detect"
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SKILL, "assets")
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "xiaoyu-ai-detect")

# 档位定义：bits -> (仓库内路径, 本地落点相对名, sha256, 类型, 体积描述, 档名)
TIERS = {
    4: ("quant/int4_packed.pt", "int4_packed.pt",
        "80a2505fbc922ab33cc4e9875dd87b626225daba83bf980316c1ad07b045f2d9",
        "packed", "60.2 MB", "int4 轻量档（默认）"),
    8: ("quant/int8_packed.pt", "int8_packed.pt",
        "bf6456105bbdd92c38a41671cf53882d90731129c24a4fbb6a79197a0a8b9687",
        "packed", "101.0 MB", "int8 平衡档"),
    32: ("model.safetensors", "fp32",
         "75e56acd525cc7452f70741c898d4ad42dd9304d5ad6df8f45a4726f237d6ea7",
         "hf", "390.1 MB", "fp32 原始精度档"),
}
ORDER = (4, 8, 32)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def local_path(bits, root):
    return os.path.join(root, TIERS[bits][1])


def exists(bits, root):
    p = local_path(bits, root)
    return os.path.isdir(p) if TIERS[bits][3] == "hf" else os.path.isfile(p)


def payload(bits, root):
    """真正要算哈希的那个文件。fp32 是目录，实际校验其中的 model.safetensors。"""
    p = local_path(bits, root)
    if TIERS[bits][3] == "hf":
        return os.path.join(p, "model.safetensors")
    return p


def check_one(bits, root):
    """返回 (是否存在, 结论)。结论：OK / MISMATCH:<实测> / 缺件 / 读不到。"""
    if not exists(bits, root):
        return False, None
    f = payload(bits, root)
    if not os.path.isfile(f):
        return True, "缺件（目录在但 %s 不在）" % os.path.basename(f)
    try:
        got = sha256(f)
    except OSError as e:
        return True, "读不到（%s）" % str(e)[:60]
    return True, ("OK" if got == TIERS[bits][2] else "MISMATCH:%s" % got)


def human(bits, root):
    p = local_path(bits, root)
    if not exists(bits, root):
        return "不存在"
    f = payload(bits, root)
    mb = os.path.getsize(f) / 1048576.0 if os.path.isfile(f) else 0.0
    _, verdict = check_one(bits, root)
    tag = "校验通过" if verdict == "OK" else (verdict or "无法校验")
    return "%.1f MB  %s  %s" % (mb, tag, p)


def do_check(root):
    print("本地权重状态（缓存目录 %s）" % root)
    any_good = False
    for b in ORDER:
        _, verdict = check_one(b, root)
        any_good = any_good or (verdict == "OK")
        mark = "OK  " if verdict == "OK" else ("    " if verdict is None else "!!  ")
        print("  %s %-20s %s" % (mark, TIERS[b][5], human(b, root)))
    # 技能内置的也算数（离线自包含分发时会放这里）
    for b in ORDER:
        if TIERS[b][3] == "hf":
            continue
        p = os.path.join(ASSETS, TIERS[b][1])
        if os.path.isfile(p):
            ok = (sha256(p) == TIERS[b][2])
            any_good = any_good or ok
            print("  %s %-20s %.1f MB  %s  技能内置 %s"
                  % ("OK  " if ok else "!!  ", TIERS[b][5], os.path.getsize(p) / 1048576.0,
                     "校验通过" if ok else "哈希不符", p))
    if any_good:
        print("\n至少一档可用。detect.py 默认按 int4 → int8 → fp32 取第一个找到的；"
              "要指定档位加 --bits 4|8|32。")
    else:
        print("\n一档都没有。跑一次：python download.py  （或 --bits 8 / --bits 32）")


def manual_hint(bits):
    return ("手动下载：打开 https://modelscope.cn/models/%s\n"
            "          下 %s，放到 %s（或任意目录后设环境变量 XIAOYU_WEIGHTS 指向该目录）"
            % (MODEL_ID, TIERS[bits][0], local_path(bits, DEFAULT_DIR)))


def do_download(bits, root, force):
    repo_rel, _, want, kind, size, label = TIERS[bits]
    dst = local_path(bits, root)

    if exists(bits, root) and not force:
        _, verdict = check_one(bits, root)
        if verdict == "OK":
            print("已存在且校验通过：%s（%s）" % (dst, size))
            return
        print("已存在但%s：%s" % (verdict or "无法校验", dst))
        print("重下请加 --force（会自动覆盖）")
        return

    try:
        from modelscope import snapshot_download
    except ImportError:
        sys.exit("缺 modelscope：pip install modelscope\n" + manual_hint(bits))

    os.makedirs(root, exist_ok=True)
    pats = [repo_rel]
    if kind == "hf":
        pats.append("config.json")      # fp32 目录需要 config 才能被 from_pretrained 直读
    print("从 %s 拉取 %s（%s）…" % (MODEL_ID, repo_rel, label))

    d, last = None, None
    for attempt in range(3):
        try:
            d = snapshot_download(MODEL_ID, allow_patterns=pats)
            break
        except KeyboardInterrupt:
            sys.exit("已中断")
        except Exception as e:
            last = e
            print("  第 %d 次失败：%s" % (attempt + 1, str(e)[:160]))
            if attempt < 2:
                wait = 3 ** attempt
                print("  %d 秒后重试…" % wait)
                time.sleep(wait)

    if d is None:
        msg = str(last).lower()
        print("\n下载失败，三次都没成功。")
        if any(k in msg for k in ("404", "notfound", "not found", "notexist", "does not exist")):
            print("远端返回「找不到」——通常是模型仓库暂不可用（尚未发布或正在迁移），")
            print("不是你的操作问题。稍后重试，或按下面方式手动下载：")
        elif any(k in msg for k in ("proxy", "timeout", "timed out", "connection", "ssl")):
            print("看着像网络/代理问题。若环境里设了 HTTP(S)_PROXY，试试先清掉再跑：")
            print("  Linux/macOS:  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy")
            print("  Windows CMD:  set HTTP_PROXY=  &  set HTTPS_PROXY=")
        else:
            print("原因见上面的报错。手动下载方式：")
        print(manual_hint(bits))
        sys.exit(1)

    src = os.path.join(d, *repo_rel.split("/"))
    if not os.path.isfile(src):
        sys.exit("下载完成但没找到 %s（目录 %s）\n%s" % (repo_rel, d, manual_hint(bits)))

    if kind == "hf":
        os.makedirs(dst, exist_ok=True)
        shutil.copyfile(src, os.path.join(dst, "model.safetensors"))
        cfg_src = os.path.join(d, "config.json")
        if os.path.isfile(cfg_src):
            shutil.copyfile(cfg_src, os.path.join(dst, "config.json"))
        else:
            print("  !! 没拿到 config.json，fp32 档需要它才能加载；"
                  "可从仓库根目录手动补一份到 %s" % dst)
    else:
        shutil.copyfile(src, dst)

    got = sha256(payload(bits, root))
    if got != want:
        sys.exit("sha256 不符，文件已损坏，已保留在原处供排查：\n"
                 "  期望 %s\n  实测 %s\n删掉后重跑本脚本" % (want, got))
    print("完成 → %s（%s，sha256 校验通过）" % (dst, size))
    if bits != 4:
        print("提示：用这一档要显式加 --bits %d，否则 detect.py 按 int4 → int8 → fp32 取第一个找到的。" % bits)


def main():
    ap = argparse.ArgumentParser(
        description="下载晓语中文 AI 检测的权重（int4 / int8 / fp32 三档）")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 32],
                    help="下载哪一档：4 int4(默认,60MB) / 8 int8(101MB) / 32 fp32(390MB)")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="落点目录")
    ap.add_argument("--check", action="store_true", help="只查本地各档状态（含 sha256 校验）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    a = ap.parse_args()

    if a.check:
        do_check(a.dir)
        return
    do_download(a.bits, a.dir, a.force)


if __name__ == "__main__":
    main()
